"""공통 유틸: 설정 로드, parquet 청크/샤드 라이터, 체크포인트, 타임스탬프 처리."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hf_dataset_path(repo_id: str, path_in_repo: str, revision: str | None = None) -> str:
    repo = f"{repo_id}@{revision}" if revision and revision != "main" else repo_id
    return f"datasets/{repo}/{path_in_repo}"


def stream_jsonl(repo_id: str, path_in_repo: str, revision: str | None = None):
    """HF Hub의 raw jsonl을 다운로드 없이 라인 단위 스트리밍.

    datasets 라이브러리는 이 레포의 구식 로딩 스크립트를 거부하므로(>=3.0),
    HfFileSystem으로 파일을 직접 연다. (decisions.md D-007)
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open(hf_dataset_path(repo_id, path_in_repo, revision), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_jsonl_offsets(repo_id: str, path_in_repo: str, start_offset: int = 0,
                       max_reconnect: int = 20, revision: str | None = None):
    """HF의 raw jsonl을 바이트 오프셋과 함께 스트리밍(재개 가능 + 자가 복구).

    각 라인을 파싱해 (obj, next_offset)을 yield한다. next_offset은 그 라인 "다음"의
    바이트 위치(=다음 라인 시작)로, 체크포인트에 저장하면 seek로 정확히 이어받을 수 있다.
    HfFileSystem 파일은 HTTP range 요청으로 seek를 지원하므로 이미 받은 부분을
    재다운로드하지 않는다. (resumable ingestion)

    네트워크 끊김/httpx 클라이언트 종료 같은 일시 오류가 나면, 연결을 새로 열고
    마지막 라인 경계(pos)로 seek해 같은 프로세스 안에서 이어서 읽는다(자가 복구).
    max_reconnect회를 넘으면 예외를 올려 상위(런너 재시도/체크포인트 재개)로 넘긴다.
    """
    from huggingface_hub import HfFileSystem

    full = hf_dataset_path(repo_id, path_in_repo, revision)
    pos = start_offset          # 다음에 읽을 라인의 시작 오프셋(항상 라인 경계)
    reconnects = 0
    while True:
        try:
            fs = HfFileSystem()
            f = fs.open(full, "rb")
            try:
                if pos:
                    f.seek(pos)
                while True:
                    line = f.readline()
                    if not line:
                        return                       # EOF: 정상 종료
                    pos = f.tell()
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    yield obj, pos
            finally:
                try:
                    f.close()
                except Exception:
                    pass
        except Exception as e:                       # 네트워크/HTTP 일시 오류 → 재연결
            reconnects += 1
            if reconnects > max_reconnect:
                raise
            wait = min(60, 2 ** min(reconnects, 6))
            print(f"  [stream] 연결 오류({type(e).__name__}), {wait}s 후 재연결 "
                  f"@ offset {pos:,} (재연결 {reconnects}/{max_reconnect})", flush=True)
            time.sleep(wait)
            continue


def save_ckpt(path: Path, data: dict):
    """체크포인트를 원자적으로 저장(tmp 작성 → fsync → rename). 전원 차단 대비."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_ckpt(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


class ShardSink:
    """rows를 샤드 parquet(part0000, part0001, ...)로 나눠 쓰고, 완료 시 단일
    canonical parquet으로 병합한다. 샤드 경계를 체크포인트와 맞추면 중단/재개 시
    "완료된 샤드"만 신뢰하면 되므로 전원 차단에도 강건하다.

    불변식: shard_idx = '완전히 커밋된 샤드 수' = '다음에 쓸 샤드 인덱스'.
    roll()로 현재 샤드를 닫은 뒤에만 shard_idx가 증가하고, 그 직후 체크포인트를 저장한다.
    """

    def __init__(self, canonical_path: Path, chunk_rows: int = 200_000,
                 empty_schema: pa.Schema | None = None):
        self.canonical = Path(canonical_path)
        self.dir = self.canonical.parent
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem = self.canonical.stem
        self.chunk_rows = chunk_rows
        self.empty_schema = empty_schema
        self.shard_idx = 0
        self._buf: list[dict] = []
        self._writer: pq.ParquetWriter | None = None

    def _shard_path(self, i: int) -> Path:
        return self.dir / f"{self.stem}__part{i:05d}.parquet"

    def _shard_index(self, path: Path) -> int | None:
        prefix = f"{self.stem}__part"
        if not path.stem.startswith(prefix):
            return None
        try:
            return int(path.stem.removeprefix(prefix))
        except ValueError:
            return None

    def resume_from(self, shard_idx: int):
        self.shard_idx = shard_idx
        # Shards at or after the checkpoint index were not committed. Remove them
        # before rewriting that index so stale partial shards cannot be finalized.
        for shard in self.dir.glob(f"{self.stem}__part*.parquet"):
            idx = self._shard_index(shard)
            if idx is not None and idx >= shard_idx:
                shard.unlink()

    def add(self, row: dict):
        self._buf.append(row)
        if len(self._buf) >= self.chunk_rows:
            self._flush_buf()

    def _flush_buf(self):
        if not self._buf:
            return
        table = pa.Table.from_pylist(self._buf)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._shard_path(self.shard_idx),
                                            table.schema, compression="zstd")
        else:
            table = table.cast(self._writer.schema)
        self._writer.write_table(table)
        self._buf = []

    def roll(self):
        """현재 샤드를 닫고(footer 작성) shard_idx를 advance. 체크포인트 경계에서 호출."""
        self._flush_buf()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self.shard_idx += 1

    def finalize(self) -> int:
        """마지막 샤드를 닫고 모든 샤드를 canonical로 병합 후 샤드 삭제. 총 행 수 반환."""
        self.roll()
        shards = sorted(self.dir.glob(f"{self.stem}__part*.parquet"))
        if shards:
            import duckdb
            con = duckdb.connect()
            files = ", ".join(f"'{s}'" for s in shards)
            tmp = self.canonical.with_name(self.canonical.name + ".tmp")
            if tmp.exists():
                tmp.unlink()
            con.execute(f"""COPY (SELECT * FROM read_parquet([{files}], union_by_name=true))
                            TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
            total = con.execute(f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
            con.close()
            os.replace(tmp, self.canonical)
            for s in shards:
                s.unlink()
            return total
        if self.empty_schema is not None:
            tmp = self.canonical.with_name(self.canonical.name + ".tmp")
            if tmp.exists():
                tmp.unlink()
            pq.write_table(self.empty_schema.empty_table(), tmp, compression="zstd")
            os.replace(tmp, self.canonical)
        return 0


def ts_to_date(ts_ms) -> datetime | None:
    """Amazon Reviews 2023의 timestamp(ms) → datetime(UTC). 비정상값은 None."""
    try:
        ts = int(ts_ms)
        if ts > 10**12 * 10:  # 일부 레코드는 µs
            ts //= 1000
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def parse_price(raw) -> float | None:
    """price 필드는 float / '$12.99' / 'from $9.99' / None 등 혼재."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    s = str(raw).replace("$", "").replace(",", "").strip()
    for tok in s.split():
        try:
            v = float(tok)
            return v if v > 0 else None
        except ValueError:
            continue
    return None


class ChunkedParquetWriter:
    """rows(dict)를 모아 청크 단위로 단일 parquet 파일에 append."""

    def __init__(self, out_path: Path, chunk_rows: int = 200_000,
                 empty_schema: pa.Schema | None = None):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_rows = chunk_rows
        self.buffer: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.total = 0
        # 0행이어도 파일이 존재해야 하위 파이프라인이 깨지지 않는다
        self.empty_schema = empty_schema

    def add(self, row: dict):
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.out_path, table.schema, compression="zstd")
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.total += len(self.buffer)
        self.buffer = []

    def close(self) -> int:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        elif self.empty_schema is not None:
            pq.write_table(self.empty_schema.empty_table(), self.out_path,
                           compression="zstd")
        return self.total


def write_run_log(name: str, stats: dict):
    """수집 실행 메타데이터를 기록 (재현성/품질 리포트의 원천)."""
    log_dir = PROJECT_ROOT / "data" / "raw" / "_run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stats = {"run": name, "finished_at": datetime.now(timezone.utc).isoformat(), **stats}
    with open(log_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
