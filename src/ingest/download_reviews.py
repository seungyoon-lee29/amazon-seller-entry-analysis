"""Stage 1c — 리뷰 수집 (단일 패스, 이중 출력).

31GB 리뷰 jsonl을 1회 스트리밍하면서:
  (a) slim:  전체 카테고리 리뷰의 최소 컬럼(텍스트 제외) → 니치별 월간 집계용
  (b) full:  타깃 니치 ASIN의 전체 컬럼(텍스트 포함)     → 텍스트 마이닝용

기간 필터(config.period)는 여기서 적용해 저장량을 줄인다.

사용:
    python src/ingest/download_reviews.py                # universe 필요
    python src/ingest/download_reviews.py --limit 20000  # 스모크 테스트
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
import pyarrow as pa  # noqa: E402
from src.ingest.common import (  # noqa: E402
    PROJECT_ROOT, ShardSink, iter_jsonl_offsets, load_ckpt, load_config,
    save_ckpt, ts_to_date, write_run_log,
)

CKPT_EVERY = 2_000_000   # 이 "스캔" 행 수마다 샤드를 닫고 체크포인트 저장
SCHEMA_VERSION = 2        # v2: slim에 timestamp_ms/user_id 보존


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=cfg["dataset"]["main_category"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # 타깃 니치 ASIN 셋 로드 (없으면 slim만 수집)
    staging = PROJECT_ROOT / cfg["paths"]["staging"]
    uni_path = staging / f"universe_{cfg['target_niche']['name']}.parquet"
    sample_uni_path = staging / f"universe_{cfg['target_niche']['name']}_sample.parquet"
    if args.limit and sample_uni_path.exists():
        uni_path = sample_uni_path
    target_asins: set[str] = set()
    if uni_path.exists():
        target_asins = set(
            duckdb.sql(f"SELECT parent_asin FROM '{uni_path}'").df()["parent_asin"])
        print(f"target niche ASINs: {len(target_asins):,}")
    else:
        msg = "universe 파일 없음 → full 출력이 비어 Q2가 깨집니다. build_universe.py를 먼저 실행하세요."
        if args.limit:
            print(f"[경고] {msg}")
        else:
            sys.exit(msg)

    start_date = datetime.fromisoformat(cfg["period"]["start"]).date()
    end_date = datetime.fromisoformat(cfg["period"]["end"]).date()

    suffix = f"_sample{args.limit}" if args.limit else ""
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    slim_out = raw / f"reviews_slim_{args.category}{suffix}.parquet"
    full_out = raw / f"reviews_full_{cfg['target_niche']['name']}{suffix}.parquet"
    ckpt_path = raw / f"reviews_{args.category}{suffix}.ckpt.json"
    resumable = args.limit is None

    slim = ShardSink(slim_out, cfg["runtime"]["chunk_rows"])
    full_schema = pa.schema([
        ("parent_asin", pa.string()), ("asin", pa.string()),
        ("timestamp_ms", pa.int64()), ("date", pa.string()),
        ("rating", pa.float64()), ("verified_purchase", pa.bool_()),
        ("review_title", pa.string()), ("text", pa.string()),
        ("helpful_vote", pa.int64()), ("user_id", pa.string()),
    ])
    full = ShardSink(full_out, cfg["runtime"]["chunk_rows"], empty_schema=full_schema)
    path_in_repo = f"raw/review_categories/{args.category}.jsonl"
    revision = cfg["dataset"].get("revision")

    # ── 재개/스킵 판단 ──
    start_offset = 0
    n_in = n_period = n_full = 0
    if resumable:
        ckpt = load_ckpt(ckpt_path)
        if ckpt and ckpt.get("done"):
            if ckpt.get("schema_version") != SCHEMA_VERSION:
                sys.exit(f"기존 리뷰 parquet은 오래된 스키마입니다: {slim_out.name}. "
                         "dedup 손실 방지를 반영하려면 reviews parquet/ckpt를 삭제 후 재수집하세요.")
            if ckpt.get("dataset_revision") != revision:
                sys.exit(f"체크포인트 dataset revision이 현재 설정과 다릅니다: "
                         f"{ckpt.get('dataset_revision')} != {revision}")
            print(f"이미 완료됨(skip): {slim_out.name}")
            return
        if ckpt is None and slim_out.exists():
            sys.exit(f"canonical은 있지만 완료 체크포인트가 없습니다: {slim_out.name}. "
                     "부분 파일일 수 있으니 파일을 검증하거나 삭제 후 재수집하세요.")
        if ckpt:
            if ckpt.get("schema_version") != SCHEMA_VERSION:
                sys.exit(f"기존 리뷰 체크포인트는 오래된 스키마입니다: {ckpt_path}. "
                         "reviews parquet/part/ckpt를 삭제 후 재수집하세요.")
            if ckpt.get("dataset_revision") != revision:
                sys.exit(f"체크포인트 dataset revision이 현재 설정과 다릅니다: "
                         f"{ckpt.get('dataset_revision')} != {revision}")
            start_offset = ckpt["offset"]
            slim.resume_from(ckpt["slim_shard"])
            full.resume_from(ckpt["full_shard"])
            n_in, n_period, n_full = ckpt["n_in"], ckpt["n_period"], ckpt["n_full"]
            print(f"재개: offset={start_offset:,} slim_shard={ckpt['slim_shard']} "
                  f"full_shard={ckpt['full_shard']} scanned={n_in:,}")

    last_off = start_offset
    for row, off in iter_jsonl_offsets(
        cfg["dataset"]["repo_id"], path_in_repo, start_offset, revision=revision
    ):
        if args.limit and n_in >= args.limit:
            break
        n_in += 1
        last_off = off
        dt = ts_to_date(row.get("timestamp"))
        if dt is not None and (start_date <= dt.date() <= end_date):
            n_period += 1
            base = {
                "parent_asin": row.get("parent_asin"),
                "asin": row.get("asin"),
                "timestamp_ms": row.get("timestamp"),
                "date": dt.date().isoformat(),
                "rating": row.get("rating"),
                "verified_purchase": bool(row.get("verified_purchase")),
                "user_id": row.get("user_id"),
            }
            slim.add(base)
            if row.get("parent_asin") in target_asins:
                n_full += 1
                full.add({
                    **base,
                    "review_title": (row.get("title") or "")[:300],
                    "text": (row.get("text") or "")[:5000],
                    "helpful_vote": row.get("helpful_vote"),
                    "user_id": row.get("user_id"),
                })
        if n_in % CKPT_EVERY == 0:
            if resumable:
                slim.roll()
                full.roll()
                save_ckpt(ckpt_path, {
                    "offset": off, "slim_shard": slim.shard_idx,
                    "full_shard": full.shard_idx, "n_in": n_in,
                    "n_period": n_period, "n_full": n_full,
                    "schema_version": SCHEMA_VERSION,
                    "dataset_revision": revision, "done": False})
            print(f"  ... {n_in:,} scanned / {n_period:,} kept (ckpt @ {off:,})", flush=True)

    slim_rows = slim.finalize()
    full_rows = full.finalize()
    if resumable:
        save_ckpt(ckpt_path, {
            "offset": last_off, "slim_shard": slim.shard_idx,
            "full_shard": full.shard_idx, "n_in": n_in,
            "n_period": n_period, "n_full": n_full,
            "schema_version": SCHEMA_VERSION,
            "dataset_revision": revision, "done": True})
    stats = {
        "category": args.category, "rows_scanned": n_in,
        "rows_in_period": n_period, "rows_full_niche": n_full,
        "slim_rows": slim_rows, "full_rows": full_rows,
        "dataset_revision": revision,
        "limit": args.limit,
    }
    write_run_log(f"reviews_{args.category}{suffix}", stats)
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
