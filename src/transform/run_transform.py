"""Stage 2 실행기 — SQL 템플릿 치환 후 DuckDB 실행, 결과를 DuckDB에 적재.

사용:
    python src/transform/run_transform.py            # 전체 데이터 → analysis.duckdb
    python src/transform/run_transform.py --sample   # 스모크 샘플 → analysis_sample.duckdb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
from src.ingest.common import PROJECT_ROOT, load_config, write_run_log  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="_sample* parquet 사용")
    ap.add_argument("--db", default=None,
                    help="DuckDB 파일 경로 (기본: full은 analysis.duckdb, sample은 analysis_sample.duckdb)")
    args = ap.parse_args()

    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    staging = PROJECT_ROOT / cfg["paths"]["staging"]
    def input_files(prefix: str, where: Path) -> list[Path]:
        if args.sample:
            files = sorted(where.glob(f"{prefix}*_sample*.parquet"))
        else:
            files = [p for p in sorted(where.glob(f"{prefix}*.parquet"))
                     if "_sample" not in p.name]
        if not files:
            sys.exit(f"입력 없음: {where}/{prefix}*.parquet — ingestion을 먼저 실행하세요.")
        return files

    def glob_sql(files: list[Path]) -> str:
        return ", ".join(f"'{f}'" for f in files)

    universe_name = cfg["target_niche"]["name"]
    universe_path = staging / f"universe_{universe_name}.parquet"
    if args.sample:
        sample_universe_path = staging / f"universe_{universe_name}_sample.parquet"
        if sample_universe_path.exists():
            universe_path = sample_universe_path

    meta_files = input_files("meta_", raw)
    slim_files = input_files("reviews_slim_", raw)
    full_files = input_files("reviews_full_", raw)

    default_db = "analysis_sample.duckdb" if args.sample else "analysis.duckdb"
    db_path = Path(args.db) if args.db else \
        PROJECT_ROOT / cfg["paths"]["marts"] / default_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    def has_column(files: list[Path], column: str) -> bool:
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{glob_sql(files)}], union_by_name=true)"
        ).df()["column_name"]
        return column in set(cols)

    slim_timestamp_expr = (
        "CAST(timestamp_ms AS BIGINT)" if has_column(slim_files, "timestamp_ms")
        else "NULL::BIGINT"
    )
    slim_user_expr = (
        "CAST(user_id AS VARCHAR)" if has_column(slim_files, "user_id")
        else "NULL::VARCHAR"
    )

    subs = {
        "{{meta_glob}}": glob_sql(meta_files),
        "{{slim_glob}}": glob_sql(slim_files),
        "{{full_glob}}": glob_sql(full_files),
        "{{universe_path}}": str(universe_path),
        "{{slim_timestamp_ms}}": slim_timestamp_expr,
        "{{slim_user_id}}": slim_user_expr,
    }

    # 대용량(수천만 행) 변환 안정화 설정:
    # - preserve_insertion_order=false: 정렬 유지 부담 제거 → 메모리 대폭 절감
    # - temp_directory: 디스크 여유가 큰 마트 폴더에 스필(기본 /tmp는 작을 수 있음)
    # 마트 SQL은 OOM 폭발 쿼리를 ASOF 조인으로 재작성해 스필 자체가 작다(D-011).
    tmp_dir = PROJECT_ROOT / cfg["paths"]["marts"] / ".duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{tmp_dir}'")

    sql_dir = PROJECT_ROOT / "src" / "transform"
    sql_files = [
        sql_dir / "01_staging.sql",
        sql_dir / "02_marts.sql",
    ]
    stats = {"sample": args.sample, "db": str(db_path.relative_to(PROJECT_ROOT)), "tables": {}}
    for sql_file in sql_files:
        sql = sql_file.read_text(encoding="utf-8")
        for k, v in subs.items():
            sql = sql.replace(k, v)
        t0 = time.time()
        con.execute(sql)
        print(f"[OK] {sql_file.name} ({time.time()-t0:.1f}s)")

    # marts를 parquet으로도 내보내기 (노트북/대시보드에서 DB 락 없이 사용)
    marts_dir = PROJECT_ROOT / cfg["paths"]["marts"]
    for (tbl,) in con.execute("SHOW TABLES").fetchall():
        n = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        stats["tables"][tbl] = n
        print(f"  {tbl}: {n:,} rows")
        if tbl.startswith(("dim_", "fct_", "mart_")):
            out_pq = marts_dir / f"{tbl}{'_sample' if args.sample else ''}.parquet"
            con.execute(f"COPY {tbl} TO '{out_pq}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    write_run_log(f"transform{'_sample' if args.sample else ''}", stats)
    con.close()
    print(f"done → {db_path}")


if __name__ == "__main__":
    main()
