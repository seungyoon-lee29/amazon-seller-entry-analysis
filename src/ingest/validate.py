"""Stage 1d — 수집 데이터 품질 리포트 생성.

data/raw/*.parquet 전체를 점검해 docs/data_quality_report.md 자동 생성.
점검 항목: 행 수, null 비율, 기간 커버리지, 중복률, 평점 분포 정상성.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
from src.ingest.common import PROJECT_ROOT, load_config  # noqa: E402


def profile_parquet(con, path: Path) -> list[str]:
    lines = [f"### `{path.name}`\n"]
    cols = con.execute(f"DESCRIBE SELECT * FROM '{path}'").df()
    n = con.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0]
    lines.append(f"- 행 수: **{n:,}**")

    null_stats = []
    for c in cols["column_name"]:
        nn = con.execute(
            f'SELECT count(*) FROM \'{path}\' WHERE "{c}" IS NULL').fetchone()[0]
        if n:
            null_stats.append((c, nn / n))
    worst = sorted(null_stats, key=lambda x: -x[1])[:5]
    lines.append("- null 비율 상위: " + ", ".join(f"`{c}` {r:.1%}" for c, r in worst))

    if "date" in set(cols["column_name"]):
        lo, hi = con.execute(f"SELECT min(date), max(date) FROM '{path}'").fetchone()
        lines.append(f"- 기간: {lo} ~ {hi}")
    if "rating" in set(cols["column_name"]):
        dist = con.execute(f"""
            SELECT rating, count(*) c FROM '{path}'
            WHERE rating IS NOT NULL GROUP BY 1 ORDER BY 1""").df()
        total = dist["c"].sum()
        lines.append("- 평점 분포: " + ", ".join(
            f"{int(r.rating)}★ {r.c/total:.0%}" for r in dist.itertuples()))
    if "parent_asin" in set(cols["column_name"]) and "date" not in set(cols["column_name"]):
        dup = con.execute(f"""
            SELECT 1 - count(DISTINCT parent_asin)::DOUBLE / count(*)
            FROM '{path}' WHERE parent_asin IS NOT NULL""").fetchone()[0]
        lines.append(f"- parent_asin 중복률: {dup:.2%}")
    lines.append("")
    return lines


def main():
    cfg = load_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    staging = PROJECT_ROOT / cfg["paths"]["staging"]
    files = sorted(list(raw.glob("*.parquet")) + list(staging.glob("*.parquet")))
    if not files:
        sys.exit("점검할 parquet이 없습니다.")

    con = duckdb.connect()
    out = [f"# 데이터 품질 리포트\n",
           f"> 생성: {datetime.now(timezone.utc).isoformat()} · 자동 생성 문서 (validate.py)\n"]
    for f in files:
        out += profile_parquet(con, f)

    report = PROJECT_ROOT / "docs" / "data_quality_report.md"
    report.write_text("\n".join(out), encoding="utf-8")
    print(f"written: {report}")


if __name__ == "__main__":
    main()
