"""Stage 1b — 타깃 니치 ASIN universe 구축.

meta parquet에서 config의 키워드 규칙으로 wall_shelves 니치 ASIN 목록을 만든다.
결과: data/staging/universe_{niche}.parquet + 매칭 근거 컬럼(검수용)

사용:
    python src/ingest/build_universe.py
    python src/ingest/build_universe.py --meta data/raw/meta_Home_and_Kitchen_sample5000.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
from src.ingest.common import PROJECT_ROOT, load_config, write_run_log  # noqa: E402


def main():
    cfg = load_config()
    niche = cfg["target_niche"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", nargs="*", default=None,
                    help="meta parquet 경로(들). 기본: main + cross_listing 카테고리")
    args = ap.parse_args()

    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    metas = [Path(m) if Path(m).is_absolute() else PROJECT_ROOT / m for m in args.meta] if args.meta else [
        raw / f"meta_{cfg['dataset']['main_category']}.parquet",
        raw / f"meta_{cfg['dataset']['cross_listing_category']}.parquet",
    ]
    metas = [m for m in metas if m.exists()]
    if not metas:
        sys.exit("meta parquet이 없습니다. download_meta.py를 먼저 실행하세요.")

    title_pred = " OR ".join(
        f"lower(title) LIKE '%{kw.lower()}%'" for kw in niche["title_keywords"])
    cat_pred = " OR ".join(
        f"categories LIKE '%{kw}%'" for kw in niche["category_keywords"])
    # 오탐 제외: 제목 단독 매칭(카테고리 미확증)에만 적용. 카테고리 확증은 Amazon
    # 자체 분류이므로 더 강한 신호 → 항상 유지. (검수 결과 D-013)
    exclude_kws = niche.get("exclude_keywords", [])
    exclude_pred = " OR ".join(
        f"lower(title) LIKE '%{kw.lower()}%'" for kw in exclude_kws) or "FALSE"

    con = duckdb.connect()
    files = ", ".join(f"'{m}'" for m in metas)
    df = con.execute(f"""
        SELECT *,
               ({title_pred})                  AS matched_title,
               ({cat_pred})                    AS matched_category,
               lower(title) LIKE '%hexagon%' OR lower(title) LIKE '%honeycomb%'
                                               AS is_hexagon
        FROM read_parquet([{files}], union_by_name=true)
        WHERE ({cat_pred})
           OR (({title_pred}) AND NOT ({exclude_pred}))
        QUALIFY row_number() OVER (
            PARTITION BY parent_asin
            ORDER BY matched_category DESC, is_hexagon DESC,
                     coalesce(rating_number, 0) DESC, title ASC
        ) = 1
    """).df()

    # Sample smoke runs must not clobber the full universe. Infer sample mode from
    # the input file names because smoke passes an explicit sample meta path.
    sample_suffix = "_sample" if all("_sample" in m.stem for m in metas) else ""
    out = PROJECT_ROOT / cfg["paths"]["staging"] / f"universe_{niche['name']}{sample_suffix}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # 제외된 오탐(제목 단독 매칭 ∩ exclude) 건수 — 검수 추적용
    excluded_fp = con.execute(f"""
        SELECT count(DISTINCT parent_asin)
        FROM read_parquet([{files}], union_by_name=true)
        WHERE NOT ({cat_pred}) AND ({title_pred}) AND ({exclude_pred})
    """).fetchone()[0]

    stats = {
        "niche": niche["name"],
        "asin_count": len(df),
        "hexagon_count": int(df["is_hexagon"].sum()),
        "matched_by_title_only": int((df["matched_title"] & ~df["matched_category"]).sum()),
        "matched_by_category_only": int((~df["matched_title"] & df["matched_category"]).sum()),
        "excluded_false_positive": int(excluded_fp),
        "exclude_keywords": exclude_kws,
        "inputs": [str(m.name) for m in metas],
        "output": str(out.relative_to(PROJECT_ROOT)),
        "sample": bool(sample_suffix),
    }
    write_run_log(f"universe_{niche['name']}{sample_suffix}", stats)
    print(f"done: {stats}")
    print("\n[검수 안내] 아래 쿼리로 30개 샘플을 직접 눈으로 확인할 것:")
    print(f"  duckdb -c \"SELECT title FROM '{out}' USING SAMPLE 30\"")


if __name__ == "__main__":
    main()
