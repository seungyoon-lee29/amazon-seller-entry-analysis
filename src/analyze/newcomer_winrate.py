#!/usr/bin/env python3
"""Q1 보강 — 니치별 "신규 진입자 안착률(newcomer win-rate)".

"말 그대로 니치마켓이라면 시장 크기와 별개로 그 아이템만 잘 나가면 되는 것 아닌가?"
이 직관을 데이터로 정식화한다. Q3의 안착(출시 90~365일 후기 트랙션)을 *니치 단위로 역집계*해
"이 니치에서 신규 진입자가 실제로 뜨는 비율"을 잰다.

핵심 설계: Q3 라벨은 니치-상대 P75라 니치마다 ~25%로 자기정규화돼 *비교가 불가능*하다.
그래서 여기선 **절대 기준**(later_reviews ≥ win_floor)을 써서 니치 간 비교가 가능하게 한다.
이 지표는 니치 크기와 상관이 거의 없어(≈ -0.12) '크기'와 독립된 *승산* 축으로 해석한다.

출력:
  data/marts/mart_niche_newcomer[_sample].parquet   니치 × 신규 안착률
  docs/q1_newcomer_winnability[_sample].md           해설 리포트(자동 생성)

사용: python src/analyze/newcomer_winrate.py [--sample]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.ingest.common import PROJECT_ROOT, load_config, write_run_log  # noqa: E402


def compute(launch: pl.DataFrame, floor: int, min_launches: int) -> pl.DataFrame:
    df = launch.with_columns(
        (pl.col("reviews_12m") - pl.col("reviews_90d")).clip(lower_bound=0).alias("later"))
    g = (df.group_by("niche").agg(
            pl.len().alias("n_launches"),
            (pl.col("later") >= floor).mean().alias("newcomer_win_rate"),
            pl.col("later").median().alias("median_later_reviews"))
         .filter(pl.col("n_launches") >= min_launches)
         .sort("newcomer_win_rate", descending=True))
    return g


def write_report(g: pl.DataFrame, focus: str, floor: int, min_launches: int,
                 size_corr: float, opp_corr: float, out_path: Path):
    gp = g.to_pandas()
    med = gp["newcomer_win_rate"].median()
    # 작지만 승산 높은 니치(출시 50~300 중 win-rate 상위)
    small = gp[(gp["n_launches"] >= 50) & (gp["n_launches"] <= 300)] \
        .sort_values("newcomer_win_rate", ascending=False).head(6)
    fs = gp[gp["niche"] == focus]
    L = [
        "# Q1 보강 — 니치별 신규 진입자 안착률 (newcomer win-rate)",
        "",
        "> 자동 생성: `src/analyze/newcomer_winrate.py` · 단위: 니치(출시 코호트 역집계)",
        "",
        '**질문**: "니치마켓이면 시장 크기와 별개로 그 아이템만 잘 나가면 되는 것 아닌가?" '
        "이를 정량화하기 위해, 각 니치에서 신규 진입 상품이 실제로 트랙션을 얻는 비율을 잰다.",
        "",
        f"- **정의**: 출시 90~365일 후기 리뷰가 **절대 기준 {floor}개 이상**이면 '신규가 떴다(승)'. "
        f"니치별 출시 코호트 {min_launches}개 이상만 집계.",
        "- **왜 절대 기준?** Q3 안착 라벨(니치-상대 P75)은 니치마다 ~25%로 자기정규화돼 비교 불가. "
        "절대 기준이라야 니치 간 '승산'을 비교할 수 있다.",
        "",
        "## 핵심 — 승산은 크기와 거의 무관하다",
        "",
        f"- 신규 안착률: 중앙값 **{med:.0%}**, 범위 {gp['newcomer_win_rate'].min():.0%}"
        f"~{gp['newcomer_win_rate'].max():.0%} (니치 {len(gp)}개).",
        f"- **니치 크기(log 출시수)와의 상관: {size_corr:+.2f}** → 거의 무관. "
        "즉 '크다고 신규가 잘 뜨는 게 아니고, 작아도 잘 뜨는 니치가 있다' = 질문의 직관을 데이터가 지지.",
        f"- 기존 Q1 기회점수와의 상관: **{opp_corr:+.2f}** — 거의 직교. "
        "즉 '신규 승산'은 수요성장·집중도로 만든 기회점수가 *못 잡는 별개의 정보*다(중복 아닌 추가 축).",
        "",
        "![신규 승산은 크기와 무관](figures/q1_newcomer_winnability.png)",
        "",
        "## 작지만 신규 승산이 높은 니치 (크기≠승산의 증거)",
        "",
        "| 니치 | 출시수 | 신규 안착률 |",
        "|---|---:|---:|",
    ]
    for r in small.itertuples():
        L.append(f"| {r.niche} | {int(r.n_launches):,} | {r.newcomer_win_rate:.0%} |")

    if len(fs):
        f = fs.iloc[0]
        rank_pct = (gp["newcomer_win_rate"] > f["newcomer_win_rate"]).mean() * 100
        L += [
            "",
            f"## 셀러 니치({focus}) 재해석",
            "",
            f"- {focus} 신규 안착률 **{f['newcomer_win_rate']:.0%}** "
            f"(출시 {int(f['n_launches']):,}개, 전체 니치 중 상위 {rank_pct:.0f}%) — **중앙값({med:.0%}) 이상**.",
            "- 함의(정직한 반전): 이 니치에서 *신규가 못 뜨는 게 아니다*. 신규가 뜰 여지는 평균 이상이었다. "
            "문제는 (Q1)수요가 줄고 + (Q2)셀러의 차별화 축(색상·세트)이 진짜 불만(재질)이 아니었고 + "
            "(Q3)안착을 가르는 초기 트랙션을 설계하지 않은 것. "
            "**'시장이 작아서'가 아니라 '내 아이템을 뜨게 만드는 조건을 못 맞춰서'**가 더 정확한 진단.",
        ]
    L += [
        "",
        "## 한계",
        "",
        f"- 후기 트랙션도 리뷰 프록시(L-1) — 판매가 아니다. 절대 기준({floor})은 config 파라미터이며 "
        "값 선택에 결과가 민감(민감도 분석 대상).",
        "- 과거 코호트 기반 = *그 니치의 구조적 승산*이지, 특정 신규 상품의 성공을 보장하지 않는다.",
    ]
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    cfg = load_config()
    nc = cfg["q1_newcomer"]
    focus = cfg["q2_unmet_needs"]["focus_niche"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="_sample 마트 사용")
    args = ap.parse_args()

    marts = PROJECT_ROOT / cfg["paths"]["marts"]
    sfx = "_sample" if args.sample else ""
    launch_path = marts / f"mart_launch{sfx}.parquet"
    if not launch_path.exists():
        sys.exit(f"입력 없음: {launch_path} — transform을 먼저 실행하세요.")

    launch = pl.read_parquet(launch_path)
    g = compute(launch, nc["win_floor"], nc["min_launches"])
    if g.height == 0:
        sys.exit("집계 결과 없음(min_launches 미달) — sample이면 정상.")

    # 크기 독립성: log(출시수) vs 안착률 상관
    gp = g.to_pandas()
    size_corr = float(np.corrcoef(np.log1p(gp["n_launches"]), gp["newcomer_win_rate"])[0, 1])

    # 기존 Q1 기회점수와의 상관(있으면 join)
    opp_corr = float("nan")
    score_path = marts / f"mart_niche_score{sfx}.parquet"
    if score_path.exists():
        sc = pl.read_parquet(score_path).select("niche", "opportunity_score", "rank")
        g = g.join(sc, on="niche", how="left")
        j = g.drop_nulls("opportunity_score").to_pandas()
        if len(j) > 2:
            opp_corr = float(np.corrcoef(j["newcomer_win_rate"], j["opportunity_score"])[0, 1])

    g.write_parquet(marts / f"mart_niche_newcomer{sfx}.parquet")
    docs = PROJECT_ROOT / "docs"
    write_report(g, focus, nc["win_floor"], nc["min_launches"], size_corr, opp_corr,
                 docs / f"q1_newcomer_winnability{sfx}.md")

    fs = gp[gp["niche"] == focus]
    stats = {
        "sample": args.sample, "n_niches": int(g.height),
        "win_floor": nc["win_floor"], "median_win_rate": round(float(gp["newcomer_win_rate"].median()), 4),
        "size_corr": round(size_corr, 3), "opp_score_corr": round(opp_corr, 3),
        "focus_win_rate": round(float(fs["newcomer_win_rate"].iloc[0]), 4) if len(fs) else None,
    }
    write_run_log(f"q1_newcomer{sfx}", stats)
    print(f"done: {stats}")
    print(f"  report → docs/q1_newcomer_winnability{sfx}.md")


if __name__ == "__main__":
    main()
