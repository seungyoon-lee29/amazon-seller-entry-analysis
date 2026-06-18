"""Stage 3 / Q1 — 니치 기회 스코어링.

mart_niche_monthly(니치×월 패널)에서 니치별 기회 지표를 만들고,
표준화 → 가중합으로 "진입 기회 점수"를 매긴다. 핵심은 점수 자체가 아니라
**가중치 민감도 분석** — 가중치를 흔들어도 상위 니치가 유지되는지 검증한다
(설계서 Stage 3: "가중치는 민감도 분석 필수").

지표(모두 높을수록 진입 매력적이 되도록 부호 정렬):
  demand_growth    리뷰 증가율(최근 vs 직전 창)        — 수요 모멘텀
  market_openness  1 - 상위5 상품 리뷰 점유율           — 진입 여지(낮은 집중도)
  quality_gap      저평점(≤2★) 비율                     — 품질 개선 기회
  uncrowded        신규 진입 강도(신규/활성)의 역        — 낮은 포화도

해석 제한(limitations.md L-1): 리뷰는 판매의 불완전 프록시이므로 수요는
"절대량"이 아니라 "증가율(추세)"로만 점수에 반영한다.

출력:
  data/marts/mart_niche_score[_sample].parquet   니치별 지표·z점수·기회점수·랭크 안정성
  docs/q1_niche_scorecard.md                      니치 기회 스코어카드(자동 생성)

사용:
    python src/analyze/niche_score.py                       # 전체 데이터
    python src/analyze/niche_score.py --sample              # 스모크 샘플
    python src/analyze/niche_score.py --sample --min-months 1 --min-active 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ingest.common import PROJECT_ROOT, load_config, write_run_log  # noqa: E402

# 지표 정의: (패널 집계 컬럼, 방향). 방향 +1 = 클수록 좋음, -1 = 작을수록 좋음.
# market_openness/uncrowded는 "원지표(집중도/포화도)"가 작을수록 좋으므로 -1.
METRIC_DIRECTION = {
    "demand_growth": +1,
    "market_openness": +1,   # 이미 (1 - 집중도)로 계산해 부호 정렬됨
    "quality_gap": +1,
    "uncrowded": +1,         # 이미 (1 - 포화도)로 계산해 부호 정렬됨
}


def aggregate_indicators(con, panel_path: Path, recent_m: int, prior_m: int) -> pd.DataFrame:
    """패널을 최근/직전 창으로 나눠 니치별 원지표를 집계한다."""
    max_month = con.execute(
        f"SELECT max(review_month) FROM '{panel_path}'").fetchone()[0]
    # 창 경계: recent = (max - recent_m, max], prior = (max - recent_m - prior_m, max - recent_m]
    df = con.execute(f"""
        WITH p AS (
            SELECT *,
                   date_diff('month', review_month, DATE '{max_month}') AS m_ago
            FROM '{panel_path}'
        )
        SELECT
            niche,
            count(*)                                              AS n_months,
            sum(n_reviews) FILTER (WHERE m_ago < {recent_m})      AS recent_reviews,
            sum(n_reviews) FILTER (WHERE m_ago >= {recent_m}
                                     AND m_ago < {recent_m + prior_m}) AS prior_reviews,
            avg(n_active_products) FILTER (WHERE m_ago < {recent_m}) AS avg_active_recent,
            -- 리뷰 가중 평균 집중도/저평점비율 (월별 단순평균보다 안정적)
            sum(top5_review_share * n_reviews) FILTER (WHERE m_ago < {recent_m})
                / nullif(sum(n_reviews) FILTER (WHERE m_ago < {recent_m}), 0)
                                                                  AS avg_top5_share,
            sum(low_rating_share * n_reviews) FILTER (WHERE m_ago < {recent_m})
                / nullif(sum(n_reviews) FILTER (WHERE m_ago < {recent_m}), 0)
                                                                  AS low_rating_share,
            sum(n_new_products) FILTER (WHERE m_ago < {recent_m}) AS new_products_recent
        FROM p
        GROUP BY niche
    """).df()
    return df


def attach_target_flags(con, df: pd.DataFrame, dim_path: Path) -> pd.DataFrame:
    """니치별로 셀러 타깃 니치 포함 여부 / hexagon 상품 수를 붙인다(회고 검증용)."""
    flags = con.execute(f"""
        SELECT niche,
               max(CASE WHEN in_target_niche THEN 1 ELSE 0 END) AS contains_target,
               sum(CASE WHEN is_hexagon THEN 1 ELSE 0 END)      AS hexagon_products
        FROM '{dim_path}' GROUP BY niche
    """).df()
    return df.merge(flags, on="niche", how="left")


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """원지표 → 부호 정렬된 해석용 지표(높을수록 매력적)."""
    df = df.copy()
    # 수요 증가율: 직전 창 대비. 직전 창 0이면 비교 불가(NaN) → 이후 중앙값 대치 + low_confidence.
    df["demand_growth"] = np.where(
        df["prior_reviews"] > 0,
        df["recent_reviews"] / df["prior_reviews"] - 1.0,
        np.nan,
    )
    df["market_openness"] = 1.0 - df["avg_top5_share"]
    df["quality_gap"] = df["low_rating_share"]
    # 신규 진입 강도 = 신규 상품 / 평균 활성 상품. 클수록 포화 → 역으로 매력도.
    intensity = df["new_products_recent"] / df["avg_active_recent"].replace(0, np.nan)
    df["new_entry_intensity"] = intensity
    df["uncrowded"] = 1.0 - intensity.clip(upper=1.0)
    df["low_confidence"] = df["demand_growth"].isna()
    return df


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def score_and_rank(df: pd.DataFrame, weights: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """z-표준화 후 가중합. z 행렬도 반환(민감도 분석에서 재사용)."""
    df = df.copy()
    zcols = []
    for m, direction in METRIC_DIRECTION.items():
        # 결측은 니치 간 중앙값으로 대치(중립). 방향은 이미 지표에 반영됨.
        filled = df[m].fillna(df[m].median())
        zname = f"z_{m}"
        df[zname] = zscore(filled * direction)
        zcols.append(zname)
    Z = df[zcols].to_numpy()                       # (n_niche, n_metric)
    w = np.array([weights[m] for m in METRIC_DIRECTION])
    # 가중합은 elementwise로 계산(Z@w와 동일). numpy 2.x의 matmul이 macOS
    # Accelerate에서 유한 입력에도 허위 FPE 경고를 내므로 회피.
    df["opportunity_score"] = (Z * w).sum(axis=1)
    df = df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df, df[["niche"] + zcols].set_index("niche")


def sensitivity(zmat: pd.DataFrame, base_weights: dict, cfg_s: dict,
                seed: int) -> pd.DataFrame:
    """가중치를 Dirichlet으로 섭동해 랭크 안정성을 측정한다.

    반환: 니치별 median_rank, rank_p10, rank_p90, prob_top_n,
          그리고 (메타) 섭동 랭킹과 기본 랭킹의 평균 Spearman 상관.
    """
    rng = np.random.default_rng(seed)
    metrics = list(METRIC_DIRECTION)
    base = np.array([base_weights[m] for m in metrics], dtype=float)
    base = base / base.sum()
    alpha = cfg_s["dirichlet_alpha"] * base
    Z = zmat[[f"z_{m}" for m in metrics]].to_numpy()
    niches = zmat.index.to_numpy()
    n = len(niches)
    top_n = min(cfg_s["top_n"], n)

    base_scores = (Z * base).sum(axis=1)
    base_order = np.argsort(-base_scores)
    base_rank = np.empty(n, int)
    base_rank[base_order] = np.arange(1, n + 1)

    ranks = np.empty((cfg_s["n_samples"], n), dtype=int)
    spearmans = []
    for i in range(cfg_s["n_samples"]):
        w = rng.dirichlet(alpha)
        scores = (Z * w).sum(axis=1)
        order = np.argsort(-scores)
        r = np.empty(n, int)
        r[order] = np.arange(1, n + 1)
        ranks[i] = r
        # Spearman = 순위 간 피어슨 상관
        if n > 1:
            spearmans.append(np.corrcoef(r, base_rank)[0, 1])

    out = pd.DataFrame({
        "niche": niches,
        "median_rank": np.median(ranks, axis=0),
        "rank_p10": np.percentile(ranks, 10, axis=0),
        "rank_p90": np.percentile(ranks, 90, axis=0),
        "prob_top_n": (ranks <= top_n).mean(axis=0),
    })
    out.attrs["mean_spearman"] = float(np.mean(spearmans)) if spearmans else float("nan")
    out.attrs["top_n"] = top_n
    return out


def write_scorecard(df: pd.DataFrame, sens: pd.DataFrame, cfg: dict,
                    n_excluded: int, out_path: Path):
    s = cfg["q1_scoring"]
    mean_sp = sens.attrs["mean_spearman"]
    top_n = sens.attrs["top_n"]
    w = s["weights"]
    lines = [
        "# Q1 — 니치 기회 스코어카드",
        "",
        "> 자동 생성: `src/analyze/niche_score.py` · 점수는 z-표준화 지표의 가중합.",
        f"> 창: 최근 {s['recent_months']}개월 vs 직전 {s['prior_months']}개월 · "
        f"필터: 관측 ≥{s['min_months']}개월 & 평균 활성상품 ≥{s['min_active_products']} "
        f"(제외 {n_excluded}개 니치)",
        "",
        "**해석 제한(L-1)**: 리뷰는 판매의 불완전 프록시. 수요는 절대량이 아니라 "
        "*증가율(추세)* 로만 반영했고, 점수는 니치 *간* 순위가 아니라 "
        "*상대적 진입 매력도*의 참고치로만 해석한다.",
        "",
        f"가중치(기본): "
        + ", ".join(f"{k} {v}" for k, v in w.items()),
        "",
        "![니치 기회 지도(아키타입 사분면)](figures/q1_opportunity_map.png)",
        "",
        "> 우상단(성장+개방)=진짜기회. 성장하면서 안 붐비는 니치는 906개 중 소수뿐이고, "
        "셀러가 들어간 Floating Shelves(★)는 좌측 쇠퇴 구역에 있었다.",
        "",
        "## 기회 점수 상위 니치",
        "",
        "| 랭크 | 니치 | 점수 | 수요증가 | 진입여지 | 품질갭(≤2★) | 비포화 | "
        f"top{top_n}안정성 | 타깃 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    sm = sens.set_index("niche")
    for r in df.head(15).itertuples():
        st = sm.loc[r.niche]
        # ★ = 셀러 자사(hexagon) 상품이 실제로 속한 니치. contains_target(≥1 매치)는
        # 교차등록 오탐까지 잡아 너무 느슨하므로 hexagon 보유로 좁힌다.
        tgt = "★" if getattr(r, "hexagon_products", 0) else ""
        conf = " ⚠" if r.low_confidence else ""
        lines.append(
            f"| {r.rank} | {r.niche}{conf} | {r.opportunity_score:+.2f} | "
            f"{_pct(r.demand_growth)} | {r.market_openness:.2f} | "
            f"{r.quality_gap:.2f} | {r.uncrowded:.2f} | "
            f"{st.prob_top_n:.0%} | {tgt} |"
        )
    lines += [
        "",
        "⚠ = 직전 창 데이터 없음(증가율 중앙값 대치). ★ = 셀러 자사(hexagon) 상품이 속한 니치.",
        "",
        "## 점수 분해 — 무엇이 점수를 견인하나",
        "",
        "기회점수는 블랙박스가 아니다. 상위 니치 대부분은 *수요 성장*이 점수를 견인하며, "
        "어떤 니치는 품질갭·진입여지가 동인이다. (가중 z-기여 분해)",
        "",
        "![기회점수 분해](figures/q1_score_decomposition.png)",
        "",
        "## 가중치 민감도",
        "",
        f"- 섭동 {s['sensitivity']['n_samples']}회(Dirichlet α={s['sensitivity']['dirichlet_alpha']}) "
        f"랭킹과 기본 랭킹의 평균 Spearman 상관: **{mean_sp:.3f}**",
        f"  (1에 가까울수록 가중치를 흔들어도 순위가 안정적 → 결론이 가중치에 덜 민감)",
        f"- `prob_top_n`: 가중치를 흔들었을 때 해당 니치가 상위 {top_n}에 드는 빈도. "
        "이 값이 높은 니치가 가중치와 무관하게 견고한 기회.",
        "",
        "![순위 견고성(가중치 섭동 p10~p90)](figures/q1_rank_uncertainty.png)",
    ]
    # 셀러 타깃 니치 회고 검증 한 줄.
    # 타깃 universe는 183개 니치에 흩어지므로(교차등록) "점수 최고 니치"를 집으면
    # 오탐을 셀러 니치로 오인한다. 셀러 자사(hexagon) 상품이 가장 많이 속한 니치를
    # 주력 니치로 본다 (없으면 타깃 상품 보유 니치 중 자사 상품 최다).
    tgt_rows = df[df.get("hexagon_products", 0) > 0]
    if not len(tgt_rows):
        tgt_rows = df[df.get("contains_target", 0) == 1]
    if len(tgt_rows):
        tr = tgt_rows.sort_values("hexagon_products", ascending=False).iloc[0]
        lines += [
            "",
            "## 셀러 타깃 니치 회고 검증",
            "",
            f"- 셀러 주력 니치 `{tr.niche}`(자사 hexagon 상품 {int(tr.hexagon_products)}개): "
            f"기회점수 {tr.opportunity_score:+.2f} (랭크 {int(tr['rank'])}/{len(df)}), "
            f"수요증가 {_pct(tr.demand_growth)}. "
            f'"당시 이 분석이 있었다면 이 니치에 진입했을까?"의 정량 근거.',
        ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:+.0%}"


def main():
    cfg = load_config()
    s = cfg["q1_scoring"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="_sample 마트 사용")
    ap.add_argument("--min-months", type=int, default=s["min_months"])
    ap.add_argument("--min-active", type=int, default=s["min_active_products"])
    args = ap.parse_args()

    marts = PROJECT_ROOT / cfg["paths"]["marts"]
    sfx = "_sample" if args.sample else ""
    panel = marts / f"mart_niche_monthly{sfx}.parquet"
    dim = marts / f"dim_product{sfx}.parquet"
    if not panel.exists():
        sys.exit(f"입력 없음: {panel} — transform을 먼저 실행하세요.")

    con = duckdb.connect()
    raw = aggregate_indicators(con, panel, s["recent_months"], s["prior_months"])
    raw = attach_target_flags(con, raw, dim)

    # 얇은 니치 제외 (안정적 지표 보장)
    n_before = len(raw)
    keep = (raw["n_months"] >= args.min_months) & \
           (raw["avg_active_recent"].fillna(0) >= args.min_active)
    df = raw[keep].copy()
    n_excluded = n_before - len(df)
    if df.empty:
        sys.exit(f"기준 통과 니치 0개 (관측≥{args.min_months}월 & 활성≥{args.min_active}). "
                 "샘플 데이터면 --min-months 1 --min-active 1 로 재시도.")

    df = compute_metrics(df)
    df, zmat = score_and_rank(df, s["weights"])
    sens = sensitivity(zmat, s["weights"], s["sensitivity"], cfg["runtime"]["seed"])
    df = df.merge(sens, on="niche", how="left")

    out_pq = marts / f"mart_niche_score{sfx}.parquet"
    keep_cols = [
        "niche", "rank", "opportunity_score", "n_months",
        "recent_reviews", "prior_reviews", "demand_growth", "market_openness",
        "quality_gap", "uncrowded", "new_entry_intensity", "avg_top5_share",
        "low_rating_share", "z_demand_growth", "z_market_openness",
        "z_quality_gap", "z_uncrowded", "median_rank", "rank_p10", "rank_p90",
        "prob_top_n", "low_confidence", "contains_target", "hexagon_products",
    ]
    df[keep_cols].to_parquet(out_pq, index=False)

    scorecard = PROJECT_ROOT / "docs" / f"q1_niche_scorecard{sfx}.md"
    write_scorecard(df, sens, cfg, n_excluded, scorecard)

    stats = {
        "sample": args.sample,
        "niches_scored": len(df),
        "niches_excluded": n_excluded,
        "mean_spearman": round(sens.attrs["mean_spearman"], 4),
        "top_niche": df.iloc[0]["niche"],
        "top_score": round(float(df.iloc[0]["opportunity_score"]), 4),
        "output": str(out_pq.relative_to(PROJECT_ROOT)),
        "scorecard": str(scorecard.relative_to(PROJECT_ROOT)),
    }
    write_run_log(f"q1_niche_score{sfx}", stats)
    print(f"done: {stats}")
    print(f"  scorecard → {scorecard}")


if __name__ == "__main__":
    main()
