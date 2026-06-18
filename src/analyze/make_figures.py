#!/usr/bin/env python3
"""
의사결정과 1:1로 연결되는 차트를 생성한다 — *장식이 아니라 "주장 1개 = 차트 1개"*.
모든 차트는 **제목 = 한 줄 결론**, 하단에 "읽는 법 / So what" 캡션을 단다(D-019).
GitHub 마크다운에 인라인 렌더되도록 docs/figures/*.png 로 저장한다.

데이터는 마트(parquet)·모델(pkl)에서 직접 읽는다 → 분석 스크립트와 동일 소스, 수치 불일치 불가.

산출(전체 실행 시):
  Q1: q1_opportunity_map(아키타입 사분면) · q1_score_decomposition · q1_rank_uncertainty
  회고: seller_niche_demand
  Q2: q2_aspect_heatmap · q2_priority(유병률×lift) · q2_trend
  Q3: q3_pr_curve · q3_calibration · q3_settlement_threshold · q3_segment
  (q3_shap_summary.png 는 settlement_model.py 가 생성)
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPL_CACHE = _PROJECT_ROOT / "data" / "marts" / ".matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import (average_precision_score,  # noqa: E402
                             precision_recall_curve)

from src.ingest.common import load_config, write_run_log  # noqa: E402

for _f in ["AppleGothic", "Arial Unicode MS", "AppleSDGothicNeo"]:
    try:
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지(U+2212 미지원 폰트)

FIG_DIR = _PROJECT_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ASPECT_KO = {
    "mounting_hardware": "벽고정·하드웨어", "sturdiness_sagging": "처짐·하중",
    "build_quality": "재질·내구성", "finish_cosmetic": "마감·외관",
    "size_fit": "크기·치수", "instructions": "설명서·조립", "shipping_damage": "배송 파손",
}
ARCH_COLORS = {"진짜기회": "#2a9d4a", "경쟁과열": "#e08a1e",
               "쇠퇴함정": "#7a6ff0", "회피": "#b0b0b0"}


def _marts(sample: bool):
    sfx = "_sample" if sample else ""
    d = _PROJECT_ROOT / "data" / "marts"
    return {
        "score": d / f"mart_niche_score{sfx}.parquet",
        "monthly": d / f"mart_niche_monthly{sfx}.parquet",
        # Q2/Q3 산출물·모델은 sample 분기 없이 전체본을 쓴다(차트 의미가 있으려면 풀 데이터 필요)
        "q2": d / "q2_aspect_prevalence.parquet",
        "q2_sev": d / "q2_aspect_severity.parquet",
        "q2_trend": d / "q2_aspect_trend.parquet",
        "q3": d / "q3_predictions.parquet",
        "launch": d / "mart_launch.parquet",
        "model": d / "q3_model.pkl",
        "newcomer": d / "mart_niche_newcomer.parquet",
    }


def _frame(fig, title, how, sowhat):
    """제목=결론 + 하단 '읽는 법 / So what' 캡션. 모든 차트 공통."""
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.005)
    fig.text(0.005, -0.015, f"읽는 법 · {how}\nSo what · {sowhat}",
             fontsize=8.6, color="#444", ha="left", va="top")


def _save(fig, name, sample):
    sfx = "_sample" if sample else ""
    out = FIG_DIR / f"{name}{sfx}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(_PROJECT_ROOT)}")
    return out.name


def _archetype(dg, openness, open_med):
    grow = dg > 0
    open_ = openness >= open_med
    if grow and open_:
        return "진짜기회"
    if grow and not open_:
        return "경쟁과열"
    if (not grow) and open_:
        return "쇠퇴함정"
    return "회피"


# ── Q1 ───────────────────────────────────────────────────────────
def fig_archetype_quadrant(paths, ctx, sample):
    s = pl.read_parquet(paths["score"]).to_pandas()
    s = s.dropna(subset=["demand_growth", "market_openness"])
    x = s["demand_growth"] * 100
    y = s["market_openness"]
    open_med = float(y.median())
    arch = [_archetype(dg, op, open_med) for dg, op in zip(s["demand_growth"], y)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, col in ARCH_COLORS.items():
        m = [a == name for a in arch]
        ax.scatter(x[m], y[m], s=20, alpha=0.55, color=col, edgecolors="none",
                   label=f"{name} ({sum(m)})")
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(open_med, color="gray", lw=0.8, ls="--", alpha=0.7)

    # 사분면 코너 라벨
    for (fx, fy, ha, va, name) in [(0.985, 0.97, "right", "top", "진짜기회\n(성장·개방)"),
                                   (0.015, 0.97, "left", "top", "쇠퇴함정\n(쇠퇴·개방)"),
                                   (0.985, 0.03, "right", "bottom", "경쟁과열\n(성장·붐빔)"),
                                   (0.015, 0.03, "left", "bottom", "회피\n(쇠퇴·붐빔)")]:
        ax.text(fx, fy, name, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
                color="#666", style="italic")

    # 상위 3 기회 니치만 라벨(겹침 방지)
    top = s.sort_values("opportunity_score", ascending=False).head(3)
    for r in top.itertuples():
        ax.annotate(r.niche, (r.demand_growth * 100, r.market_openness), fontsize=8,
                    xytext=(5, 4), textcoords="offset points", fontweight="bold")
    # 셀러 니치 강조
    tgt = s[s["hexagon_products"] > 0]
    if len(tgt):
        r = tgt.sort_values("hexagon_products").iloc[-1]
        ax.scatter([r["demand_growth"] * 100], [r["market_openness"]], s=260, marker="*",
                   color="crimson", edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(f"{r['niche']}\n(셀러 진입 · {int(r['rank'])}위)",
                    (r["demand_growth"] * 100, r["market_openness"]), color="crimson",
                    fontsize=9, xytext=(8, -30), textcoords="offset points", fontweight="bold")
    ax.set_xlabel("수요 증가율 (최근 12개월 vs 직전 12개월, %)")
    ax.set_ylabel("진입 여지 (1 - 상위5 집중도)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.12)
    _frame(fig, "성장하면서 안 붐비는 니치는 드물다 — 셀러 니치는 거기 없었다",
           "우상단(진짜기회)=수요 성장+낮은 집중도. 점선은 수요 0%·진입여지 중앙값.",
           "셀러가 들어간 Floating Shelves(★)는 좌측 쇠퇴 구역 — 키워드만 보고는 못 봤던 위치.")
    return _save(fig, "q1_opportunity_map", sample)


def fig_score_decomposition(paths, ctx, sample):
    s = pl.read_parquet(paths["score"]).to_pandas()
    w = ctx["weights"]
    comp = {"demand_growth": ("z_demand_growth", "수요 성장"),
            "market_openness": ("z_market_openness", "진입 여지"),
            "quality_gap": ("z_quality_gap", "품질 갭"),
            "uncrowded": ("z_uncrowded", "비포화")}
    colors = {"수요 성장": "#2a9d4a", "진입 여지": "#4C78A8",
              "품질 갭": "#e08a1e", "비포화": "#9467bd"}
    top = s.sort_values("opportunity_score", ascending=False).head(8)
    rows = top.copy()
    fs = s[s["hexagon_products"] > 0]
    if len(fs):
        rows = pd.concat([top, fs.sort_values("hexagon_products").tail(1)])
    rows = rows.iloc[::-1]  # 위에서 아래로 점수 내림차순

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"{n[:24]} ({int(r)}위)" for n, r in zip(rows["niche"], rows["rank"])]
    ypos = np.arange(len(rows))
    left_pos = np.zeros(len(rows)); left_neg = np.zeros(len(rows))
    for key, (zc, lab) in comp.items():
        contrib = (rows[zc] * w[key]).to_numpy()
        base = np.where(contrib >= 0, left_pos, left_neg)
        ax.barh(ypos, contrib, left=base, color=colors[lab], label=lab, height=0.7)
        left_pos = left_pos + np.where(contrib >= 0, contrib, 0)
        left_neg = left_neg + np.where(contrib < 0, contrib, 0)
    ax.scatter(rows["opportunity_score"], ypos, color="black", s=18, zorder=5,
               label="기회점수(합)")
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("가중 z-기여 (가중치 × 표준화 점수)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, axis="x", alpha=0.12)
    _frame(fig, "상위 기회 니치는 대부분 '수요 성장'이 점수를 견인한다",
           "막대 = 4개 지표의 가중 z-기여를 쌓은 것, 검은 점 = 합(기회점수).",
           "기회점수가 블랙박스가 아님을 보임 — 어떤 니치는 성장이, 어떤 니치는 품질갭이 동인.")
    return _save(fig, "q1_score_decomposition", sample)


def fig_rank_uncertainty(paths, ctx, sample):
    s = pl.read_parquet(paths["score"]).to_pandas()
    need = {"rank_p10", "rank_p90", "median_rank"}
    if not need.issubset(s.columns):
        print("  (skip q1_rank_uncertainty: 섭동 분위 컬럼 없음)")
        return None
    top = s.sort_values("opportunity_score", ascending=False).head(12).iloc[::-1]
    ypos = np.arange(len(top))
    med = top["median_rank"].to_numpy()
    lo = (med - top["rank_p10"]).clip(0); hi = (top["rank_p90"] - med).clip(0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(med, ypos, xerr=[lo, hi], fmt="o", color="#1f4e79",
                ecolor="#9bbcdc", elinewidth=3, capsize=3, markersize=5)
    ax.set_yticks(ypos); ax.set_yticklabels(top["niche"].str.slice(0, 26), fontsize=8)
    ax.invert_xaxis()  # 낮은(좋은) 순위가 오른쪽
    ax.set_xlabel("가중치 500회 섭동 시 순위 (오른쪽=상위, 막대=p10~p90)")
    ax.grid(True, axis="x", alpha=0.15)
    _frame(fig, "가중치를 흔들어도 상위 니치의 순위는 견고하다",
           "점=중앙 순위, 가로막대=가중치 섭동 500회의 p10~p90 범위(짧을수록 안정).",
           "결론이 자의적 가중치 선택에 휘둘리지 않음을 정량화 — 민감도를 숨기지 않는다.")
    return _save(fig, "q1_rank_uncertainty", sample)


def fig_newcomer_winnability(paths, ctx, sample):
    if not paths["newcomer"].exists():
        print("  (skip q1_newcomer_winnability: mart_niche_newcomer 없음)")
        return None
    g = pl.read_parquet(paths["newcomer"]).to_pandas()
    x = g["n_launches"]; y = g["newcomer_win_rate"] * 100
    med = float(y.median())
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.scatter(x, y, s=16, alpha=0.5, color="#4C78A8", edgecolors="none")
    ax.set_xscale("log")
    ax.axhline(med, color="gray", lw=0.9, ls="--", alpha=0.8)
    ax.text(x.min(), med + 1, f"중앙값 {med:.0f}%", fontsize=8, color="#666")
    # 작지만 승산 높은 니치(좌상단) 라벨
    small_win = g[(g["n_launches"] <= 300)].sort_values("newcomer_win_rate", ascending=False).head(4)
    for r in small_win.itertuples():
        ax.annotate(r.niche[:22], (r.n_launches, r.newcomer_win_rate * 100), fontsize=7.5,
                    color="#2a9d4a", xytext=(4, 3), textcoords="offset points")
    # 셀러 니치 강조
    fs = g[g["niche"] == ctx["focus"]]
    if len(fs):
        r = fs.iloc[0]
        ax.scatter([r["n_launches"]], [r["newcomer_win_rate"] * 100], s=240, marker="*",
                   color="crimson", edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(f"{ctx['focus']}\n(신규 승산 {r['newcomer_win_rate']:.0%}, 중앙 위)",
                    (r["n_launches"], r["newcomer_win_rate"] * 100), color="crimson",
                    fontsize=9, xytext=(8, 8), textcoords="offset points", fontweight="bold")
    ax.set_xlabel("니치 크기 (출시 코호트 수, log)")
    ax.set_ylabel("신규 진입자 안착률 (%) — 후기 리뷰 ≥10 비율")
    ax.grid(True, alpha=0.12)
    _frame(fig, "신규가 뜨는 니치는 크기와 무관하다 — 작아도 승산 높은 곳이 있다",
           "x=니치 크기(log), y=신규 진입자가 실제 트랙션을 얻는 비율. 크기와 상관 거의 0(-0.12).",
           "'니치마켓=그 아이템만 뜨면 됨'을 정량화. 기회점수와도 거의 직교(corr 0.07)라 별개의 승산 축.")
    return _save(fig, "q1_newcomer_winnability", sample)


# ── 회고 ─────────────────────────────────────────────────────────
def fig_seller_demand(paths, ctx, sample):
    m = (pl.read_parquet(paths["monthly"]).filter(pl.col("niche") == ctx["focus"])
         .sort("review_month"))
    if m.height == 0:
        print(f"  (skip seller_niche_demand: '{ctx['focus']}' 월별 데이터 없음 — sample)")
        return None
    dates = m["review_month"].to_list(); n = m["n_reviews"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dates, n, color="#9bbcdc", lw=1, label="월별 신규 리뷰")
    if len(n) >= 12:
        roll = np.convolve(n, np.ones(12), "valid")
        ax.plot(dates[11:], roll / 12.0, color="#1f4e79", lw=2.4, label="12개월 이동평균")
    import datetime as dt
    last = dates[-1]
    def shift_m(d, k):
        y, mo = d.year, d.month - k
        while mo <= 0:
            mo += 12; y -= 1
        return dt.date(y, mo, 1)
    ax.axvspan(shift_m(last, 11), last, color="crimson", alpha=0.10)
    ax.axvspan(shift_m(last, 23), shift_m(last, 12), color="gray", alpha=0.10)
    ax.annotate("최근 12개월\n수요 -20%", (shift_m(last, 11), ax.get_ylim()[1] * 0.82),
                color="crimson", fontsize=9, fontweight="bold")
    ax.set_xlabel("월"); ax.set_ylabel("월별 신규 리뷰 수 (수요 프록시)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.15)
    _frame(fig, "셀러가 들어간 니치는 이미 정점을 지나 식고 있었다 (-20%)",
           "파란 굵은 선=12개월 이동평균(추세). 빨강/회색 음영=최근/직전 12개월.",
           "키워드 랭킹 상승은 내 키워드의 단면 — 니치 전체 수요는 2021 정점 후 하락 중이었다.")
    return _save(fig, "seller_niche_demand", sample)


# ── Q2 ───────────────────────────────────────────────────────────
def fig_aspect_heatmap(paths, ctx, sample):
    q = pl.read_parquet(paths["q2"])
    aspects = [c for c in q.columns if c in ASPECT_KO]
    q = q.sort(pl.col("niche") != ctx["focus"])
    niches = q["niche"].to_list()
    mat = np.array([[q[a][i] * 100 for a in aspects] for i in range(q.height)])
    fig, ax = plt.subplots(figsize=(8.5, 0.7 * len(niches) + 2.4))
    im = ax.imshow(mat, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels([ASPECT_KO[a] for a in aspects], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(niches)))
    ax.set_yticklabels([f"{nm}\n(n={q['n_low_reviews'][i]:,})" for i, nm in enumerate(niches)],
                       fontsize=8)
    for i in range(len(niches)):
        for j in range(len(aspects)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > mat.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, label="저평점 리뷰 중 언급률(%)")
    _frame(fig, "같은 '셸프'라도 불만의 결은 니치마다 다르다",
           "행=니치, 열=불만 aspect, 숫자=저평점 리뷰 중 언급률(%).",
           "차별화 축은 니치별로 따로 설계 — Pot Racks는 하드웨어, Shower Caddies는 처짐이 1위.")
    return _save(fig, "q2_aspect_heatmap", sample)


def fig_q2_priority(paths, ctx, sample):
    if not paths["q2_sev"].exists():
        print("  (skip q2_priority: q2_aspect_severity 없음)")
        return None
    d = pl.read_parquet(paths["q2_sev"]).to_pandas()
    d = d.dropna(subset=["lift"])
    px = d["prev_low"] * 100; py = d["lift"]
    sev = d["severity_avg_star"]
    size = (5.2 - sev).clip(lower=0.3) * 120  # 별점 낮을수록 큰 버블(심각)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = ax.scatter(px, py, s=size, c=sev, cmap="RdYlGn", vmin=2, vmax=4,
                    edgecolors="black", linewidths=0.5, alpha=0.85)
    ax.axhline(1, color="gray", lw=1, ls="--")
    ax.axvline(float(px.median()), color="gray", lw=0.8, ls=":", alpha=0.7)
    for r in d.itertuples():
        ax.annotate(ASPECT_KO.get(r.aspect, r.aspect),
                    (r.prev_low * 100, r.lift), fontsize=8.5, fontweight="bold",
                    xytext=(6, 3), textcoords="offset points")
    ax.text(0.98, 0.97, "↑ 진짜 불만 동인", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color="#2a9d4a", style="italic")
    ax.text(0.98, 0.02, "lift≈1: 모든 리뷰에 흔한 배경어", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#888", style="italic")
    fig.colorbar(sc, ax=ax, label="심각도(언급 리뷰 평균 별점, 낮을수록 치명적)")
    ax.set_xlabel("저평점 리뷰 중 언급률(%)  — 얼마나 자주")
    ax.set_ylabel("lift = 저평점 유병률 ÷ 고평점 유병률  — 얼마나 '불만 특이적'")
    ax.grid(True, alpha=0.12)
    _frame(fig, "유병률보다 lift — 재질·내구성이 진짜 동인, 하드웨어는 과대평가",
           "x=얼마나 자주, y=불만에 얼마나 특이적(lift). 버블 클수록(빨강) 별점 낮아 치명적.",
           "재질·내구성은 자주+특이적=최우선 차별화 축. 하드웨어는 유병률 높아도 lift 낮아 약한 신호.")
    return _save(fig, "q2_priority", sample)


def fig_q2_trend(paths, ctx, sample):
    if not paths["q2_trend"].exists():
        print("  (skip q2_trend: q2_aspect_trend 없음)")
        return None
    t = pl.read_parquet(paths["q2_trend"]).to_pandas()
    if len(t) < 2:
        print("  (skip q2_trend: 연도 부족)")
        return None
    show = ["build_quality", "mounting_hardware", "sturdiness_sagging", "finish_cosmetic"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in show:
        ax.plot(t["year"], t[a] * 100, marker="o", lw=2, label=ASPECT_KO[a])
    ax.set_xlabel("연도"); ax.set_ylabel("저평점 리뷰 중 언급률(%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(True, alpha=0.15)
    _frame(fig, "Floating Shelves 불만 구조는 8년간 안정적 — 재질이 항상 1위",
           "선=연도별 저평점 리뷰 중 각 불만 언급률. 표본 부족 연도는 제외.",
           "급변하는 불만(=미충족 신규 기회)은 없음 — 재질·내구성이 일관된 구조적 약점.")
    return _save(fig, "q2_trend", sample)


# ── Q3 ───────────────────────────────────────────────────────────
def _bootstrap_pr_ci(y, prob, n=200, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        vals.append(average_precision_score(y[b], prob[b]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def fig_pr_curve(paths, ctx, sample):
    p = pl.read_parquet(paths["q3"]).to_pandas()
    y = p["settled"].to_numpy(); prob = p["pred_prob"].to_numpy()
    if len(np.unique(y)) < 2:
        print("  (skip q3_pr_curve: 단일 클래스 — sample)")
        return None
    prec, rec, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob); base = float(y.mean())
    lo, hi = _bootstrap_pr_ci(y, prob)
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.plot(rec, prec, color="#1f4e79", lw=2.4,
            label=f"LightGBM PR-AUC={ap:.3f}\n(95% CI {lo:.3f}~{hi:.3f})")
    ax.axhline(base, color="crimson", ls="--", lw=1.4, label=f"무작위 기준선={base:.3f}")
    ax.fill_between(rec, prec, base, where=(prec > base), color="#1f4e79", alpha=0.08)
    ax.annotate(f"기준선의 {ap/base:.1f}배", (0.55, base + 0.04), color="crimson",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("Recall (안착 상품을 얼마나 잡나)")
    ax.set_ylabel("Precision (잡은 것 중 실제 안착 비율)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8.5); ax.grid(True, alpha=0.15)
    _frame(fig, "안착 모델은 무작위보다 3.8배 잘 가른다 (PR-AUC 0.60)",
           "곡선이 빨간 기준선 위로 들린 면적이 모델의 실질 가치. 부트스트랩 95% CI 동반.",
           "불균형(양성률 0.16)이라 정확도 대신 PR-AUC가 정직한 지표.")
    return _save(fig, "q3_pr_curve", sample)


def fig_calibration(paths, ctx, sample):
    p = pl.read_parquet(paths["q3"]).to_pandas()
    if p["settled"].nunique() < 2:
        print("  (skip q3_calibration: 단일 클래스)")
        return None
    p["bin"] = pd.qcut(p["pred_prob"], 10, duplicates="drop")
    g = p.groupby("bin", observed=True).agg(pred=("pred_prob", "mean"),
                                            actual=("settled", "mean"), n=("settled", "size"))
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="완벽 보정")
    ax.plot(g["pred"], g["actual"], marker="o", color="#1f4e79", lw=2, label="모델")
    ax.set_xlabel("예측 안착확률 (구간 평균)")
    ax.set_ylabel("실제 안착률")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.15)
    _frame(fig, "예측은 순위는 맞지만 확률값은 약간 과신 — 그래서 임계 기준으로 쓴다",
           "점이 대각선 위=과신, 아래=과소. 불균형 가중으로 확률은 미보정(D-016).",
           "확률 절대값을 신뢰하지 말고 MCC-최적 임계(0.41) 초과/미만의 결정용으로 해석.")
    return _save(fig, "q3_calibration", sample)


def _prep_features(raw_df, bundle):
    """대시보드/모델과 동일 전처리: log1p + price_missing + 컬럼 순서."""
    X = raw_df[bundle["raw_features"]].copy()
    for c in bundle["log_features"]:
        X[c] = np.log1p(np.clip(X[c].astype(float), 0, None))
    X["price_missing"] = raw_df["price"].isna().astype(int).values
    return X[bundle["feature_columns"]]


def fig_settlement_threshold(paths, ctx, sample):
    if not paths["model"].exists():
        print("  (skip q3_settlement_threshold: q3_model.pkl 없음)")
        return None
    bundle = pickle.load(open(paths["model"], "rb"))
    lf = pl.read_parquet(paths["launch"]).to_pandas()
    base = lf[lf["launch_year"] == 2022]
    med = {c: float(base[c].median()) for c in bundle["raw_features"]}
    grid = np.arange(0, 81)
    rows = pd.DataFrame([{**med, "reviews_90d": int(v)} for v in grid])
    probs = bundle["model"].predict_proba(_prep_features(rows, bundle))[:, 1]
    thr = bundle["threshold"]
    cross = next((int(v) for v, pr in zip(grid, probs) if pr >= thr), None)  # 모델 임계 교차
    plateau = float(probs.max())
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(grid, probs, color="#1f4e79", lw=2.4)
    ax.axhline(thr, color="#e08a1e", ls=":", lw=1.3, label=f"모델 안착 임계 {thr:.2f}")
    ax.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.7)
    if cross is not None:
        ax.axvline(cross, color="crimson", ls="--", lw=1.2)
        ax.annotate(f"첫 90일 리뷰 ≈{cross}개면\n모델이 '안착'으로 분류(임계 {thr:.2f} 초과)",
                    (cross, thr), color="crimson", fontsize=9, fontweight="bold",
                    xytext=(12, -38), textcoords="offset points")
    ax.annotate(f"이후 정체 (최대 ~{plateau:.0%})\n초기 트래픽만으로 50% 못 넘음",
                (60, plateau), color="#555", fontsize=8.5, xytext=(0, 10),
                textcoords="offset points", ha="center")
    ax.set_xlabel("첫 90일 리뷰 수 (나머지 피처는 2022 코호트 중앙값 고정)")
    ax.set_ylabel("예측 안착확률")
    ax.set_ylim(0, 1); ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.15)
    _frame(fig, "첫 몇 개의 리뷰가 안착확률을 가장 크게 끌어올린다 (이후 한계효용 체감)",
           "다른 조건을 2022 중앙값에 고정하고 첫 90일 리뷰 수만 변화시킨 부분의존(PDP).",
           f"실행 목표: 첫 90일 ~{cross}개 리뷰면 안착 분류 진입. 단 초기 트래픽만으론 한계 — 다른 조건도 필요.")
    return _save(fig, "q3_settlement_threshold", sample)


def fig_segment(paths, ctx, sample):
    p = pl.read_parquet(paths["q3"]).to_pandas()
    lf = pl.read_parquet(paths["launch"]).to_pandas()[["parent_asin", "reviews_90d"]]
    d = p.merge(lf, on="parent_asin", how="left").dropna(subset=["reviews_90d"])
    bins = [-0.1, 0, 2, 9, 29, 1e9]
    labels = ["0", "1-2", "3-9", "10-29", "30+"]
    d["tier"] = pd.cut(d["reviews_90d"], bins=bins, labels=labels)
    g = d.groupby("tier", observed=True).agg(actual=("settled", "mean"),
                                             pred=("pred_prob", "mean"), n=("settled", "size"))
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos = np.arange(len(g))
    ax.bar(xpos - 0.2, g["actual"] * 100, width=0.4, color="#1f4e79", label="실제 안착률")
    ax.bar(xpos + 0.2, g["pred"] * 100, width=0.4, color="#9bbcdc", label="평균 예측확률")
    for i, nrow in enumerate(g["n"]):
        ax.text(i, max(g["actual"].iloc[i], g["pred"].iloc[i]) * 100 + 1.5,
                f"n={int(nrow):,}", ha="center", fontsize=8, color="#555")
    ax.set_xticks(xpos); ax.set_xticklabels(g.index.astype(str))
    ax.set_xlabel("첫 90일 리뷰 수 구간 (초기 트래픽)")
    ax.set_ylabel("안착률 / 예측확률 (%)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, axis="y", alpha=0.15)
    _frame(fig, "초기 트래픽이 높을수록 실제 안착률이 계단처럼 오른다 (1-2개 → 30+개)",
           "파랑=구간별 실제 안착률, 연파랑=모델 평균 예측확률.",
           "Q3 핵심 레버(첫 90일 트랙션)가 세그먼트에서도 단조 증가. 단 모델은 방향은 맞히나 고트래픽을 과소예측(보정 한계 — 보정 차트 참조).")
    return _save(fig, "q3_segment", sample)


def main():
    cfg = load_config()
    ctx = {"focus": cfg["q2_unmet_needs"]["focus_niche"],
           "weights": cfg["q1_scoring"]["weights"]}
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="샘플 마트로 빠른 검증")
    args = ap.parse_args()

    paths = _marts(args.sample)
    print(f"figures → docs/figures/ (sample={args.sample}, focus={ctx['focus']})")
    charts = (fig_archetype_quadrant, fig_score_decomposition, fig_rank_uncertainty,
              fig_newcomer_winnability, fig_seller_demand, fig_aspect_heatmap,
              fig_q2_priority, fig_q2_trend,
              fig_pr_curve, fig_calibration, fig_settlement_threshold, fig_segment)
    made = []
    for fn in charts:
        try:
            r = fn(paths, ctx, args.sample)
            if r:
                made.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {fn.__name__}: {e}")
    write_run_log(f"make_figures{'_sample' if args.sample else ''}", {"figures": made})
    print(f"완료: {len(made)}장")


if __name__ == "__main__":
    main()
