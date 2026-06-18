#!/usr/bin/env python3
"""
니치 진입 의사결정 대시보드 (Streamlit) — self-serve 탐색기.

실행: streamlit run streamlit_app.py   (또는 streamlit run app/dashboard.py)

소형 마트만 읽는다(배포 가능하도록 263MB dim_product 미사용):
  data/marts/mart_niche_score.parquet      Q1 기회 스코어
  data/marts/q2_aspect_prevalence.parquet  Q2 니치별 불만 유병률
  data/marts/q3_predictions.parquet        Q3 test 코호트 예측
  data/marts/q3_model.pkl                  Q3 학습 모델(입력 폼 예측용)

모든 수치는 분석 파이프라인 산출물에서 직접 로드 → 문서/차트와 동일 소스.
"""
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "marts" / ".matplotlib"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.ingest.common import load_config  # noqa: E402
WEIGHTS = load_config()["q1_scoring"]["weights"]

for _f in ["AppleGothic", "Arial Unicode MS", "AppleSDGothicNeo"]:
    try:
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

MARTS = PROJECT_ROOT / "data" / "marts"
FIGS = PROJECT_ROOT / "docs" / "figures"
ASPECT_KO = {
    "mounting_hardware": "벽고정·하드웨어", "sturdiness_sagging": "처짐·하중",
    "build_quality": "재질·내구성", "finish_cosmetic": "마감·외관",
    "size_fit": "크기·치수", "instructions": "설명서·조립", "shipping_damage": "배송 파손",
}


@st.cache_data
def load_score():
    return pd.read_parquet(MARTS / "mart_niche_score.parquet")


@st.cache_data
def load_q2():
    return pd.read_parquet(MARTS / "q2_aspect_prevalence.parquet")


@st.cache_data
def load_q3_pred():
    return pd.read_parquet(MARTS / "q3_predictions.parquet")


@st.cache_resource
def load_q3_model():
    p = MARTS / "q3_model.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


# ── 탭 1: Q1 기회 ────────────────────────────────────────────────
def tab_q1():
    s = load_score()
    st.subheader("Q1 — 니치 기회 스코어")
    st.caption("성장하면서(수요증가율↑) 덜 붐비는(진입여지↑) 니치가 기회. "
               "리뷰≠판매(L-1)라 수요는 *증가율*로만 해석.")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    x, y, c = s["demand_growth"] * 100, s["market_openness"], s["opportunity_score"]
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=20, alpha=0.6, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="기회 점수")
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    tgt = s[s["hexagon_products"] > 0]
    if len(tgt):
        r = tgt.sort_values("hexagon_products").iloc[-1]
        ax.scatter([r["demand_growth"] * 100], [r["market_openness"]], s=220,
                   marker="*", color="crimson", edgecolors="black", zorder=5)
        ax.annotate(f"{r['niche']} ({int(r['rank'])}위)",
                    (r["demand_growth"] * 100, r["market_openness"]),
                    color="crimson", fontsize=9, xytext=(6, -16),
                    textcoords="offset points", fontweight="bold")
    ax.set_xlabel("수요 증가율 (%)"); ax.set_ylabel("진입 여지 (1 - 상위5 집중도)")
    ax.grid(True, alpha=0.15)
    st.pyplot(fig)

    st.markdown("**니치 드릴다운**")
    pick = st.selectbox("니치 선택", s.sort_values("rank")["niche"].tolist())
    row = s[s["niche"] == pick].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기회점수 랭크", f"{int(row['rank'])} / {len(s)}")
    c2.metric("수요 증가율", f"{row['demand_growth']*100:.0f}%")
    c3.metric("진입 여지", f"{row['market_openness']:.2f}")
    c4.metric("저평점 비율", f"{row['low_rating_share']*100:.0f}%")
    st.caption("순위 안정성(prob_top_n): 가중치를 흔들었을 때 상위5에 드는 빈도 = "
               f"**{row['prob_top_n']:.2f}**")

    # 점수 분해 — 이 니치 점수를 무엇이 견인하나
    comp = {"z_demand_growth": ("수요 성장", "demand_growth"),
            "z_market_openness": ("진입 여지", "market_openness"),
            "z_quality_gap": ("품질 갭", "quality_gap"),
            "z_uncrowded": ("비포화", "uncrowded")}
    if all(z in s.columns for z in comp):
        labels = [v[0] for v in comp.values()]
        contrib = [row[z] * WEIGHTS[v[1]] for z, v in comp.items()]
        fig2, ax2 = plt.subplots(figsize=(6.5, 2.6))
        colors = ["#2a9d4a" if c >= 0 else "#c44e52" for c in contrib]
        ax2.barh(labels, contrib, color=colors)
        ax2.axvline(0, color="gray", lw=0.8)
        ax2.set_title(f"'{pick}' 기회점수 분해 (가중 z-기여, 합={row['opportunity_score']:+.2f})",
                      fontsize=10)
        st.pyplot(fig2)
        st.caption("초록=점수를 올린 요인, 빨강=내린 요인. 같은 점수라도 견인 요인은 니치마다 다르다.")

    # 신규 진입자 안착률(승산) — 크기와 별개의 축
    ncp = MARTS / "mart_niche_newcomer.parquet"
    if ncp.exists():
        nc = pd.read_parquet(ncp)
        hit = nc[nc["niche"] == pick]
        if len(hit):
            wr = float(hit["newcomer_win_rate"].iloc[0]); med = float(nc["newcomer_win_rate"].median())
            st.caption(f"🆕 **신규 진입자 안착률** = {wr:.0%} (전체 중앙값 {med:.0%}) — "
                       "이 니치에서 신규 상품이 실제로 트랙션을 얻는 비율(크기와 거의 무관, 기회점수와 직교).")
        if (FIGS / "q1_newcomer_winnability.png").exists():
            st.image(str(FIGS / "q1_newcomer_winnability.png"))

    st.markdown("**기회점수 상위 니치**")
    show = s.sort_values("opportunity_score", ascending=False).head(20)[
        ["rank", "niche", "opportunity_score", "demand_growth",
         "market_openness", "low_rating_share", "prob_top_n"]].copy()
    show["demand_growth"] = (show["demand_growth"] * 100).round(0)
    st.dataframe(show, use_container_width=True, hide_index=True)


# ── 탭 2: Q2 미충족 니즈 ─────────────────────────────────────────
def tab_q2():
    q = load_q2()
    st.subheader("Q2 — 미충족 니즈 (저평점 리뷰 불만 유병률)")
    st.caption("같은 '셸프'라도 니치마다 불만의 결이 다르다 → 차별화 축을 니치별로 설계.")
    aspects = [c for c in q.columns if c in ASPECT_KO]

    pick = st.selectbox("니치 선택", q["niche"].tolist())
    row = q[q["niche"] == pick].iloc[0]
    vals = pd.Series({ASPECT_KO[a]: row[a] * 100 for a in aspects}
                     ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(vals.index, vals.values, color="#c44e52")
    for i, v in enumerate(vals.values):
        ax.text(v + 0.4, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("저평점 리뷰 중 언급 비율 (%)")
    ax.set_title(f"{pick} 불만 유병률 (저평점 n={int(row['n_low_reviews']):,})")
    st.pyplot(fig)

    st.markdown("**니치 간 비교 (%)**")
    comp = q[["niche", "n_low_reviews"] + aspects].copy()
    for a in aspects:
        comp[a] = (comp[a] * 100).round(0)
    comp = comp.rename(columns={a: ASPECT_KO[a] for a in aspects})
    st.dataframe(comp, use_container_width=True, hide_index=True)

    sevp = MARTS / "q2_aspect_severity.parquet"
    if sevp.exists():
        st.markdown("**유병률 vs lift — 진짜 불만 동인 가리기** (focus 니치)")
        sv = pd.read_parquet(sevp)
        sv["불만"] = sv["aspect"].map(ASPECT_KO).fillna(sv["aspect"])
        show = sv[["불만", "prev_low", "prev_high", "lift", "severity_avg_star"]].copy()
        show.columns = ["불만", "저평점 유병률", "고평점 유병률", "lift", "심각도(평균★)"]
        st.dataframe(show.round(3), use_container_width=True, hide_index=True)
        st.caption("lift = 저평점÷고평점 유병률. lift≫1이면 진짜 불만 동인, lift≈1이면 모든 리뷰에 "
                   "흔한 배경어 → 유병률이 높아도 차별화 효력은 약하다.")
        if (FIGS / "q2_priority.png").exists():
            st.image(str(FIGS / "q2_priority.png"))


# ── 탭 3: Q3 안착 예측 (정적 탐색 + 입력 폼) ──────────────────────
def tab_q3():
    st.subheader("Q3 — 시장 안착 예측")
    sub = st.radio("보기", ["예측 탐색", "내 조건으로 예측해보기"], horizontal=True)

    if sub == "예측 탐색":
        p = load_q3_pred()
        st.caption("test(2022) 코호트의 예측 안착확률 분포와 실제 안착률. "
                   "모집단은 '리뷰 ≥1 받은 상품'(생존편향 L-2).")
        niches = ["(전체)"] + sorted(p["niche"].unique().tolist())
        pick = st.selectbox("필터 및 검색", niches,
                            help="목록에서 고르거나, 칸에 니치 이름을 타이핑해 검색하세요.")
        d = p if pick == "(전체)" else p[p["niche"] == pick]
        c1, c2, c3 = st.columns(3)
        c1.metric("상품 수", f"{len(d):,}")
        c2.metric("실제 안착률", f"{d['settled'].mean()*100:.1f}%")
        c3.metric("평균 예측확률", f"{d['pred_prob'].mean():.3f}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(d[d["settled"] == 1]["pred_prob"], bins=30, alpha=0.6,
                label="실제 안착", color="#1f4e79", density=True)
        ax.hist(d[d["settled"] == 0]["pred_prob"], bins=30, alpha=0.6,
                label="실제 정체", color="#c44e52", density=True)
        ax.set_xlabel("예측 안착확률"); ax.set_ylabel("밀도")
        ax.legend(); ax.set_title("예측확률 분포 — 안착 vs 정체 분리도")
        st.pyplot(fig)
        st.markdown("**심화 진단**")
        cols = st.columns(3)
        for c, (png, cap) in zip(cols, [
                ("q3_calibration.png", "보정: 확률은 미보정 → 임계 기준 사용"),
                ("q3_settlement_threshold.png", "안착 임계: 첫 90일 리뷰의 한계효용"),
                ("q3_segment.png", "세그먼트: 트래픽 따라 안착률 6%→92%")]):
            if (FIGS / png).exists():
                c.image(str(FIGS / png), caption=cap)
        return

    # 입력 폼 → 모델 예측
    bundle = load_q3_model()
    if bundle is None:
        st.warning("q3_model.pkl 이 없습니다. `python src/analyze/settlement_model.py` 를 먼저 실행하세요.")
        return
    st.caption("첫 90일 조건과 출시 시점 니치 상태를 입력하고 **[안착 확률 예측]** 버튼을 누르면 "
               "결과가 나온다. 피처는 모두 point-in-time(출시 시점/첫 90일).")
    with st.form("q3_predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            reviews_90d = st.number_input("첫 90일 리뷰 수", 0, 2000, 20)
            avg_rating_90d = st.slider("첫 90일 평균 평점", 1.0, 5.0, 4.2, 0.1)
            low_share_90d = st.slider("첫 90일 저평점 비율", 0.0, 1.0, 0.10, 0.01)
            price_unknown = st.checkbox("가격 모름(결측)", value=False)
            price_val = st.number_input("가격($)", 0.0, 1000.0, 25.0)
        with col2:
            niche_products_before = st.number_input("출시 시점 니치 누적 상품수", 0, 20000, 200)
            niche_reviews_before = st.number_input("출시 시점 니치 누적 리뷰수", 0, 2000000, 5000)
            niche_avg_rating_before = st.slider("출시 시점 니치 평균평점", 1.0, 5.0, 4.3, 0.1)
        submitted = st.form_submit_button("안착 확률 예측", type="primary",
                                          use_container_width=True)

    if not submitted:
        st.info("조건을 설정한 뒤 **[안착 확률 예측]** 버튼을 누르세요.")
        return

    price = np.nan if price_unknown else price_val
    raw = {
        "price": price, "reviews_90d": reviews_90d, "avg_rating_90d": avg_rating_90d,
        "low_share_90d": low_share_90d, "niche_products_before": niche_products_before,
        "niche_reviews_before": niche_reviews_before,
        "niche_avg_rating_before": niche_avg_rating_before,
    }
    X = pd.DataFrame([{c: raw[c] for c in bundle["raw_features"]}])
    for c in bundle["log_features"]:                     # settlement_model.make_features 와 동일
        X[c] = np.log1p(np.clip(X[c].astype(float), 0, None))
    X["price_missing"] = pd.isna(pd.Series([raw["price"]])).astype(int).values
    X = X[bundle["feature_columns"]]                     # 모델 기대 컬럼 순서
    prob = float(bundle["model"].predict_proba(X)[:, 1][0])
    thr = bundle["threshold"]

    # 결과: 게이지(임시 — plotly go.Indicator). 임계값을 기준으로 색 구간 분할.
    import plotly.graph_objects as go
    g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 40}},
        delta={"reference": thr * 100, "suffix": "p",
               "increasing": {"color": "#2a9d4a"}, "decreasing": {"color": "#c44e52"}},
        title={"text": "예측 안착 확률 (vs 임계값)", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#1f4e79"},
            "steps": [{"range": [0, thr * 100], "color": "#f7dada"},
                      {"range": [thr * 100, 100], "color": "#d9efdc"}],
            "threshold": {"line": {"color": "#e08a1e", "width": 4},
                          "thickness": 0.85, "value": thr * 100},
        },
    ))
    g.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
    st.plotly_chart(g, use_container_width=True)

    if prob >= thr:
        st.success(f"**안착 가능성 있음** — 예측 {prob:.1%} ≥ 임계 {thr:.0%}. 초기 조건이 안착 신호에 부합.")
    else:
        st.error(f"**안착 미달** — 예측 {prob:.1%} < 임계 {thr:.0%}. "
                 "첫 90일 리뷰 확보·초기 평점 관리·리스팅 보강 후 재검토 권장.")
    st.caption(f"주황선=모델 안착 임계({thr:.0%}, MCC 최적). 모델 특성상 예측은 ~50%에서 정체하므로 "
               "절대값보다 임계 초과 여부로 해석. test PR-AUC {:.3f}(기준선 {:.3f})·MCC {:.3f}, 확률 미보정(D-016)."
               .format(bundle["test_metrics"]["pr_auc"], bundle["test_metrics"]["base_rate"],
                       bundle["test_metrics"]["mcc"]))


def main():
    st.set_page_config(page_title="아마존 니치 진입 대시보드", layout="wide")
    st.title("전직 셀러의 카테고리 진입 의사결정 대시보드")
    st.markdown("Q1 기회 → Q2 차별화 → Q3 안착, 세 렌즈를 직접 탐색. "
                "데이터: Amazon Reviews 2023 (Home & Kitchen).")
    t1, t2, t3 = st.tabs(["① Q1 기회", "② Q2 미충족 니즈", "③ Q3 안착 예측"])
    with t1:
        tab_q1()
    with t2:
        tab_q2()
    with t3:
        tab_q3()


if __name__ == "__main__":
    main()
