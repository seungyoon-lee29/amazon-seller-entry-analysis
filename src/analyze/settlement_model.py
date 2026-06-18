"""Stage 5 / Q3 — 시장 안착 예측 모델.

출시 코호트(mart_launch)로 "어떤 초기 조건이 12개월 내 시장 안착을 가르는가"를
분류·해석한다. 신규 셀러의 진입 의사결정(이 조건이면 안착 가능성이 높은가)에 연결한다.

설계 규율(상위 문서 R2/R4, decisions.md D-015):
  · 생존편향(L-2): 모집단은 "리뷰 ≥1 받은 상품"이므로 타깃을 "성공"이 아니라
    **"시장 안착 vs 조기 정체"**로 명명한다.
  · 누수 방지: 라벨을 *초기창 이후*(출시 90~365일) 트랙션으로 정의한다. 피처
    reviews_90d(첫 90일)가 라벨에 기계적으로 포함되지 않게 하기 위함.
    → 90일 트랙션이 후기 트랙션을 예측하는 것은 "정당한 조기 신호"(누수 아님).
  · point-in-time: 피처는 출시 시점/첫 90일에 알 수 있는 것만.
  · 시간 기반 split: 과거 코호트로 학습 → 미래 코호트 예측(랜덤 split 금지).

"안착"의 조작적 정의(config q3_settlement.label):
  later_reviews(90~365일) ≥ max(니치별 P75, min_floor)

모델: LogisticRegression(베이스라인·해석) → LightGBM(주모델). 불균형이므로
PR-AUC를 1차 지표로, 임계값은 검증셋에서 MCC 최대화로 고른 뒤 test에 적용해
precision/recall/MCC/혼동행렬을 보고. SHAP로 피처 기여를 해석한다.

출력:
  data/marts/q3_predictions[_sample].parquet   test 코호트 예측확률
  docs/q3_settlement_model[_sample].md          모델 카드(정의·split·지표·SHAP·해석)
  docs/figures/q3_shap_summary[_sample].png     SHAP beeswarm(가능 시)

사용:
    python src/analyze/settlement_model.py            # 전체
    python src/analyze/settlement_model.py --sample   # 스모크
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPL_CACHE = _PROJECT_ROOT / "data" / "marts" / ".matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.ingest.common import PROJECT_ROOT, load_config, write_run_log  # noqa: E402


def build_dataset(con, launch_path: Path, cfg_q3: dict) -> pd.DataFrame:
    """mart_launch에서 라벨·피처를 만든다.

    누수 차단 두 겹:
      ① 라벨 윈도우: 90~365일 트랙션(`reviews_12m - reviews_90d`) — 피처 reviews_90d와
         기계적으로 겹치지 않음 (D-015).
      ② 니치 임계: **train 연도만으로** 계산한 니치별 P75를 valid/test에 적용 — 임계가
         미래(test) 결과를 엿보지 못하게 함 (감사에서 발견된 잔여 누수, D-016).
    평점 게이트(avg_rating_12m)는 피처 avg_rating_90d와 corr≈0.92라 라벨-피처 누수가
    되어 **제거**한다. 안착은 "지속되는 후기 트랙션 볼륨"으로만 정의(D-016).
    """
    lab = cfg_q3["label"]
    train_max = cfg_q3["split"]["train_max_year"]
    df = con.execute(f"""
        WITH base AS (
            SELECT *, (reviews_12m - reviews_90d) AS later_reviews
            FROM '{launch_path}'
        ),
        niche_thr AS (  -- 니치별 후기 트랙션 P75 — train 연도만으로 산정(누수 방지)
            SELECT niche,
                   quantile_cont(later_reviews, {lab['later_window_quantile']}) AS thr
            FROM base
            WHERE launch_year <= {train_max}
            GROUP BY niche
        )
        SELECT b.*,
               greatest(coalesce(t.thr, 0), {lab['min_floor']}) AS settle_thr,
               CASE WHEN b.later_reviews
                         >= greatest(coalesce(t.thr, 0), {lab['min_floor']})
                    THEN 1 ELSE 0 END AS settled
        FROM base b
        LEFT JOIN niche_thr t USING (niche)
    """).df()
    return df


def make_features(df: pd.DataFrame, cfg_q3: dict) -> tuple[pd.DataFrame, list[str]]:
    """피처 행렬 생성: log1p 변환 + price 결측 지시자. point-in-time 컬럼만 사용."""
    feats = list(cfg_q3["features"])
    X = df[feats].copy()
    for c in cfg_q3["log_features"]:
        X[c] = np.log1p(X[c].clip(lower=0))
    # price 결측은 65%에 달함(파싱 실패) → 결측 자체가 신호일 수 있어 지시자 추가
    X["price_missing"] = df["price"].isna().astype(int)
    return X, list(X.columns)


def time_split(df: pd.DataFrame, X: pd.DataFrame, cfg_q3: dict):
    sp = cfg_q3["split"]
    y = df["settled"].to_numpy()
    yr = df["launch_year"].to_numpy()
    tr = yr <= sp["train_max_year"]
    va = yr == sp["valid_year"]
    te = yr == sp["test_year"]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te]), te


def validate_binary_split(name: str, X: pd.DataFrame, y: np.ndarray, min_rows: int):
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if len(X) < min_rows or n_pos == 0 or n_neg == 0:
        sys.exit(f"{name} split 표본 부족/단일 클래스 "
                 f"(n={len(X)}, 양성={n_pos}, 음성={n_neg}) — Q3를 검증할 수 없습니다.")


def best_threshold_mcc(y_true, prob) -> float:
    """검증셋에서 MCC를 최대화하는 임계값을 고른다(불균형 분류의 임계 선택)."""
    grid = np.unique(np.quantile(prob, np.linspace(0.5, 0.99, 50)))
    best_t, best_m = 0.5, -1.0
    for t in grid:
        m = matthews_corrcoef(y_true, (prob >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t)


def eval_at(y_true, prob, thr) -> dict:
    pred = (prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "threshold": float(thr),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "pos_rate": float(np.mean(y_true)),
    }


def fit_logreg(Xtr, ytr, Xte):
    """베이스라인: 중앙값 대치 + 표준화 + 로지스틱 회귀(class_weight balanced)."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(Xtr, ytr)
    return pipe, pipe.predict_proba(Xte)[:, 1]


def fit_lgbm(Xtr, ytr, Xva, yva, cfg_lgbm):
    """주모델: LightGBM(결측 native 처리, 불균형 가중, 검증셋 early stopping)."""
    import lightgbm as lgb
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=cfg_lgbm["n_estimators"],
        learning_rate=cfg_lgbm["learning_rate"],
        num_leaves=cfg_lgbm["num_leaves"],
        min_child_samples=cfg_lgbm["min_child_samples"],
        scale_pos_weight=pos_w,
        random_state=42, n_jobs=1, verbose=-1,
        # 재현성: 멀티스레드 비결정성 제거(재실행 시 수치 고정) — D-019
        deterministic=True, force_row_wise=True,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(cfg_lgbm["early_stopping_rounds"], verbose=False)])
    return model


def ensure_mpl_cache():
    _MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)


def shap_table(model, X: pd.DataFrame, n: int, seed: int):
    """SHAP로 피처 기여(평균 |SHAP|)와 방향(피처값과 SHAP의 상관)을 계산."""
    ensure_mpl_cache()
    import shap
    Xs = X.sample(min(n, len(X)), random_state=seed) if len(X) > n else X
    expl = shap.TreeExplainer(model)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        sv = expl.shap_values(Xs)
    if isinstance(sv, list):           # 일부 버전은 클래스별 리스트 반환
        sv = sv[1]
    sv = np.asarray(sv)
    rows = []
    for i, c in enumerate(Xs.columns):
        col = Xs[c].to_numpy(dtype=float)
        mask = np.isfinite(col)
        direction = (np.corrcoef(col[mask], sv[mask, i])[0, 1]
                     if mask.sum() > 2 and np.std(col[mask]) > 0 else np.nan)
        rows.append({"feature": c,
                     "mean_abs_shap": float(np.abs(sv[:, i]).mean()),
                     "direction": float(direction)})
    tbl = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return tbl, Xs, sv


def save_shap_plot(sv, Xs, out_path: Path) -> bool:
    try:
        ensure_mpl_cache()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(sv, Xs, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        return True
    except Exception as e:
        print(f"  [shap plot skipped] {type(e).__name__}: {e}")
        return False


def write_model_card(stats, base_m, lgbm_m, shap_tbl, cfg_q3, plot_ok,
                     plot_rel, out_path: Path):
    lab, sp = cfg_q3["label"], cfg_q3["split"]
    label_kr = {
        "reviews_90d": "첫 90일 리뷰 수(log)", "avg_rating_90d": "첫 90일 평균 평점",
        "low_share_90d": "첫 90일 저평점 비율", "price": "가격",
        "price_missing": "가격 결측 여부",
        "niche_products_before": "출시 시점 니치 상품수(log)",
        "niche_reviews_before": "출시 시점 니치 리뷰수(log)",
        "niche_avg_rating_before": "출시 시점 니치 평균평점",
    }
    dir_kr = lambda d: "↑(높을수록 안착)" if d > 0.05 else ("↓(높을수록 정체)" if d < -0.05 else "~중립")
    # make_figures.py 가 생성하는 PR 커브(있으면 임베드)
    pr_rel = plot_rel.replace("q3_shap_summary", "q3_pr_curve")
    pr_ok = (out_path.parent / pr_rel).exists()

    def mrow(name, m):
        return (f"| {name} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['brier']:.3f} | "
                f"{m['precision']:.2f} | {m['recall']:.2f} | {m['mcc']:.3f} |")

    L = [
        "# Q3 — 시장 안착 예측 모델 (모델 카드)",
        "",
        "> 자동 생성: `src/analyze/settlement_model.py` · 단위: 출시 코호트(mart_launch)",
        "",
        "## 문제 정의와 모집단 (정직성)",
        "",
        "- **타깃**: 출시 후 *안착*(later traction 지속) vs *조기 정체*. 생존편향(L-2)상 "
        "리뷰 0개로 사라진 상품은 데이터에 없으므로 '성공 예측'이 아니라 "
        "'리뷰를 받은 상품 중 안착 vs 정체'다.",
        "- **안착의 조작적 정의**: 출시 90~365일 리뷰 수가 니치별 "
        f"P{int(lab['later_window_quantile']*100)}(train 연도로 산정) 이상"
        f"(하한 {lab['min_floor']}). 평점 게이트는 피처와의 누수(corr≈0.92)로 제거 — "
        "안착은 '지속되는 후기 트랙션 볼륨'으로만 정의(D-016).",
        f"- **양성률**: train {stats['pos_rate_train']:.1%} / valid "
        f"{stats['pos_rate_valid']:.1%} / test {stats['pos_rate_test']:.1%} "
        "(시간에 따른 드리프트 존재 — 오래된 코호트가 성숙).",
        "",
        "## 누수 방지 설계 (R4 / D-015 / D-016)",
        "",
        "- **라벨 윈도우**: 초기창 이후(90~365일) 트랙션 → 피처 `reviews_90d`(첫 90일)가 "
        "라벨에 기계적으로 포함되지 않음.",
        f"  점검: corr(reviews_90d, 후기 트랙션)={stats['corr_90d_later']:.2f} "
        f"vs corr(reviews_90d, 12개월 전체)={stats['corr_90d_12m']:.2f}. 후기창 정의가 "
        "기계적 성분을 줄임. 남은 상관은 '초기 견인 → 후기 견인'의 정당한 신호(누수 아님).",
        "- **니치 임계**: train 연도만으로 산정한 P75를 valid/test에 적용 → 라벨이 미래(test) "
        "결과를 엿보지 않음(D-016, 감사에서 발견된 잔여 누수 수정).",
        "- **평점 게이트 제거**: avg_rating_12m(라벨 후보)는 피처 avg_rating_90d와 corr≈0.92로 "
        "라벨-피처 누수 → 제거. 안착은 후기 트랙션 볼륨으로만 정의(D-016).",
        "- 모든 피처는 출시 시점/첫 90일에 관측 가능. `launch_year`는 피처가 아니라 split 키.",
        "",
        "## 검증 설계",
        "",
        f"- **시간 기반 split**: train(≤{sp['train_max_year']}, n={stats['n_train']:,}) → "
        f"valid({sp['valid_year']}, n={stats['n_valid']:,}) → "
        f"test({sp['test_year']}, n={stats['n_test']:,}). 랜덤 split 금지.",
        "- 불균형 → **PR-AUC를 1차 지표**. 임계값은 valid에서 **MCC 최대화**로 선택 후 test 적용.",
        "",
        "## 성능 (test 셋)",
        "",
        "| 모델 | PR-AUC | ROC-AUC | Brier | Precision | Recall | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        mrow("LogReg (베이스라인)", base_m),
        mrow("LightGBM (주모델)", lgbm_m),
        "",
        f"- test 양성률(무작위 기준선 PR-AUC) = **{lgbm_m['pos_rate']:.3f}**. "
        f"LightGBM PR-AUC {lgbm_m['pr_auc']:.3f} → 기준선 대비 "
        f"{lgbm_m['pr_auc']/max(lgbm_m['pos_rate'],1e-9):.1f}배.",
        f"- 선택 임계값 {lgbm_m['threshold']:.3f}에서 혼동행렬(test): "
        f"TP={lgbm_m['tp']:,} FP={lgbm_m['fp']:,} FN={lgbm_m['fn']:,} TN={lgbm_m['tn']:,}.",
    ]
    if pr_ok:
        L += ["", f"![Q3 PR 커브]({pr_rel})",
              "> 불균형(양성률 0.157)이라 정확도가 아닌 PR-AUC가 정직한 지표. "
              "곡선이 기준선 위로 들린 면적이 모델의 실질 가치."]
    L += [
        "",
        "## 피처 기여 (SHAP · LightGBM)",
        "",
        "| 피처 | 평균 |SHAP| | 방향 |",
        "|---|---:|---|",
    ]
    for r in shap_tbl.itertuples():
        L.append(f"| {label_kr.get(r.feature, r.feature)} | {r.mean_abs_shap:.3f} | "
                 f"{dir_kr(r.direction)} |")
    if plot_ok:
        L += ["", f"![SHAP summary]({plot_rel})"]

    # 심화 진단 차트(make_figures.py 생성). 있으면 임베드.
    depth = [(pr_rel.replace("q3_pr_curve", "q3_calibration"), "보정 곡선",
              "예측 확률은 순위는 맞으나 미보정(과신/압축) → 절대값 대신 임계 기준으로 사용"),
             (pr_rel.replace("q3_pr_curve", "q3_settlement_threshold"), "안착 임계(PDP)",
              "첫 90일 리뷰가 안착확률을 가장 크게 끌어올리되 한계효용 체감 — 초기 트래픽만으론 50% 못 넘음"),
             (pr_rel.replace("q3_pr_curve", "q3_segment"), "초기 트래픽 세그먼트",
              "실제 안착률이 트래픽 구간 따라 6%→92%로 단조 증가(모델은 고트래픽 과소예측)")]
    have = [(r, t, d) for r, t, d in depth if (out_path.parent / r).exists()]
    if have:
        L += ["", "## 심화 진단 (D-019)", ""]
        for rel, title, note in have:
            L += [f"**{title}** — {note}", "", f"![{title}]({rel})", ""]

    top_feats = [label_kr.get(f, f) for f in shap_tbl["feature"].head(3)]
    L += [
        "",
        "## 비즈니스 해석 (So what — 신규 셀러 진입 의사결정)",
        "",
        f"- 안착을 가르는 가장 큰 초기 레버는 **{top_feats[0]}**(압도적), 그 다음 "
        f"{top_feats[1]}·{top_feats[2]}. 즉 출시 직후 트랙션이 안착 확률을 가장 크게 "
        "좌우하고, 초기 평점은 부차적으로 기여한다.",
        "- 진입 결정에 쓰는 법: 첫 90일 지표를 이 모델에 넣어 안착 확률을 추정 → 임계 미만이면 "
        "리스팅/가격/초기 리뷰 확보 전략을 보강하거나 진입을 재고. (Q2 미충족 니즈가 "
        "초기 평점 방어의 구체 레버를 제공)",
        "",
        "## 한계",
        "",
        "- 라벨이 니치 상대 P75 기준 → '안착' 정의가 임계에 민감(분위/하한은 config화).",
        "- 불균형 가중(scale_pos_weight/class_weight)으로 예측확률은 보정(calibration)되지 "
        "않음 → Brier는 참고치. 순위 지표(PR-AUC·ROC-AUC)와 임계 기반 지표(MCC) 위주로 해석.",
        "- 출시일=첫 리뷰일 프록시(L-3): 첫 90일 창이 실제 출시 후 시점과 어긋날 수 있음.",
        "- 리뷰≠판매(L-1), 생존편향(L-2)은 모집단·타깃 명명으로만 완화되며 제거되지 않음.",
    ]
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    cfg = load_config()
    q3 = cfg["q3_settlement"]
    seed = cfg["runtime"]["seed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="_sample 마트 사용")
    args = ap.parse_args()

    marts = PROJECT_ROOT / cfg["paths"]["marts"]
    sfx = "_sample" if args.sample else ""
    launch = marts / f"mart_launch{sfx}.parquet"
    if not launch.exists():
        sys.exit(f"입력 없음: {launch} — transform을 먼저 실행하세요.")

    con = duckdb.connect()
    df = build_dataset(con, launch, q3)
    # 재현성: DuckDB 스캔이 행 순서를 보장하지 않아 LightGBM 히스토그램이 미세 변동 → 고정(D-019)
    df = df.sort_values("parent_asin").reset_index(drop=True)
    X, feat_cols = make_features(df, q3)
    (Xtr, ytr), (Xva, yva), (Xte, yte), te_mask = time_split(df, X, q3)

    validate_binary_split("train", Xtr, ytr, 50)
    validate_binary_split("valid", Xva, yva, 10)
    validate_binary_split("test", Xte, yte, 10)

    # 모델 적합. numpy 2.x matmul이 macOS Accelerate에서 유한 입력에도 내는
    # 허위 FPE 경고를 억제(D-012와 동일 현상 — sklearn 솔버/corrcoef 내부).
    with np.errstate(all="ignore"):
        _, base_prob = fit_logreg(Xtr, ytr, Xte)
        lgbm = fit_lgbm(Xtr, ytr, Xva, yva, q3["lgbm"])
        va_prob = lgbm.predict_proba(Xva)[:, 1]
        te_prob = lgbm.predict_proba(Xte)[:, 1]

        thr = best_threshold_mcc(yva, va_prob)     # 임계값은 valid에서 선택
        base_thr = best_threshold_mcc(yva, fit_logreg(Xtr, ytr, Xva)[1])
        base_m = eval_at(yte, base_prob, base_thr)
        lgbm_m = eval_at(yte, te_prob, thr)

        # SHAP 해석
        shap_tbl, Xs, sv = shap_table(lgbm, Xte, q3["shap_sample"], seed)
    plot_rel = f"figures/q3_shap_summary{sfx}.png"
    plot_ok = save_shap_plot(sv, Xs, PROJECT_ROOT / "docs" / plot_rel)

    # 누수 점검 상관(보고용)
    corr = con.execute(f"""SELECT corr(reviews_90d, reviews_12m-reviews_90d) a,
        corr(reviews_90d, reviews_12m) b FROM '{launch}'""").fetchone()

    stats = {
        "sample": args.sample,
        "n_train": len(Xtr), "n_valid": len(Xva), "n_test": len(Xte),
        "pos_rate_train": float(ytr.mean()), "pos_rate_valid": float(yva.mean()),
        "pos_rate_test": float(yte.mean()),
        "corr_90d_later": float(corr[0]), "corr_90d_12m": float(corr[1]),
        "lgbm_pr_auc": round(lgbm_m["pr_auc"], 4),
        "lgbm_roc_auc": round(lgbm_m["roc_auc"], 4),
        "lgbm_mcc": round(lgbm_m["mcc"], 4),
        "base_pr_auc": round(base_m["pr_auc"], 4),
        "top_feature": shap_tbl.iloc[0]["feature"],
    }

    # 예측 저장 (test 코호트)
    out_pred = df.loc[te_mask, ["parent_asin", "niche", "launch_year",
                                "settled"]].copy()
    out_pred["pred_prob"] = te_prob
    out_pred.to_parquet(marts / f"q3_predictions{sfx}.parquet", index=False)

    # 학습된 모델 + 전처리 메타 + 임계값 저장 (대시보드 입력 폼이 로드해 동일 전처리 재현)
    import pickle
    bundle = {
        "model": lgbm,
        "threshold": float(thr),
        "raw_features": list(q3["features"]),       # 입력 폼이 받는 원본 피처
        "log_features": list(q3["log_features"]),   # log1p(clip≥0) 적용 대상
        "feature_columns": list(feat_cols),         # 모델이 기대하는 최종 컬럼 순서
        "test_metrics": {"pr_auc": lgbm_m["pr_auc"], "roc_auc": lgbm_m["roc_auc"],
                         "mcc": lgbm_m["mcc"], "base_rate": float(yte.mean())},
    }
    with open(marts / f"q3_model{sfx}.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"  model → data/marts/q3_model{sfx}.pkl (thr={thr:.3f})")

    write_model_card(stats, base_m, lgbm_m, shap_tbl, q3, plot_ok, plot_rel,
                     PROJECT_ROOT / "docs" / f"q3_settlement_model{sfx}.md")
    write_run_log(f"q3_settlement{sfx}", stats)
    print(f"done: {stats}")
    print(f"  model card → docs/q3_settlement_model{sfx}.md")


if __name__ == "__main__":
    main()
