"""Stage 4 / Q2 — 미충족 니즈 텍스트 마이닝.

저평점(≤N★) 리뷰에서 "고객이 반복적으로 불평하는 것"을 뽑아 차별화 진입점을 도출한다.
셀러 주력 니치(Floating Shelves, D-012)를 1차 대상으로 하고, 다른 상위 니치와의
유병률 비교도 제공한다.

두 트랙으로 교차검증한다(어느 한쪽도 단독으로 신뢰하지 않는다):
  (1) aspect 사전  — 셸프 불만의 알려진 실패 모드를 시드 구문으로 *조작적 정의*하고
                     니치별 유병률(불만 리뷰 중 언급 비율)을 센다. 규칙·투명·감사가능.
  (2) NMF 토픽    — TF-IDF + NMF로 비지도 토픽을 발굴해, 사전이 놓친 신호가 있는지 점검.

왜 임베딩/LLM이 아니라 사전+NMF인가 → decisions.md D-014.
요지: 실패 모드 어휘가 좁고 영어 단일이라, 투명·재현·감사 가능한 규칙/선형대수 기법이
이 과업에선 LLM보다 방어력이 높다. LLM은 "차별점"이 아니라(설계서 R9), 검증 설계가 차별점.

검증(LLM 출력 맹신 금지의 규칙판): aspect 라벨링 표본을 무작위 추출해 수동 검수용으로
별도 저장한다(`docs/q2_validation_sample.md`).

출력:
  data/marts/q2_aspect_prevalence[_sample].parquet   니치 × aspect 유병률
  data/marts/q2_topics[_sample].parquet              NMF 토픽(상위 용어·크기)
  docs/q2_unmet_needs[_sample].md                    미충족 니즈 리포트(자동 생성)
  docs/q2_validation_sample[_sample].md              수동 검수용 표본

사용:
    python src/analyze/unmet_needs.py            # 전체 데이터
    python src/analyze/unmet_needs.py --sample   # 스모크 샘플
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import NMF  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from src.ingest.common import PROJECT_ROOT, load_config, write_run_log  # noqa: E402


def compile_aspects(aspects: dict) -> dict:
    """aspect → 컴파일된 정규식(시드 구문 OR, 단어경계). 부분문자열 오탐 방지."""
    compiled = {}
    for name, phrases in aspects.items():
        alt = "|".join(re.escape(p) for p in phrases)
        compiled[name] = re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)
    return compiled


def load_low_reviews(con, fct_path: Path, cfg_q2: dict, niches: list[str]) -> pd.DataFrame:
    """저평점·텍스트 보유 리뷰를 대상 니치로 한정해 로드한다."""
    niche_list = ", ".join(f"'{n}'" for n in niches)
    df = con.execute(f"""
        SELECT niche, parent_asin, review_title, text, rating, helpful_vote
        FROM '{fct_path}'
        WHERE rating <= {cfg_q2['low_rating_max']}
          AND text IS NOT NULL
          AND length(text) >= {cfg_q2['min_text_len']}
          AND niche IN ({niche_list})
    """).df()
    # HTML 잔재(<br /> 등) 제거 — NMF에 'br' 같은 유사토큰이 끼는 것 방지
    for c in ("text", "review_title"):
        df[c] = df[c].fillna("").str.replace(r"<[^>]+>", " ", regex=True)
    return df


def label_aspects(df: pd.DataFrame, compiled: dict) -> pd.DataFrame:
    """각 리뷰에 aspect 플래그(0/1)를 붙인다. 제목+본문을 함께 검사."""
    blob = (df["review_title"].fillna("") + ". " + df["text"].fillna("")).str.lower()
    for name, rx in compiled.items():
        df[f"a_{name}"] = blob.str.contains(rx).astype(int)
    return df


def aspect_prevalence(df: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    """니치 × aspect 유병률(저평점 리뷰 중 언급 비율) + 니치별 저평점 리뷰 수."""
    rows = []
    for niche, g in df.groupby("niche"):
        row = {"niche": niche, "n_low_reviews": len(g)}
        for a in aspects:
            row[a] = g[f"a_{a}"].mean()
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_low_reviews", ascending=False)
    return out.reset_index(drop=True)


def _wilson(k: int, n: int) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간. n=0이면 (0,0)."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_focus_all(con, fct_path: Path, cfg_q2: dict, focus: str) -> pd.DataFrame:
    """focus 니치의 *전 평점* 리뷰를 로드(lift·심각도·추세용). 날짜 포함."""
    df = con.execute(f"""
        SELECT review_title, text, rating, review_date
        FROM '{fct_path}'
        WHERE niche = '{focus}' AND text IS NOT NULL
          AND length(text) >= {cfg_q2['min_text_len']}
    """).df()
    for c in ("text", "review_title"):
        df[c] = df[c].fillna("").str.replace(r"<[^>]+>", " ", regex=True)
    return df


def severity_lift(df_all: pd.DataFrame, aspects: list[str], low_max: int,
                  high_min: int) -> pd.DataFrame:
    """aspect별 저평점/고평점 유병률·lift·심각도(언급 리뷰 평균 별점)·Wilson CI."""
    low = df_all[df_all["rating"] <= low_max]
    high = df_all[df_all["rating"] >= high_min]
    rows = []
    for a in aspects:
        col = f"a_{a}"
        k_low, n_low = int(low[col].sum()), len(low)
        prev_low = k_low / n_low if n_low else 0.0
        prev_high = high[col].mean() if len(high) else 0.0
        lift = prev_low / prev_high if prev_high > 0 else np.nan
        sev = df_all.loc[df_all[col] == 1, "rating"].mean() if df_all[col].any() else np.nan
        lo, hi = _wilson(k_low, n_low)
        rows.append({"aspect": a, "prev_low": prev_low, "prev_high": prev_high,
                     "lift": lift, "severity_avg_star": sev,
                     "prev_low_ci_lo": lo, "prev_low_ci_hi": hi,
                     "n_low": n_low, "n_high": len(high)})
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def trend_by_year(df_all: pd.DataFrame, aspects: list[str], low_max: int,
                  min_year_n: int = 50) -> pd.DataFrame:
    """저평점 리뷰의 연도별 aspect 유병률(추세). 표본 적은 연도는 제외."""
    low = df_all[df_all["rating"] <= low_max].copy()
    low["year"] = pd.to_datetime(low["review_date"]).dt.year
    rows = []
    for y, g in low.groupby("year"):
        if len(g) < min_year_n:
            continue
        row = {"year": int(y), "n_low": len(g)}
        for a in aspects:
            row[a] = float(g[f"a_{a}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def representative_examples(g: pd.DataFrame, aspect: str, k: int) -> list[str]:
    """aspect를 가진 리뷰 중 helpful_vote 높은 순 대표 예시 k개(중복 텍스트 제거)."""
    hit = g[g[f"a_{aspect}"] == 1].copy()
    if hit.empty:
        return []
    hit["_key"] = hit["text"].str.slice(0, 80).str.lower()
    hit = hit.sort_values("helpful_vote", ascending=False).drop_duplicates("_key")
    out = []
    for r in hit.head(k).itertuples():
        title = (r.review_title or "").strip()
        txt = " ".join(str(r.text).split())[:200]
        out.append(f"«{title}» {txt}")
    return out


def nmf_topics(texts: list[str], n_topics: int, max_features: int,
               seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """TF-IDF + NMF로 비지도 토픽을 뽑는다. 토픽별 상위 용어와 크기 반환."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=5, max_df=0.5, max_features=max_features)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    model = NMF(n_components=n_topics, init="nndsvda", random_state=seed,
                max_iter=400)
    # numpy 2.x matmul이 macOS Accelerate에서 유한 입력에도 허위 FPE 경고를 냄(D-012).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        W = model.fit_transform(X)    # (n_docs, n_topics)
    H = model.components_             # (n_topics, n_terms)
    assign = W.argmax(axis=1)         # 각 문서의 대표 토픽
    rows = []
    for t in range(n_topics):
        top_terms = vocab[np.argsort(H[t])[::-1][:8]]
        rows.append({
            "topic": t,
            "size": int((assign == t).sum()),
            "top_terms": ", ".join(top_terms),
        })
    topics = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return topics, assign


def write_report(focus: str, prev: pd.DataFrame, focus_examples: dict,
                 topics: pd.DataFrame, cfg_q2: dict, n_low_focus: int,
                 out_path: Path, sev_df: pd.DataFrame | None = None,
                 trend_df: pd.DataFrame | None = None):
    aspects = list(cfg_q2["aspects"].keys())
    label_kr = {
        "mounting_hardware": "벽 고정·하드웨어",
        "sturdiness_sagging": "처짐·흔들림·하중",
        "build_quality": "재질·내구성",
        "finish_cosmetic": "마감·외관",
        "size_fit": "크기·치수 기대 불일치",
        "instructions": "설명서·조립",
        "shipping_damage": "배송 파손",
    }
    fr = prev[prev["niche"] == focus].iloc[0]
    ranked = sorted(aspects, key=lambda a: fr[a], reverse=True)

    L = [
        "# Q2 — 미충족 니즈 텍스트 마이닝",
        "",
        "> 자동 생성: `src/analyze/unmet_needs.py` · 대상: 저평점"
        f"(≤{cfg_q2['low_rating_max']}★) 리뷰 · 1차 니치: **{focus}**"
        f"(저평점 리뷰 {n_low_focus:,}건)",
        "",
        "**방법(교차검증)**: ① aspect 사전 = 셸프 불만의 *조작적 정의*(규칙·투명) → 유병률 측정, "
        "② NMF 비지도 토픽 = 사전이 놓친 신호 점검. "
        "임베딩/LLM을 쓰지 않은 이유는 decisions.md D-014.",
        "",
        f"## {focus} — 미충족 니즈 순위 (저평점 리뷰 중 언급 비율)",
        "",
        "| 순위 | 불만 aspect | 유병률 | 차별화 진입점(So what) |",
        "|---:|---|---:|---|",
    ]
    sowhat = {
        "mounting_hardware": "동봉 앵커/나사를 드라이월·스터드용 고급 하드웨어로 업그레이드, 설치 영상 제공",
        "sturdiness_sagging": "하중 스펙 명시 + 히든 브래킷 보강, '처지지 않는' 메시지로 차별화",
        "build_quality": "원목/후판 등 소재 등급 상향, 두께·재질 사진 전면 배치",
        "finish_cosmetic": "마감 QC 강화, 색상·결 실물 사진, 휨 방지 포장",
        "size_fit": "치수 다이어그램·실측 사진으로 기대 일치, 사이즈 옵션 확대",
        "instructions": "단계별 그림 설명서 + QR 설치영상, 부속 체크리스트 동봉",
        "shipping_damage": "모서리 보강 포장으로 전송 중 파손 방지",
    }
    for i, a in enumerate(ranked, 1):
        L.append(f"| {i} | {label_kr.get(a, a)} | {fr[a]:.0%} | {sowhat.get(a, '')} |")

    L += ["", f"### 상위 불만 대표 리뷰 (helpful_vote 순)", ""]
    for a in ranked[:4]:
        L.append(f"**{label_kr.get(a, a)}** ({fr[a]:.0%})")
        for ex in focus_examples.get(a, []):
            L.append(f"- {ex}")
        L.append("")

    L += [
        f"## 니치 간 비교 — aspect 유병률",
        "",
        "동일 불만이 니치마다 얼마나 다른지. 특정 불만이 유독 높은 니치 = 그 축으로 진입 시 차별화 여지.",
        "",
        "| 니치 | 저평점N | " + " | ".join(label_kr.get(a, a) for a in aspects) + " |",
        "|---|---:|" + "---:|" * len(aspects),
    ]
    for r in prev.itertuples():
        cells = " | ".join(f"{getattr(r, a):.0%}" for a in aspects)
        L.append(f"| {r.niche} | {r.n_low_reviews:,} | {cells} |")
    L += ["", "![니치별 불만 유병률 히트맵](figures/q2_aspect_heatmap.png)"]

    if sev_df is not None and len(sev_df):
        L += [
            "",
            f"## 이 불만이 *진짜 동인*인가 — lift·심각도 ({focus}, 심화)",
            "",
            "유병률만으로는 '저평점에 흔한 단어'와 '저평점을 *만드는* 불만'을 못 가른다. "
            "**lift = 저평점 유병률 ÷ 고평점(≥4★) 유병률**: lift≫1이면 그 aspect가 불만 리뷰에 "
            "특이적(진짜 차별화 동인), lift≈1이면 모든 리뷰에 흔한 배경어. "
            "**심각도 = 그 aspect를 언급한 리뷰의 평균 별점**(낮을수록 치명적).",
            "",
            "| 불만 aspect | 저평점 유병률 (95% CI) | 고평점 유병률 | lift | 심각도(평균★) |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in sev_df.itertuples():
            lift = "—" if pd.isna(r.lift) else f"{r.lift:.1f}×"
            sev = "—" if pd.isna(r.severity_avg_star) else f"{r.severity_avg_star:.2f}"
            L.append(f"| {label_kr.get(r.aspect, r.aspect)} | "
                     f"{r.prev_low:.0%} ({r.prev_low_ci_lo:.0%}~{r.prev_low_ci_hi:.0%}) | "
                     f"{r.prev_high:.0%} | {lift} | {sev} |")
        top = sev_df.dropna(subset=["lift"]).iloc[0] if sev_df["lift"].notna().any() else None
        if top is not None:
            L += ["", f"> **읽는 법**: lift가 가장 높은 **{label_kr.get(top.aspect, top.aspect)}"
                  f"**(lift {top.lift:.1f}×)는 고평점에선 드물고 저평점에 몰린 = 불만의 진짜 동인. "
                  "유병률이 높아도 lift≈1인 aspect는 차별화 포인트로 약하다."]
        L += ["", "![불만 우선순위(유병률×lift)](figures/q2_priority.png)"]

    if trend_df is not None and len(trend_df) >= 2:
        L += [
            "",
            f"## 불만 추세 — 연도별 유병률 ({focus}, 저평점 리뷰 중)",
            "",
            "오르는 불만 = 시장이 아직 못 푼 미충족 니즈(진입 시 선점 여지). 표본 부족 연도는 제외.",
            "",
            "| 연도 | 저평점N | " + " | ".join(label_kr.get(a, a) for a in aspects) + " |",
            "|---:|---:|" + "---:|" * len(aspects),
        ]
        for r in trend_df.itertuples():
            cells = " | ".join(f"{getattr(r, a):.0%}" for a in aspects)
            L.append(f"| {int(r.year)} | {int(r.n_low):,} | {cells} |")
        L += ["", "![불만 추세(연도별)](figures/q2_trend.png)"]

    L += [
        "",
        "## NMF 비지도 토픽 (사전 교차검증)",
        "",
        f"`{focus}` 저평점 리뷰를 TF-IDF→NMF로 분해. 사전 aspect와 겹치면 신뢰도 ↑, "
        "새 용어가 보이면 사전 보강 후보.",
        "",
        "| 토픽 | 문서수 | 상위 용어 |",
        "|---:|---:|---|",
    ]
    for r in topics.itertuples():
        L.append(f"| {r.topic} | {r.size:,} | {r.top_terms} |")

    L += [
        "",
        "## 한계와 검증",
        "",
        f"- aspect 사전은 단어경계 규칙이라 풍자/부정문 오탐 가능 → 표본 {cfg_q2['validation_sample']}건을 "
        "`q2_validation_sample.md`로 분리해 수동 검수(정합성 확인).",
        "- 저평점 리뷰는 불만에 편향된 표본 → '미충족 니즈의 존재'는 보이나 '전체 고객 중 비율'은 아님.",
        "- 리뷰≠판매(L-1), 키워드 니치 정의 잔여 오탐(L-6)은 상위 문서 참조.",
    ]
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_validation_sample(df: pd.DataFrame, aspects: list[str], n: int,
                            seed: int, out_path: Path):
    """aspect별로 고르게 표본을 뽑아 수동 검수용 md를 만든다."""
    rng = np.random.default_rng(seed)
    picks = []
    per = max(1, n // len(aspects))
    for a in aspects:
        hit = df[df[f"a_{a}"] == 1]
        if len(hit):
            idx = rng.choice(hit.index, size=min(per, len(hit)), replace=False)
            for i in idx:
                picks.append((a, df.loc[i]))
    L = ["# Q2 검수 표본 — aspect 라벨 정합성 수동 확인",
         "",
         "> 각 줄의 [aspect]가 리뷰 내용과 맞는지 직접 확인(오탐이면 사전 수정 근거). "
         "LLM/규칙 출력 맹신 금지의 규칙판.",
         ""]
    for a, r in picks:
        txt = " ".join(str(r["text"]).split())[:180]
        L.append(f"- **[{a}]** ({r['niche']}, {int(r['rating'])}★) «{r['review_title']}» {txt}")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    cfg = load_config()
    q2 = cfg["q2_unmet_needs"]
    seed = cfg["runtime"]["seed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="_sample 마트 사용")
    args = ap.parse_args()

    marts = PROJECT_ROOT / cfg["paths"]["marts"]
    sfx = "_sample" if args.sample else ""
    fct = marts / f"fct_review{sfx}.parquet"
    if not fct.exists():
        sys.exit(f"입력 없음: {fct} — transform을 먼저 실행하세요.")

    con = duckdb.connect()
    focus = q2["focus_niche"]
    # 비교 대상 니치 = 저평점 리뷰가 많은 상위 N (focus 포함 보장)
    top_niches = con.execute(f"""
        SELECT niche FROM '{fct}'
        WHERE rating <= {q2['low_rating_max']} AND text IS NOT NULL
          AND niche != 'OTHER_SMALL'
        GROUP BY niche ORDER BY count(*) DESC LIMIT {q2['compare_niches']}
    """).df()["niche"].tolist()
    niches = list(dict.fromkeys([focus] + top_niches))

    df = load_low_reviews(con, fct, q2, niches)
    if df.empty or focus not in set(df["niche"]):
        sys.exit(f"대상 저평점 리뷰 없음(focus={focus}) — Q2를 검증할 수 없습니다.")

    compiled = compile_aspects(q2["aspects"])
    aspects = list(q2["aspects"].keys())
    df = label_aspects(df, compiled)

    prev = aspect_prevalence(df, aspects)

    focus_df = df[df["niche"] == focus]
    n_low_focus = len(focus_df)
    focus_examples = {a: representative_examples(focus_df, a, q2["examples_per_aspect"])
                      for a in aspects}

    # 심화(D-019): focus 니치 전 평점을 로드해 lift·심각도·연도 추세 계산
    df_all = label_aspects(load_focus_all(con, fct, q2, focus), compiled)
    sev_df = severity_lift(df_all, aspects, q2["low_rating_max"], q2["high_rating_min"])
    trend_df = trend_by_year(df_all, aspects, q2["low_rating_max"])
    sev_df.to_parquet(marts / f"q2_aspect_severity{sfx}.parquet", index=False)
    trend_df.to_parquet(marts / f"q2_aspect_trend{sfx}.parquet", index=False)

    # NMF 토픽 (focus 니치). 문서가 너무 적으면 토픽 수를 줄인다.
    texts = focus_df["text"].fillna("").tolist()
    n_topics = min(q2["nmf_topics"], max(2, len(texts) // 5))
    try:
        topics, _ = nmf_topics(texts, n_topics, q2["nmf_max_features"], seed)
    except ValueError:  # 어휘가 너무 적은 샘플 데이터
        topics = pd.DataFrame([{"topic": 0, "size": len(texts),
                                "top_terms": "(데이터 부족 — 전체 수집 후 의미 있음)"}])

    # 출력
    prev.to_parquet(marts / f"q2_aspect_prevalence{sfx}.parquet", index=False)
    topics.to_parquet(marts / f"q2_topics{sfx}.parquet", index=False)
    docs = PROJECT_ROOT / "docs"
    write_report(focus, prev, focus_examples, topics, q2, n_low_focus,
                 docs / f"q2_unmet_needs{sfx}.md", sev_df, trend_df)
    write_validation_sample(focus_df, aspects, q2["validation_sample"], seed,
                            docs / f"q2_validation_sample{sfx}.md")

    fr = prev[prev["niche"] == focus].iloc[0]
    top_aspect = max(aspects, key=lambda a: fr[a])
    stats = {
        "sample": args.sample,
        "focus_niche": focus,
        "n_low_reviews_focus": int(n_low_focus),
        "n_niches_compared": len(prev),
        "top_aspect": top_aspect,
        "top_aspect_prevalence": round(float(fr[top_aspect]), 4),
        "nmf_topics": int(len(topics)),
    }
    write_run_log(f"q2_unmet_needs{sfx}", stats)
    print(f"done: {stats}")
    print(f"  report → {docs / f'q2_unmet_needs{sfx}.md'}")


if __name__ == "__main__":
    main()
