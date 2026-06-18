# 데이터 사전 (Data Dictionary)

> 모든 마트·지표·라벨·피처의 정의를 한곳에 명문화한다. 코드(`config.yaml`, `src/`)가
> 진실의 소스이며, 이 문서는 그 정의를 사람이 읽는 형태로 요약한다.

## 핵심 개념 정의

| 용어 | 정의 | 근거 |
|---|---|---|
| **니치(niche)** | 메타데이터 category 트리의 leaf 노드. 상품 30개 미만 leaf와 빈 categories는 `OTHER_SMALL`로 병합 | D-008, 설계서 3.3 |
| **상품 단위** | `parent_asin`(변형 묶음). 색상/사이즈 변형은 리뷰를 공유 | D-004 |
| **출시일(launch)** | 메타에 출시일 없음 → **첫 리뷰 날짜**로 프록시 | L-3 |
| **수요 프록시** | 월별 신규 리뷰 수. 리뷰≠판매이므로 *증가율·상대비교*로만 해석 | L-1 |
| **타깃 니치(universe)** | wall/floating shelf 키워드+카테고리 매칭 ASIN. "shelf bracket" 제목단독 제외 | D-006, D-013 |
| **hexagon 상품** | 셀러 자사 품목(회고 검증 앵커) | D-012 |

## 마트 테이블 (`data/marts/`)

| 테이블 | 입자 | 용도 | 핵심 컬럼 |
|---|---|---|---|
| `dim_product` | parent_asin | 상품 차원 | niche, price, average_rating, first_review_date, in_target_niche, is_hexagon |
| `fct_review` | 리뷰 | Q2 텍스트 마이닝(타깃 니치) | rating, review_title, text, helpful_vote, niche |
| `mart_niche_monthly` | 니치×월 | Q1 스코어링 | n_reviews, n_active_products, low_rating_share, n_new_products, top5_review_share |
| `mart_launch` | 출시 코호트 | Q3 안착 예측 | reviews_90d, avg_rating_90d, niche_*_before, reviews_12m, launch_year |

## Q1 — 니치 기회 스코어 (지표는 모두 "높을수록 매력적"으로 부호 정렬)

| 지표 | 정의 | 방향 |
|---|---|---|
| `demand_growth` | 최근 12개월 / 직전 12개월 리뷰 증가율 | 높을수록↑ |
| `market_openness` | 1 − 상위5 상품 리뷰 점유율 | 집중도 낮을수록↑ |
| `quality_gap` | 저평점(≤2★) 비율 | 불만 많을수록 진입기회↑ |
| `uncrowded` | 1 − 신규진입강도(신규/활성) | 포화 낮을수록↑ |
| `opportunity_score` | 위 4개 z-표준화 가중합(가중치 config화) | — |
| `prob_top_n` | 가중치 Dirichlet 섭동 시 상위 N 진입 빈도(랭크 안정성) | — |

## Q2 — 미충족 니즈 (저평점 ≤2★ 리뷰)

| 산출 | 정의 |
|---|---|
| **aspect** | 셸프 불만 실패 모드 7종. 각 aspect는 시드 구문 집합으로 *조작적 정의*(config `q2_unmet_needs.aspects`), 단어경계 substring 매칭 |
| **유병률** | 니치 내 저평점 리뷰 중 해당 aspect를 언급한 비율 |
| **NMF 토픽** | TF-IDF(1-2gram)→NMF 비지도 토픽(사전 교차검증용) |

aspect 7종: mounting_hardware(벽고정·하드웨어), sturdiness_sagging(처짐·하중),
build_quality(재질·내구성), finish_cosmetic(마감), size_fit(치수), instructions(설명서),
shipping_damage(배송 파손 — bare "missing" 제외, D-014/5일차).

## Q3 — 안착 예측 (단위: 출시 코호트)

**라벨 `settled`** (조작적 정의, D-015·D-016):
- `later_reviews = reviews_12m − reviews_90d` (출시 90~365일 트랙션)
- `settled = 1` ⟺ `later_reviews ≥ max(niche_P75_train, 3)`
- 니치 P75는 **train 연도만으로** 산정(누수 방지). 평점 게이트는 피처 누수(corr 0.92)로 제거.

**피처** (모두 point-in-time — 출시 시점/첫 90일):

| 피처 | 정의 |
|---|---|
| `price` | 가격(결측 65% → `price_missing` 지시자 동반, LightGBM native 결측) |
| `reviews_90d` | 첫 90일 리뷰 수(log1p) |
| `avg_rating_90d` | 첫 90일 평균 평점 |
| `low_share_90d` | 첫 90일 저평점 비율 |
| `niche_products_before` | 출시 시점 니치 누적 상품 수(log1p) |
| `niche_reviews_before` | 출시 시점 니치 누적 리뷰 수(log1p) |
| `niche_avg_rating_before` | 출시 시점 니치 평균 평점 |

`launch_year`는 피처가 아니라 **시간 split 키**(train ≤2020 / valid 2021 / test 2022).

**평가**: PR-AUC(1차, 불균형), ROC-AUC, MCC. 임계값은 valid에서 MCC 최대화로 선택.
Brier는 불균형 가중으로 미보정이라 참고치(D-016).
