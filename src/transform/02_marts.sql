-- ─────────────────────────────────────────────────────────────
-- Stage 2-2: marts — 분석 단위 테이블
-- 핵심 규율: mart_launch의 모든 피처는 point-in-time (출시 시점에 알 수
-- 있었던 정보만). 미래 정보 누수 금지. (설계서 R4)
-- ─────────────────────────────────────────────────────────────

-- [dim_product] 상품 차원 테이블: 메타 + 첫/마지막 리뷰일(출시일 프록시)
CREATE OR REPLACE TABLE dim_product AS
SELECT
    m.parent_asin,
    m.title,
    m.main_category,
    m.category_leaf,
    -- 니치: 상품 수 30개 미만 leaf는 'OTHER_SMALL'로 합침 (설계서 3.3).
    -- leaf가 NULL(빈 categories)인 상품도 OTHER_SMALL로 보낸다 — 가짜 '' 니치 방지.
    CASE WHEN m.category_leaf IS NOT NULL
              AND count(*) OVER (PARTITION BY m.category_leaf) >= 30
         THEN m.category_leaf ELSE 'OTHER_SMALL' END AS niche,
    m.price,
    m.average_rating,
    m.rating_number,
    m.store,
    r.first_review_date,                       -- 출시일 프록시 (한계 L-3)
    r.last_review_date,
    r.n_reviews_in_period,
    u.parent_asin IS NOT NULL                  AS in_target_niche,
    coalesce(u.is_hexagon, FALSE)              AS is_hexagon
FROM stg_meta m
LEFT JOIN (
    SELECT parent_asin,
           min(review_date) AS first_review_date,
           max(review_date) AS last_review_date,
           count(*)         AS n_reviews_in_period
    FROM stg_review_slim GROUP BY 1
) r USING (parent_asin)
LEFT JOIN stg_universe u USING (parent_asin);

-- [fct_review] 리뷰 사실 테이블 (타깃 니치, 텍스트 분석용)
-- in_target_niche로 거른다: 풀텍스트 리뷰는 수집 당시 universe ASIN으로 받았는데,
-- universe 오탐 정리(D-013) 후 비-타깃이 된 ASIN(브래킷 등)의 리뷰는 Q2 모집단에서 제외.
CREATE OR REPLACE TABLE fct_review AS
SELECT f.*, d.niche, d.price, d.is_hexagon
FROM stg_review_full f
JOIN dim_product d USING (parent_asin)
WHERE d.in_target_niche;

-- [mart_niche_monthly] 니치 × 월 패널 → Q1 니치 스코어링의 원천
CREATE OR REPLACE TABLE mart_niche_monthly AS
WITH product_launch AS (
    SELECT parent_asin, niche, first_review_date,
           date_trunc('month', first_review_date) AS launch_month
    FROM dim_product WHERE first_review_date IS NOT NULL
)
SELECT
    d.niche,
    s.review_month,
    count(*)                                         AS n_reviews,
    count(DISTINCT s.parent_asin)                    AS n_active_products,
    avg(s.rating)                                    AS avg_rating,
    avg(CASE WHEN s.rating <= 2 THEN 1 ELSE 0 END)   AS low_rating_share,
    avg(CASE WHEN s.verified_purchase THEN 1 ELSE 0 END) AS verified_share,
    -- 신규 진입 상품 수 (해당 월에 첫 리뷰 발생)
    (SELECT count(*) FROM product_launch p
      WHERE p.niche = d.niche AND p.launch_month = s.review_month) AS n_new_products,
    -- 경쟁 집중도: 상위 5개 상품의 월 리뷰 점유율
    (
      SELECT sum(c) FROM (
        SELECT count(*) AS c FROM stg_review_slim s2
        JOIN dim_product d2 USING (parent_asin)
        WHERE d2.niche = d.niche AND s2.review_month = s.review_month
        GROUP BY s2.parent_asin ORDER BY c DESC LIMIT 5
      )
    )::DOUBLE / nullif(count(*), 0)                  AS top5_review_share
FROM stg_review_slim s
JOIN dim_product d USING (parent_asin)
WHERE d.niche != 'OTHER_SMALL'
GROUP BY d.niche, s.review_month;

-- [mart_launch] 출시 코호트 → Q3 안착 예측의 원천
-- 타깃: 첫 리뷰 후 12개월 내 누적 리뷰 수 / 평균 평점 (라벨은 분석 단계에서 정의)
-- 피처: 첫 90일 내 정보 + 출시 "시점까지의" 니치 상태 (point-in-time)
CREATE OR REPLACE TABLE mart_launch AS
WITH launch AS (
    SELECT parent_asin, niche, price, first_review_date
    FROM dim_product
    WHERE first_review_date IS NOT NULL
      AND niche != 'OTHER_SMALL'
      -- 12개월 관측 가능한 코호트만 (우측 절단 방지)
      AND first_review_date <= (SELECT max(review_date) FROM stg_review_slim) - INTERVAL 365 DAY
),
early AS (  -- 피처: 첫 90일
    SELECT l.parent_asin,
           count(*)    AS reviews_90d,
           avg(s.rating) AS avg_rating_90d,
           avg(CASE WHEN s.rating <= 2 THEN 1 ELSE 0 END) AS low_share_90d
    FROM launch l JOIN stg_review_slim s USING (parent_asin)
    WHERE s.review_date < l.first_review_date + INTERVAL 90 DAY
    GROUP BY 1
),
outcome AS (  -- 결과: 12개월
    SELECT l.parent_asin,
           count(*)      AS reviews_12m,
           avg(s.rating) AS avg_rating_12m
    FROM launch l JOIN stg_review_slim s USING (parent_asin)
    WHERE s.review_date < l.first_review_date + INTERVAL 365 DAY
    GROUP BY 1
),
-- 피처: 출시 "시점까지의" 니치 상태 (누수 방지 핵심).
-- 주의: (출시상품 × 같은 니치 모든 상품 × 그들의 모든 리뷰) 3중 조인은 큰 니치에서
-- 수십억 행으로 폭발(OOM). 동일 의미를 ASOF 조인(시점 누적값 조회)으로 O(N log N)에 계산.
-- 동치성: "출시 D 이전 리뷰가 1건이라도 있는 상품 수" = "첫 리뷰일 < D 인 상품 수"
-- (상품의 첫 리뷰일=가장 이른 리뷰이므로). decisions.md D-011.
niche_review_cum AS (  -- 니치×날짜: 그 날짜까지의 누적 리뷰 수/평점합
    SELECT niche, review_date,
           sum(cnt)        OVER w AS cum_reviews,
           sum(sum_rating) OVER w AS cum_sum_rating
    FROM (
        SELECT d.niche, s.review_date,
               count(*) AS cnt, sum(s.rating) AS sum_rating
        FROM stg_review_slim s JOIN dim_product d USING (parent_asin)
        WHERE d.niche != 'OTHER_SMALL'
        GROUP BY 1, 2
    )
    WINDOW w AS (PARTITION BY niche ORDER BY review_date)
),
niche_prod_cum AS (   -- 니치×날짜: 그 날짜까지 출시된(첫 리뷰) 누적 상품 수
    SELECT niche, first_review_date AS dt,
           sum(new_products) OVER (PARTITION BY niche ORDER BY first_review_date) AS cum_products
    FROM (
        SELECT niche, first_review_date, count(*) AS new_products
        FROM dim_product
        WHERE first_review_date IS NOT NULL AND niche != 'OTHER_SMALL'
        GROUP BY 1, 2
    )
),
niche_at_launch AS (
    SELECT l.parent_asin,
           pc.cum_products                              AS niche_products_before,
           rc.cum_reviews                               AS niche_reviews_before,
           rc.cum_sum_rating / nullif(rc.cum_reviews, 0) AS niche_avg_rating_before
    FROM launch l
    ASOF LEFT JOIN niche_review_cum rc
      ON l.niche = rc.niche AND rc.review_date < l.first_review_date
    ASOF LEFT JOIN niche_prod_cum pc
      ON l.niche = pc.niche AND pc.dt < l.first_review_date
)
SELECT l.*,
       year(l.first_review_date)         AS launch_year,   -- 시간 split 키
       e.reviews_90d, e.avg_rating_90d, e.low_share_90d,
       n.niche_products_before, n.niche_reviews_before, n.niche_avg_rating_before,
       o.reviews_12m, o.avg_rating_12m
FROM launch l
LEFT JOIN early   e USING (parent_asin)
LEFT JOIN niche_at_launch n USING (parent_asin)
LEFT JOIN outcome o USING (parent_asin);
