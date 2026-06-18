-- ─────────────────────────────────────────────────────────────
-- Stage 2-1: staging — 타입 정리, 중복 제거, 표준화
-- 실행: run_transform.py가 {{meta_glob}}, {{slim_glob}}, {{full_glob}} 치환
-- 원칙: raw는 절대 수정하지 않는다. staging은 1소스 = 1테이블.
-- ─────────────────────────────────────────────────────────────

-- 상품 메타 (메인 + 교차등록 카테고리 통합, parent_asin 단위 dedup)
CREATE OR REPLACE TABLE stg_meta AS
WITH meta_dedup AS (
    SELECT
        parent_asin,
        first(title ORDER BY coalesce(rating_number, 0) DESC, title ASC) AS title,
        first(main_category ORDER BY coalesce(rating_number, 0) DESC, title ASC) AS main_category,
        first(categories ORDER BY coalesce(rating_number, 0) DESC, title ASC) AS categories,
        max(price)                    AS price,          -- 교차등록 시 보수적으로 max
        max(average_rating)           AS average_rating,
        max(rating_number)            AS rating_number,
        first(store ORDER BY coalesce(rating_number, 0) DESC, title ASC) AS store
    FROM read_parquet([{{meta_glob}}], union_by_name=true)
    WHERE parent_asin IS NOT NULL
    GROUP BY parent_asin
)
SELECT
    parent_asin,
    title,
    main_category,
    categories,
    -- categories 'A > B > C' 문자열에서 leaf 추출. 빈 categories(약 8%)는
    -- 빈 문자열 leaf가 되어 가짜 니치를 만드므로 NULL로 정규화한다.
    nullif(trim(regexp_extract(categories, '([^>]+)$', 1)), '') AS category_leaf,
    price,
    average_rating,
    rating_number,
    store
FROM meta_dedup;

-- 전체 카테고리 리뷰 (slim, 텍스트 없음) — source key를 보존해 정상 same-day 리뷰 손실 방지
CREATE OR REPLACE TABLE stg_review_slim AS
SELECT DISTINCT
    parent_asin,
    asin,
    {{slim_timestamp_ms}}          AS source_timestamp_ms,
    {{slim_user_id}}               AS source_user_id,
    CAST(date AS DATE)            AS review_date,
    date_trunc('month', CAST(date AS DATE)) AS review_month,
    CAST(rating AS DOUBLE)        AS rating,
    verified_purchase
FROM read_parquet([{{slim_glob}}], union_by_name=true)
WHERE parent_asin IS NOT NULL
  AND rating BETWEEN 1 AND 5;

-- 타깃 니치 리뷰 (full, 텍스트 포함)
CREATE OR REPLACE TABLE stg_review_full AS
SELECT
    parent_asin,
    asin,
    CAST(date AS DATE)            AS review_date,
    CAST(rating AS DOUBLE)        AS rating,
    verified_purchase,
    review_title,
    text,
    CAST(helpful_vote AS INTEGER) AS helpful_vote,
    user_id,
    -- 동일 유저가 동일 상품에 남긴 중복 리뷰 제거(최신 1건)
    row_number() OVER (
        PARTITION BY parent_asin, user_id, text
        ORDER BY CAST(date AS DATE) DESC
    ) AS _rn
FROM read_parquet([{{full_glob}}])
WHERE parent_asin IS NOT NULL
  AND rating BETWEEN 1 AND 5;

DELETE FROM stg_review_full WHERE _rn > 1;
ALTER TABLE stg_review_full DROP COLUMN _rn;

-- 니치 universe (검수 플래그 포함 그대로 staging에 등록)
CREATE OR REPLACE TABLE stg_universe AS
SELECT parent_asin, title, is_hexagon, matched_title, matched_category
FROM read_parquet('{{universe_path}}');
