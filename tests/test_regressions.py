import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.analyze.settlement_model import validate_binary_split
from src.analyze.unmet_needs import compile_aspects
from src.ingest.common import ShardSink


class AspectRegexTest(unittest.TestCase):
    def test_aspect_phrases_use_both_word_boundaries(self):
        rx = compile_aspects({"mounting": ["mount"]})["mounting"]

        self.assertIsNotNone(rx.search("hard to mount on drywall"))
        self.assertIsNone(rx.search("this mountain shelf looks odd"))


class SettlementSplitValidationTest(unittest.TestCase):
    def test_one_class_validation_split_fails(self):
        X = pd.DataFrame({"reviews_90d": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y = np.zeros(len(X), dtype=int)

        with self.assertRaises(SystemExit):
            validate_binary_split("valid", X, y, 10)


class ShardSinkResumeTest(unittest.TestCase):
    def test_resume_removes_uncommitted_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "reviews.parquet"
            stale = Path(tmp) / "reviews__part00001.parquet"
            stale.write_bytes(b"partial")

            sink = ShardSink(canonical)
            sink.resume_from(1)

            self.assertFalse(stale.exists())


class TransformDedupTest(unittest.TestCase):
    def test_dedup_keeps_same_day_reviews_with_distinct_source_identity(self):
        con = duckdb.connect()
        con.execute(
            """
            CREATE TABLE raw_reviews (
                parent_asin VARCHAR,
                asin VARCHAR,
                timestamp_ms BIGINT,
                review_date DATE,
                rating DOUBLE,
                verified_purchase BOOLEAN,
                user_id VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO raw_reviews VALUES
                ('P1', 'A1', 1000, DATE '2020-01-01', 5.0, true, 'u1'),
                ('P1', 'A1', 2000, DATE '2020-01-01', 5.0, true, 'u2'),
                ('P1', 'A1', 2000, DATE '2020-01-01', 5.0, true, 'u2')
            """
        )

        n_rows = con.execute(
            """
            WITH stg_review_slim AS (
                SELECT DISTINCT
                    parent_asin,
                    asin,
                    timestamp_ms AS source_timestamp_ms,
                    user_id AS source_user_id,
                    review_date,
                    date_trunc('month', review_date) AS review_month,
                    rating,
                    verified_purchase
                FROM raw_reviews
                WHERE parent_asin IS NOT NULL
                  AND review_date IS NOT NULL
                  AND rating BETWEEN 1 AND 5
            )
            SELECT count(*) FROM stg_review_slim
            """
        ).fetchone()[0]

        self.assertEqual(n_rows, 2)


if __name__ == "__main__":
    unittest.main()
