"""Stage 1a — 상품 메타데이터 수집 (스트리밍, 전체 다운로드 금지).

raw jsonl(11.8GB)을 HuggingFace에서 스트리밍하며 필요한 필드만 parquet으로 저장.
결과: data/raw/meta_{category}.parquet (선별 필드, zstd 압축 → 약 1~2GB 예상)

사용:
    python src/ingest/download_meta.py                  # main_category 전체
    python src/ingest/download_meta.py --category Tools_and_Home_Improvement
    python src/ingest/download_meta.py --limit 5000     # 스모크 테스트
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.common import (  # noqa: E402
    PROJECT_ROOT, ShardSink, iter_jsonl_offsets, load_ckpt, load_config,
    parse_price, save_ckpt, write_run_log,
)

CKPT_EVERY = 500_000   # 이 행 수마다 샤드를 닫고 체크포인트 저장


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=cfg["dataset"]["main_category"])
    ap.add_argument("--limit", type=int, default=None, help="스모크 테스트용 행 제한")
    args = ap.parse_args()

    suffix = f"_sample{args.limit}" if args.limit else ""
    out = PROJECT_ROOT / cfg["paths"]["raw"] / f"meta_{args.category}{suffix}.parquet"
    ckpt_path = out.with_suffix(".ckpt.json")
    resumable = args.limit is None    # 샘플(--limit)은 재개 대상 아님
    sink = ShardSink(out, cfg["runtime"]["chunk_rows"])
    path_in_repo = f"raw/meta_categories/meta_{args.category}.jsonl"
    revision = cfg["dataset"].get("revision")

    # ── 재개/스킵 판단 ──
    start_offset = 0
    n_in = n_price_ok = 0
    if resumable:
        ckpt = load_ckpt(ckpt_path)
        if ckpt and ckpt.get("done"):
            if ckpt.get("dataset_revision") != revision:
                sys.exit(f"체크포인트 dataset revision이 현재 설정과 다릅니다: "
                         f"{ckpt.get('dataset_revision')} != {revision}")
            print(f"이미 완료됨(skip): {out.name}")
            return
        if ckpt is None and out.exists():
            sys.exit(f"canonical은 있지만 완료 체크포인트가 없습니다: {out.name}. "
                     "부분 파일일 수 있으니 파일을 검증하거나 삭제 후 재수집하세요.")
        if ckpt:
            if ckpt.get("dataset_revision") != revision:
                sys.exit(f"체크포인트 dataset revision이 현재 설정과 다릅니다: "
                         f"{ckpt.get('dataset_revision')} != {revision}")
            start_offset = ckpt["offset"]
            sink.resume_from(ckpt["shard_idx"])
            n_in, n_price_ok = ckpt["n_in"], ckpt["n_price_ok"]
            print(f"재개: offset={start_offset:,} shard={ckpt['shard_idx']} rows={n_in:,}")

    last_off = start_offset
    for row, off in iter_jsonl_offsets(
        cfg["dataset"]["repo_id"], path_in_repo, start_offset, revision=revision
    ):
        if args.limit and n_in >= args.limit:
            break
        n_in += 1
        last_off = off
        price = parse_price(row.get("price"))
        if price is not None:
            n_price_ok += 1
        sink.add({
            "parent_asin": row.get("parent_asin"),
            "title": (row.get("title") or "")[:500],
            "main_category": row.get("main_category"),
            # categories는 list → 분석 시 leaf 추출. 문자열로 직렬화해 저장.
            "categories": " > ".join(row.get("categories") or []),
            "price": price,
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number"),
            "store": row.get("store"),
        })
        if resumable and n_in % CKPT_EVERY == 0:
            sink.roll()
            save_ckpt(ckpt_path, {"offset": off, "shard_idx": sink.shard_idx,
                                  "n_in": n_in, "n_price_ok": n_price_ok,
                                  "dataset_revision": revision, "done": False})
            print(f"  ... {n_in:,} rows (ckpt @ offset {off:,})", flush=True)

    total = sink.finalize()
    if resumable:
        save_ckpt(ckpt_path, {"offset": last_off, "shard_idx": sink.shard_idx,
                              "n_in": n_in, "n_price_ok": n_price_ok,
                              "dataset_revision": revision, "done": True})
    stats = {
        "category": args.category, "rows_in": n_in, "rows_out": total,
        "price_parse_rate": round(n_price_ok / max(n_in, 1), 4),
        "dataset_revision": revision,
        "output": str(out.relative_to(PROJECT_ROOT)),
        "limit": args.limit,
    }
    write_run_log(f"meta_{args.category}{suffix}", stats)
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
