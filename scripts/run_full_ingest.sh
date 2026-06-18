#!/bin/bash
# 전체 수집 → 변환 → Q1/Q2/Q3 분석 (재개 가능 / 재실행 안전).
#
# 전원이 꺼지거나 중단되면 그냥 다시 실행하세요:
#   caffeinate -is bash scripts/run_full_ingest.sh
# - 완료된 단계(meta/universe/reviews)는 완료 체크포인트를 보고 자동 skip
# - 중단된 단계는 .ckpt.json의 바이트 오프셋부터 재개(이미 받은 부분 재다운로드 안 함)
# caffeinate -is = 실행 동안 시스템/idle 잠자기 방지(전원 차단은 여전히 재실행 필요).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 9
TS=$(date +%Y%m%d_%H%M%S)
LOG="data/raw/_run_logs/full_ingest_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== [$(date '+%H:%M:%S')] $* ==="; }
freeg() { df -g . | awk 'NR==2{print $4}'; }
die() { echo "!!! FAILED at: $* ($(date))"; exit 1; }

# 스트리밍 단계는 httpx 클라이언트 종료 등 일시 오류로 프로세스가 죽을 수 있다.
# 각 재시도는 체크포인트(.ckpt.json) 오프셋에서 이어받으므로 재다운로드 비용이 없다.
# 사용: retry <설명> <명령...> (최대 8회, 점증 대기)
retry() {
  local label="$1"; shift
  local attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if "$@"; then return 0; fi
    [ "$attempt" -eq 8 ] && die "$label (8회 재시도 실패)"
    local wait=$(( attempt * 30 )); [ "$wait" -gt 120 ] && wait=120
    echo "--- $label 중단(attempt $attempt) → ${wait}s 후 체크포인트에서 재개 ---"
    sleep "$wait"
  done
}

echo "START $(date) | log=$LOG | free=$(freeg)G"
[ "$(freeg)" -lt 5 ] && die "disk <5G before start"

step "meta (Home_and_Kitchen, ~11.8GB stream)"
retry "download_meta main" python3 src/ingest/download_meta.py

step "meta-cross (Tools_and_Home_Improvement)"
retry "download_meta cross" python3 src/ingest/download_meta.py --category Tools_and_Home_Improvement

step "universe (wall_shelves ASINs)"
python3 src/ingest/build_universe.py || die "build_universe"
cat data/raw/_run_logs/universe_wall_shelves.json

step "reviews (~31GB stream — longest, resumable)"
retry "download_reviews" python3 src/ingest/download_reviews.py

step "validate (data_quality_report.md)"
python3 src/ingest/validate.py || die "validate"

step "transform (staging -> marts, full)"
if ! python3 src/transform/run_transform.py; then
  echo "--- transform 실패(WAL/권한 추정) → --db /tmp 폴백 재시도 ---"
  python3 src/transform/run_transform.py --db /tmp/analysis_full.duckdb || die "transform (WAL fallback)"
fi

step "analyze (Q1 niche scoring)"
python3 src/analyze/niche_score.py || die "niche_score"

step "analyze (Q2 unmet needs)"
python3 src/analyze/unmet_needs.py || die "unmet_needs"

step "analyze (Q3 settlement model)"
python3 src/analyze/settlement_model.py || die "settlement_model"

step "DONE"
echo "free after run: $(freeg)G"
echo "FINISH $(date)"
