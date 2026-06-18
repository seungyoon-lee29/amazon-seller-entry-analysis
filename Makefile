# 아마존 셀러 진입 분석 — 파이프라인 진입점
# 사용: make <target>

PY := python3

.PHONY: help setup test smoke ingest-meta ingest-meta-cross universe ingest-reviews validate ingest-all transform transform-sample analyze analyze-sample unmet-needs settlement newcomer figures dashboard

help:
	@echo "make setup              의존성 설치"
	@echo "make test               빠른 회귀 테스트"
	@echo "make smoke              소규모 스모크 테스트 (메타 5k + 리뷰 20k)"
	@echo "make ingest-meta        메인 카테고리 메타데이터 수집 (~11.8GB 스트리밍)"
	@echo "make ingest-meta-cross  교차등록 카테고리 메타데이터 수집"
	@echo "make universe           타깃 니치 ASIN universe 구축"
	@echo "make ingest-reviews     리뷰 수집 (~31GB 스트리밍, 수 시간 소요)"
	@echo "make validate           데이터 품질 리포트 생성"
	@echo "make ingest-all         위 전체를 순서대로 실행"
	@echo "make transform          staging → marts 변환 (DuckDB SQL)"
	@echo "make analyze            Q1 니치 기회 스코어링 + 민감도 분석"
	@echo "make unmet-needs        Q2 미충족 니즈 텍스트 마이닝 (저평점 리뷰)"
	@echo "make settlement         Q3 안착 예측 모델 (LightGBM + SHAP + 모델 저장)"
	@echo "make newcomer           니치별 신규 진입자 안착률 (Q1 보강, Q3 역집계)"
	@echo "make figures            의사결정 차트 생성 → docs/figures/*.png"
	@echo "make dashboard          Streamlit 대시보드 실행 (self-serve 니치 탐색기)"

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m unittest discover -s tests

smoke: test
	$(PY) src/ingest/download_meta.py --limit 5000
	$(PY) src/ingest/build_universe.py --meta data/raw/meta_Home_and_Kitchen_sample5000.parquet
	$(PY) src/ingest/download_reviews.py --limit 20000
	$(PY) src/ingest/validate.py
	$(PY) src/transform/run_transform.py --sample
	$(PY) src/analyze/niche_score.py --sample --min-months 1 --min-active 1
	$(PY) src/analyze/make_figures.py --sample

ingest-meta:
	$(PY) src/ingest/download_meta.py

ingest-meta-cross:
	$(PY) src/ingest/download_meta.py --category Tools_and_Home_Improvement

universe:
	$(PY) src/ingest/build_universe.py

ingest-reviews:
	$(PY) src/ingest/download_reviews.py

validate:
	$(PY) src/ingest/validate.py

ingest-all: ingest-meta ingest-meta-cross universe ingest-reviews validate

transform:
	$(PY) src/transform/run_transform.py

transform-sample:
	$(PY) src/transform/run_transform.py --sample

analyze:
	$(PY) src/analyze/niche_score.py

analyze-sample:
	$(PY) src/analyze/niche_score.py --sample --min-months 1 --min-active 1

unmet-needs:
	$(PY) src/analyze/unmet_needs.py

settlement:
	$(PY) src/analyze/settlement_model.py

newcomer:
	$(PY) src/analyze/newcomer_winrate.py

figures:
	$(PY) src/analyze/make_figures.py

dashboard:
	$(PY) -m streamlit run streamlit_app.py
