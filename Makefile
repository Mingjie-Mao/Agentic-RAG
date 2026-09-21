PY := .venv/bin/python

.PHONY: setup infra models models-cpu migrate seed web run test integration browser \
	s3-retrieval s3-generate s3-public s3-freeze s3-validate s5-dialogues rerank-model \
	check-docs agent-benchmark trial-reset agent-hard-setup agent-hard-validate agent-hard-benchmark \
	checklist-ab multihop-subset multihop-ingest multihop-eval multihop-diagnose site-data

setup:
	uv venv --python 3.13 --allow-existing
	uv pip install --python .venv/bin/python -r requirements.lock
	uv pip install --python .venv/bin/python --no-deps -e .
	$(PY) -c "from pathlib import Path; p=Path('.env'); p.exists() or p.write_bytes(Path('.env.example').read_bytes())"
	npm --prefix web ci

infra:
	docker compose up -d --wait postgres opensearch

models:
	$(PY) scripts/setup_models.py

models-cpu:
	docker compose --profile cpu up -d ollama
	docker compose exec ollama ollama pull bge-m3:567m
	docker compose exec ollama ollama pull qwen2.5:7b-instruct

migrate:
	.venv/bin/alembic upgrade head

seed:
	$(PY) scripts/build_fixtures.py
	$(PY) scripts/validate_s0.py
	$(PY) scripts/seed.py

web:
	npm --prefix web run build

run:
	$(PY) scripts/run.py

test:
	.venv/bin/ruff check app agent scripts tests migrations
	.venv/bin/pytest -q
	$(PY) scripts/check_docs.py

integration:
	RAG_RUN_INTEGRATION=1 .venv/bin/pytest -q --junitxml=artifacts/s1-tests.xml

browser:
	npm --prefix web run test:e2e

s3-retrieval:
	$(PY) scripts/run_s3_baseline.py --dataset fixtures/s3 --split development --out artifacts/s3-baseline

s3-generate:
	$(PY) scripts/run_s3_baseline.py --dataset fixtures/s3 --split development --generate --out artifacts/s3-generated

s3-public:
	$(PY) scripts/run_s3_baseline.py --dataset fixtures/public_subset --split public --generate --out artifacts/s3-public

s3-validate:
	$(PY) scripts/validate_s3.py

s3-freeze:
	PYTHONPATH=scripts $(PY) scripts/freeze_s3.py

s5-dialogues:
	$(PY) scripts/build_s5_dialogues.py

rerank-model:
	$(PY) scripts/setup_reranker.py

check-docs:
	$(PY) scripts/check_docs.py

agent-benchmark:
	$(PY) scripts/run_agent_benchmark.py

agent-hard-validate:
	$(PY) scripts/run_agent_hard_benchmark.py --validate-only

agent-hard-setup:
	$(PY) scripts/setup_agent_hard_fixtures.py

agent-hard-benchmark:
	$(PY) scripts/run_agent_hard_benchmark.py

checklist-ab:
	$(PY) scripts/run_checklist_ab.py

multihop-subset:
	$(PY) scripts/build_multihop_subset.py --download

multihop-ingest:
	$(PY) scripts/ingest_multihop.py

multihop-eval:
	$(PY) scripts/run_multihop_eval.py

multihop-diagnose:
	$(PY) scripts/diagnose_multihop_verdicts.py

site-data:
	$(PY) scripts/build_site_data.py

trial-reset:
	$(PY) scripts/reset_trial.py
