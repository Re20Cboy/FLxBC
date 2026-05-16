PYTHON_VERSION := 3.12
DB := data/flxbc.db
DEMO_ROUNDS ?= 50
REAL_ROUNDS ?= 50
NODULE_ROUNDS ?= 40
EXPERIMENT_ROUNDS ?= 50
REAL_MAX_TRAIN_SAMPLES ?= 2400
REAL_MAX_TEST_SAMPLES ?= 600
NODULE_MAX_TRAIN_SAMPLES ?= 800
NODULE_MAX_TEST_SAMPLES ?= 240
AUTO_STOP_ARGS := --early-stopping --early-stopping-monitor val_loss --early-stopping-mode min --early-stopping-patience 5 --early-stopping-min-delta 0.001 --min-rounds 5
REAL_AUTO_STOP_ARGS := --early-stopping --early-stopping-monitor val_loss --early-stopping-mode min --early-stopping-patience 8 --early-stopping-min-delta 0.001 --min-rounds 10
NODULE_AUTO_STOP_ARGS := --early-stopping --early-stopping-monitor val_loss --early-stopping-mode min --early-stopping-patience 6 --early-stopping-min-delta 0.001 --min-rounds 8
REAL_ADAPTIVE_ARGS := --adaptive-rounds --round-extension 25 --max-rounds-cap 200
NODULE_ADAPTIVE_ARGS := --adaptive-rounds --round-extension 15 --max-rounds-cap 120
SYNTHETIC_TARGET_ARGS ?= --target-accuracy 0.85
REAL_TARGET_ARGS ?=
NODULE_TARGET_ARGS ?=

.PHONY: setup setup-core setup-ml setup-chain setup-all init-db demo experiment quick-experiment demo-check medmnist-check api dashboard run run-real run-real-2d run-real-3d chain deploy-chain test lint clean

setup:
	uv sync --python $(PYTHON_VERSION) --extra app --extra dev
	npm install

setup-core:
	uv sync --python $(PYTHON_VERSION) --extra dev

setup-ml:
	uv sync --python $(PYTHON_VERSION) --extra app --extra ml --extra dev

setup-chain:
	uv sync --python $(PYTHON_VERSION) --extra app --extra chain --extra dev
	npm install

setup-all:
	uv sync --python $(PYTHON_VERSION) --all-extras
	npm install

init-db:
	uv run flxbc init-db --db $(DB)

demo:
	uv run flxbc demo --synthetic --run-id demo-local --rounds $(DEMO_ROUNDS) --clients 5 --device numpy $(AUTO_STOP_ARGS) $(SYNTHETIC_TARGET_ARGS)

experiment:
	uv run flxbc experiment --synthetic --run-id compare-local --rounds $(EXPERIMENT_ROUNDS) --clients 8 --device numpy $(AUTO_STOP_ARGS) $(SYNTHETIC_TARGET_ARGS)

quick-experiment:
	uv run flxbc experiment --synthetic --run-id compare-quick --rounds 2 --clients 3 --max-train-samples 72 --max-test-samples 24 --no-failures --device numpy

demo-check:
	uv run python -m scripts.demo_check

medmnist-check:
	uv run flxbc demo --dataset pneumoniamnist --run-id medmnist-check --rounds 1 --clients 2 --max-train-samples 32 --max-test-samples 16 --no-failures

api:
	FLXBC_DB=$(DB) uv run --extra app uvicorn flxbc.api:app --reload --host 127.0.0.1 --port 8000

dashboard:
	FLXBC_DB=$(DB) uv run --extra app streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501

run:
	uv run --extra app flxbc run --synthetic --run-id demo-local --rounds $(DEMO_ROUNDS) --clients 5 --device numpy $(AUTO_STOP_ARGS) $(SYNTHETIC_TARGET_ARGS) --db $(DB)

run-real: run-real-2d

run-real-2d:
	uv run --extra app --extra ml flxbc run --dataset pneumoniamnist --run-id real-demo-2d --rounds $(REAL_ROUNDS) --clients 5 --max-train-samples $(REAL_MAX_TRAIN_SAMPLES) --max-test-samples $(REAL_MAX_TEST_SAMPLES) $(REAL_AUTO_STOP_ARGS) $(REAL_ADAPTIVE_ARGS) $(REAL_TARGET_ARGS) --db $(DB)

run-real-3d:
	uv run --extra app --extra ml flxbc run --dataset nodulemnist3d --run-id real-demo-3d --rounds $(NODULE_ROUNDS) --clients 5 --max-train-samples $(NODULE_MAX_TRAIN_SAMPLES) --max-test-samples $(NODULE_MAX_TEST_SAMPLES) $(NODULE_AUTO_STOP_ARGS) $(NODULE_ADAPTIVE_ARGS) $(NODULE_TARGET_ARGS) --db $(DB)

chain:
	npm run chain

deploy-chain:
	npm run deploy:local

test:
	uv run --extra app --extra dev pytest -q

lint:
	uv run --extra dev ruff check .

clean:
	rm -rf artifacts data .pytest_cache
