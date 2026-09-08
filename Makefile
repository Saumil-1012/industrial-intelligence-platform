
# Industrial Intelligence Platform — Makefile
# One-command setup and operations for any developer.


.PHONY: help install setup download-data validate featurize \
        train-supply train-airline train-all \
        api dashboard mlflow start stop \
        drift test lint clean

PYTHON   = python
PIP      = pip
MODULE   = industrial_intelligence_platform

#  Help 
help:
	@echo ""
	@echo "  Industrial Intelligence Platform"
	@echo "  "
	@echo "  make install          Install all dependencies"
	@echo "  make setup            Full first-time setup (install + data + featurize)"
	@echo ""
	@echo "  make download-data    Download M5 + DOT datasets from Kaggle"
	@echo "  make validate         Run Great Expectations on all datasets"
	@echo "  make featurize        Build features + write to Feast"
	@echo ""
	@echo "  make train-supply     Train supply chain LightGBM models"
	@echo "  make train-airline    Train airline XGBoost models"
	@echo "  make train-all        Train both modules"
	@echo ""
	@echo "  make api              Start FastAPI server (localhost:8000)"
	@echo "  make dashboard        Start Streamlit dashboard (localhost:8501)"
	@echo "  make mlflow           Start MLflow UI (localhost:5000)"
	@echo "  make start            Start full stack via Docker Compose"
	@echo "  make stop             Stop all Docker services"
	@echo ""
	@echo "  make drift            Run Evidently drift check (both modules)"
	@echo "  make test             Run full test suite"
	@echo "  make lint             Lint all Python files"
	@echo "  make clean            Remove build artifacts"
	@echo ""

# Install 
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅  Dependencies installed"

#  First-time setup 
setup: install
	@test -f .env || (cp .env.example .env && echo "⚠️  Created .env — add your KAGGLE credentials")
	mkdir -p models/supply_chain models/airline
	mkdir -p mlops/evidently/reports
	mkdir -p mlruns
	@echo "✅  Project setup complete"
	@echo "   Next: edit .env, then run: make download-data"

# Data 
download-data:
	$(PYTHON) data/supply_chain/download.py
	$(PYTHON) data/airline/download.py
	@echo "✅  Datasets downloaded"

validate:
	@echo "Running Great Expectations — Supply Chain..."
	$(PYTHON) great_expectations/expectations/m5_sales_suite.py
	@echo "Running Great Expectations — Airline..."
	$(PYTHON) great_expectations/expectations/airline_suite.py
	@echo "✅  Validation complete"

featurize:
	$(PYTHON) feature_store/materialize.py --sample
	@echo "✅  Features materialized to Feast"

featurize-full:
	$(PYTHON) feature_store/materialize.py
	@echo "✅  Full feature materialization complete"

#  Training
train-supply:
	$(PYTHON) -m modules.supply_chain.models.train --sample --trials 20
	@echo "✅  Supply chain models trained"

train-supply-full:
	$(PYTHON) -m modules.supply_chain.models.train --trials 50
	@echo "✅  Supply chain full training complete"

train-airline:
	$(PYTHON) -m modules.airline.models.train --sample 0.05 --trials 20
	@echo "✅  Airline models trained"

train-airline-full:
	$(PYTHON) -m modules.airline.models.train --trials 50
	@echo "✅  Airline full training complete"

train-all: train-supply train-airline
	@echo "✅  All models trained"

# Services
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db

start:
	docker compose up -d
	@echo "✅  Stack started"
	@echo "   API:       http://localhost:8000/docs"
	@echo "   Dashboard: http://localhost:8501"
	@echo "   MLflow:    http://localhost:5000"
	@echo "   Airflow:   http://localhost:8080  (admin/admin)"

stop:
	docker compose down
	@echo "✅  Stack stopped"

# MLOps
drift:
	$(PYTHON) -m mlops.evidently.drift_monitor --module both
	@echo "✅  Drift report saved to mlops/evidently/reports/"

promote:
	$(PYTHON) mlops/mlflow/promote_model.py
	@echo "✅  Model promoted staging → production"

#  Quality
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-coverage:
	$(PYTHON) -m pytest tests/ --cov=modules --cov=api --cov-report=term-missing

lint:
	flake8 modules/ api/ dashboard/ mlops/ feature_store/ \
	    --max-line-length=120 --ignore=E501,W503,E302,W291 \
	    --exclude=__pycache__

#  Clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	@echo "✅  Clean"
