# 🏭 Industrial Intelligence Platform

**Supply Chain Forecasting + Airline Operations Intelligence**

Production-grade ML platform built across two industrial domains with shared MLOps infrastructure.

**Supply Chain** — Hierarchical demand forecasting on 42,840 time series (Walmart M5 dataset) using LightGBM with Optuna tuning and quantile regression (P10/P50/P90). Real-time anomaly detection via Isolation Forest + SPC. SHAP explanations per prediction.

**Airline Operations** — Flight delay prediction trained on 1.1M real DOT flight records. XGBoost classifier (AUC 0.863, Precision 95.9%) with Platt calibration. 5-category root cause attribution. Cascading delay propagation via NetworkX directed graph.

**MLOps** — Feast feature store · Great Expectations data validation · MLflow experiment tracking · Evidently AI drift monitoring with auto-retraining · Apache Airflow orchestration · Docker Compose · GitHub Actions CI/CD.

> Built by **Saumil Savani** · BSc Information Technology · Technical University of Munich, Campus Heilbronn  
> 📧 savani600@gmail.com

---

## Live Results

| Module | Metric | Value |
|---|---|---|
| Airline | Dataset | DOT On-Time Performance 2015–2016 |
| Airline | Training records | 958,863 flights |
| Airline | AUC | **0.863** |
| Airline | Precision | **95.9%** |
| Airline | Delay MAE | **11.2 min** |
| Supply Chain | Dataset | Walmart M5 (42,840 time series) |
| Supply Chain | Model | LightGBM + Quantile Regression |

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/industrial-intelligence-platform
cd industrial-intelligence-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run demo instantly (no data needed)
```bash
# Terminal 1
uvicorn api.main:app --port 8000

# Terminal 2
streamlit run dashboard/app.py
```

Open **http://localhost:8501** — full dashboard runs in demo mode immediately.

### Train on real data
```bash
# 1. Add Kaggle credentials
cp .env.example .env   # fill in KAGGLE_USERNAME + KAGGLE_KEY

# 2. Download datasets
python data/supply_chain/download.py
python data/airline/download.py

# 3. Train
python -m modules.supply_chain.models.train --sample --trials 10
python -m modules.airline.models.train --years 2015 2016 --sample 0.1 --trials 15

# 4. Restart API — switches from DEMO → MODEL
uvicorn api.main:app --port 8000
streamlit run dashboard/app.py
```

### Full Docker stack
```bash
docker-compose up
```

| Service | URL |
|---|---|
| API + Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 (admin/admin) |

---

## Repository Structure

```text
.
├── data/
│   ├── supply_chain/          # M5 Forecasting dataset (42,840 time series)
│   └── airline/               # DOT On-Time Performance (20M+ records)
├── modules/
│   ├── supply_chain/
│   │   ├── features/          # Lag, rolling, calendar, price features
│   │   ├── models/            # LightGBM quantile regression P10/P50/P90
│   │   └── anomaly/           # Isolation Forest + SPC Z-score
│   └── airline/
│       ├── features/          # Rotation, congestion, weather features
│       ├── models/            # XGBoost + Platt calibration + root cause
│       └── cascade/           # NetworkX delay propagation graph
├── feature_store/             # Feast feature definitions + materialization
├── great_expectations/        # Data quality suites (21 checks)
├── mlops/
│   ├── airflow/dags/          # 3 DAGs: supply chain, airline, drift monitor
│   ├── evidently/             # PSI drift monitoring + auto-retrain trigger
│   └── mlflow/                # Model registry promotion script
├── api/                       # FastAPI — 8 endpoints across 2 modules
├── dashboard/                 # Streamlit — dual-tab operations dashboard
├── notebooks/                 # Runnable demos (no data needed)
├── tests/                     # 51 tests — API + feature unit tests
└── docs/                      # Architecture diagram
```

---

## Tech Stack

| Category | Tools |
|---|---|
| ML — Supply Chain | LightGBM, Isolation Forest, Optuna, SHAP |
| ML — Airline | XGBoost, Platt calibration, NetworkX, MultiOutputClassifier |
| Feature Store | Feast (5 feature views, online + offline) |
| Data Validation | Great Expectations (21 checks) |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently AI (PSI, auto-retrain trigger) |
| Orchestration | Apache Airflow (3 DAGs) |
| API | FastAPI (8 endpoints) |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker Compose (5 services) |
| CI/CD | GitHub Actions |

---

## API Reference

### Supply Chain
```bash
POST /supply/forecast    # P10/P50/P90 demand forecast + SHAP
POST /supply/anomaly     # Isolation Forest + SPC anomaly detection
GET  /supply/drift       # Live PSI drift score
GET  /supply/model-info  # Model metadata
```

### Airline
```bash
POST /airline/delay      # Delay probability + expected minutes + SHAP
POST /airline/rootcause  # 5-category root cause breakdown
POST /airline/cascade    # NetworkX cascade propagation
GET  /airline/drift      # Live drift score
GET  /airline/model-info # Model metadata
```

All endpoints work in **demo mode** before models are trained.

---

## Key Design Decisions

**Quantile regression (P10/P50/P90)** — Point forecasts alone are not enough for inventory decisions. Safety stock optimization needs uncertainty bounds.

**Platt scaling calibration** — Raw XGBoost scores are not probabilities. Platt scaling fits a sigmoid on held-out validation data for reliable probability outputs.

**Feast feature store** — Eliminates train-serve skew. Same feature computation runs offline (training) and online (inference).

**NetworkX cascade graph** — Airline delays are a directed graph problem. Flight connections are edges; delays inherit with buffer subtraction.

**Great Expectations before training** — Hard-fails the pipeline on bad data before anything reaches the model.

**Evidently AI PSI monitoring** — Tracks distribution shift per feature weekly. Auto-triggers retraining when PSI > 0.15.

---

## Dataset Sources

Not included in repo due to size. Download via:

```bash
python data/supply_chain/download.py
python data/airline/download.py
```

- [M5 Forecasting — Walmart](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [DOT Airline On-Time Performance](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)
- [OpenSky Network API](https://opensky-network.org/apidoc/rest.html)

---

## Contact

**Saumil Savani**  
BSc Information Technology · Technical University of Munich, Campus Heilbronn  
📧 ssavani600@gmail.com