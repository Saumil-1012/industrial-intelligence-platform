"""
API tests — supply chain + airline endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "supply_chain" in data["modules"]
    assert "airline"       in data["modules"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# Supply Chain
def test_supply_forecast_demo():
    r = client.post("/supply/forecast", json={
        "product_id": "HOBBIES_1_001", "location_id": "CA_1", "horizon_days": 28,
    })
    assert r.status_code == 200
    data = r.json()
    assert "forecast" in data
    assert "p50" in data["forecast"]


def test_supply_anomaly():
    r = client.post("/supply/anomaly", json={
        "product_id": "HOBBIES_1_001",
        "current_sales": 120.0,
        "rolling_mean_28": 43.0,
        "rolling_std_28": 7.0,
    })
    assert r.status_code == 200
    assert "is_anomaly" in r.json()


def test_supply_drift():
    r = client.get("/supply/drift")
    assert r.status_code == 200


def test_supply_model_info():
    r = client.get("/supply/model-info")
    assert r.status_code == 200


#  Airline
def test_airline_delay_demo():
    r = client.post("/airline/delay", json={
        "flight_id": "AA101", "origin": "JFK", "dest": "LAX",
        "carrier": "AA", "scheduled_dep": 800.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert "delay_probability" in data


def test_airline_rootcause():
    r = client.post("/airline/rootcause", json={
        "flight_id": "AA101", "delay_observed": 45.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert "primary_cause" in data


def test_airline_cascade():
    r = client.post("/airline/cascade", json={
        "delayed_flight_id": "AA101", "delay_minutes": 75.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert "total_affected" in data
    assert "affected_flights" in data


def test_airline_drift():
    r = client.get("/airline/drift")
    assert r.status_code == 200


def test_airline_model_info():
    r = client.get("/airline/model-info")
    assert r.status_code == 200
