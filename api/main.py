"""
Industrial Intelligence Platform — FastAPI Main App
Two domain routers: /supply/* and /airline/*
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import supply_chain, airline

app = FastAPI(
    title="Industrial Intelligence Platform",
    description="Supply Chain Forecasting + Airline Operations Intelligence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(supply_chain.router, prefix="/supply", tags=["Supply Chain"])
app.include_router(airline.router,      prefix="/airline", tags=["Airline"])


@app.get("/")
def root():
    return {
        "status":  "ok",
        "modules": ["supply_chain", "airline"],
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}
