import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

MODEL_PATH = os.getenv("MODEL_PATH", "/models/ppo_final.zip")
TICKERS = [t for t in os.getenv("TICKERS", "").split(",") if t]

_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = PPO.load(MODEL_PATH, device="cpu")
    _state["model"] = model
    _state["market_shape"] = tuple(model.observation_space["market"].shape)
    _state["num_assets"] = int(model.observation_space["portfolio"].shape[0])
    yield
    _state.clear()


app = FastAPI(title="DRL Portfolio Inference", version="1.0.0", lifespan=lifespan)


class Observation(BaseModel):
    market: list = Field(..., description="Nested list shaped (window, assets, features)")
    portfolio: list = Field(..., description="Current weights, length == num_assets")


class Allocation(BaseModel):
    weights: list
    weighted_assets: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ready",
        "num_assets": _state["num_assets"],
        "market_shape": _state["market_shape"],
    }


@app.post("/predict", response_model=Allocation)
def predict(obs: Observation):
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        market = np.asarray(obs.market, dtype=np.float32)
        portfolio = np.asarray(obs.portfolio, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid numeric payload: {exc}")

    if market.shape != _state["market_shape"]:
        raise HTTPException(
            status_code=422,
            detail=f"market shape {market.shape} != expected {_state['market_shape']}",
        )
    if portfolio.shape != (_state["num_assets"],):
        raise HTTPException(
            status_code=422,
            detail=f"portfolio shape {portfolio.shape} != expected {(_state['num_assets'],)}",
        )

    action, _ = model.predict({"market": market, "portfolio": portfolio}, deterministic=True)
    action = np.asarray(action, dtype=np.float64).clip(min=0.0)
    total = action.sum()
    weights = action / total if total > 0 else np.ones_like(action) / action.size

    result = {"weights": weights.tolist()}
    if TICKERS and len(TICKERS) == weights.size:
        result["weighted_assets"] = {t: float(w) for t, w in zip(TICKERS, weights)}
    return result
