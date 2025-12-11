from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import json
from pathlib import Path

import pandas as pd

from app.core.settings import get_settings


# Map backend model names to CSV column names
MODEL_COL = {
    "catboost": "catboost_proba",
    "lama": "lama_proba",
    "lgbm": "lgbm_proba",
}


@lru_cache(maxsize=None)
def _load_product_csv_map() -> Dict[str, str]:
    st = get_settings()
    with open(st.product_csv_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _load_model_ranking() -> Dict[str, List[str]]:
    st = get_settings()
    with open(st.model_ranking_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_actual_model_for_product(product_id: str, frontend_model: str) -> str:
    # frontend_model is one of model1|model2|model3
    rank = _load_model_ranking().get(product_id) or ["catboost", "lama", "lgbm"]
    idx = 0 if frontend_model == "model1" else 1 if frontend_model == "model2" else 2
    if idx >= len(rank):
        idx = 0
    return rank[idx]


def _read_product_df(product_id: str, actual_model: str) -> pd.DataFrame:
    st = get_settings()
    csv_map = _load_product_csv_map()
    filename = csv_map.get(product_id)
    if not filename:
        raise FileNotFoundError(f"No CSV mapped for product {product_id}")
    path = (st.data_dir / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    user_col = "user_id" if "user_id" in df.columns else df.columns[0]
    proba_col = MODEL_COL.get(actual_model)
    if proba_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == (proba_col or "").lower()]
        if not matches:
            raise ValueError(f"Probability column for model '{actual_model}' not found in {path.name}")
        proba_col = matches[0]

    clean = pd.DataFrame({
        "client_id": df[user_col],
        "affinity_prob": pd.to_numeric(df[proba_col], errors="coerce"),
    })
    clean = clean.dropna()
    clean = clean[clean["affinity_prob"] > 1e-6].copy()
    clean["product_id"] = product_id
    return clean


def _build_base_df(products: Dict[str, float], frontend_model: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for pid, revenue in products.items():
        actual = _resolve_actual_model_for_product(pid, frontend_model)
        dfp = _read_product_df(pid, actual)
        dfp["product_revenue"] = float(revenue)
        frames.append(dfp)
    if not frames:
        raise ValueError("No product data loaded")
    return pd.concat(frames, ignore_index=True)


def _solve_greedy(df_base: pd.DataFrame, channels: Dict[str, Tuple[int, float, float]], budget: float) -> pd.DataFrame:
    # channels: name -> (limit, cost, prob)
    chan_records = [
        {"canal_id": k, "channel_cost": float(v[1]), "channel_prob": float(v[2])}
        for k, v in channels.items()
    ]
    df_channels = pd.DataFrame(chan_records)
    df_full = df_base.merge(df_channels, how="cross")

    df_full["expected_revenue"] = (
        df_full["product_revenue"] * df_full["affinity_prob"] * df_full["channel_prob"]
    )
    df_full = df_full[df_full["expected_revenue"] > df_full["channel_cost"]].copy()
    df_full["roi_score"] = df_full["expected_revenue"] / df_full["channel_cost"]
    df_full.sort_values(by="roi_score", ascending=False, inplace=True)

    used_clients = set()
    chan_usage = {k: 0 for k in channels}
    spent = 0.0
    selected_idx: List[int] = []

    # Speed arrays
    idx_vals = df_full.index.values
    client_vals = df_full["client_id"].values
    channel_vals = df_full["canal_id"].values
    cost_vals = df_full["channel_cost"].values

    for i in range(len(idx_vals)):
        cost = float(cost_vals[i])
        if spent + cost > budget:
            continue
        client = client_vals[i]
        if client in used_clients:
            continue
        ch = channel_vals[i]
        if chan_usage[ch] >= int(channels[ch][0]):
            continue
        selected_idx.append(idx_vals[i])
        used_clients.add(client)
        chan_usage[ch] += 1
        spent += cost
        if budget - spent < 1e-6:
            break

    res = df_full.loc[selected_idx].copy()
    res.rename(columns={"channel_cost": "cost"}, inplace=True)
    return res


def run_optimizer(frontend_model: str, budget: float, enable_rr: bool, channels: Dict[str, List[float]], products: Dict[str, float]):
    """
    Returns dict(summary, channels_usage, products_distribution) compatible with OptimizeResponse.
    """
    # normalize channels to (limit, cost, rr)
    # if enable_rr is False, set rr=base depending on frontend_model
    base_rr = 0.02 if frontend_model == "model1" else (0.018 if frontend_model == "model2" else 0.017)
    ch3: Dict[str, Tuple[int, float, float]] = {}
    for k, arr in channels.items():
        limit = int(arr[0]) if len(arr) >= 1 else 0
        cost = float(arr[1]) if len(arr) >= 2 else 0.0
        rr = float(arr[2]) if (enable_rr and len(arr) >= 3) else base_rr
        ch3[k] = (max(0, limit), max(0.0, cost), max(0.0, min(1.0, rr)))

    df_base = _build_base_df(products, frontend_model)
    df_res = _solve_greedy(df_base, ch3, float(budget))

    # Build response
    if df_res.empty:
        summary = (float(budget), 0.0, 0.0, 0.0, 0.0, 0)
        return {
            "summary": summary,
            "channels_usage": {k: (0, 0.0, 0.0) for k in channels.keys()},
            "products_distribution": {k: (0, 0.0) for k in products.keys()},
        }

    spend_total = float(df_res["cost"].sum())
    revenue_total = float(df_res["expected_revenue"].sum())
    spend_pct = (spend_total / float(budget) * 100.0) if budget > 0 else 0.0
    roi = ((revenue_total - spend_total) / spend_total * 100.0) if spend_total > 0 else 0.0
    reach = int(len(df_res))

    summary = (
        float(budget),
        round(spend_total, 2),
        round(spend_pct, 2),
        round(revenue_total, 2),
        round(roi, 2),
        reach,
    )

    ch_group = df_res.groupby("canal_id").agg(
        count=("client_id", "count"), cost=("cost", "sum"), rev=("expected_revenue", "sum")
    )
    channels_usage = {}
    for ch in channels.keys():
        if ch in ch_group.index:
            row = ch_group.loc[ch]
            channels_usage[ch] = (int(row["count"]), round(float(row["cost"]), 2), round(float(row["rev"]), 2))
        else:
            channels_usage[ch] = (0, 0.0, 0.0)

    prod_group = df_res.groupby("product_id").agg(
        count=("client_id", "count"), avg_rev=("expected_revenue", "mean")
    )
    products_distribution = {}
    for pid in products.keys():
        if pid in prod_group.index:
            row = prod_group.loc[pid]
            products_distribution[pid] = (int(row["count"]), round(float(row["avg_rev"]), 2))
        else:
            products_distribution[pid] = (0, 0.0)

    return {"summary": summary, "channels_usage": channels_usage, "products_distribution": products_distribution}
