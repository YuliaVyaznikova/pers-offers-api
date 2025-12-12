from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import json
import time
from pathlib import Path

import pandas as pd

from app.core.settings import get_settings

# Optional MIP solver import (fallback to greedy if unavailable)
try:
    from mip import Model, xsum, BINARY, MAXIMIZE, OptimizationStatus  # type: ignore
    _MIP_AVAILABLE = True
except Exception:
    _MIP_AVAILABLE = False


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
    t0 = time.time()
    st = get_settings()
    csv_map = _load_product_csv_map()
    filename = csv_map.get(product_id)
    if not filename:
        raise FileNotFoundError(f"No CSV mapped for product {product_id}")
    path = (st.data_dir / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    print(f"[optimizer] reading CSV for product='{product_id}' model='{actual_model}' file='{path.name}' ...")
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
    elapsed = time.time() - t0
    print(f"[optimizer] CSV loaded product='{product_id}' rows={len(clean)} in {elapsed:.3f}s")
    return clean


def _build_base_df(products: Dict[str, float], frontend_model: str) -> pd.DataFrame:
    t0 = time.time()
    frames: List[pd.DataFrame] = []
    for pid, revenue in products.items():
        actual = _resolve_actual_model_for_product(pid, frontend_model)
        print(f"[optimizer] model mapping product='{pid}' frontend='{frontend_model}' -> actual='{actual}'")
        dfp = _read_product_df(pid, actual)
        dfp["product_revenue"] = float(revenue)
        frames.append(dfp)
    if not frames:
        raise ValueError("No product data loaded")
    base = pd.concat(frames, ignore_index=True)
    print(f"[optimizer] base dataframe built rows={len(base)} in {time.time()-t0:.3f}s")
    return base


def _solve_greedy(df_base: pd.DataFrame, channels: Dict[str, Tuple[int, float, float]], budget: float) -> pd.DataFrame:
    # channels: name -> (limit, cost, prob)
    t_all = time.time()
    chan_records = [
        {"canal_id": k, "channel_cost": float(v[1]), "channel_prob": float(v[2])}
        for k, v in channels.items()
    ]
    df_channels = pd.DataFrame(chan_records)
    t_merge0 = time.time()
    df_full = df_base.merge(df_channels, how="cross")
    t_merge = time.time() - t_merge0

    df_full["expected_revenue"] = (
        df_full["product_revenue"] * df_full["affinity_prob"] * df_full["channel_prob"]
    )
    before_filter = len(df_full)
    t_filter0 = time.time()
    df_full = df_full[df_full["expected_revenue"] > df_full["channel_cost"]].copy()
    t_filter = time.time() - t_filter0
    after_filter = len(df_full)
    t_sort0 = time.time()
    df_full["roi_score"] = df_full["expected_revenue"] / df_full["channel_cost"]
    df_full.sort_values(by="roi_score", ascending=False, inplace=True)
    t_sort = time.time() - t_sort0

    used_clients = set()
    chan_usage = {k: 0 for k in channels}
    spent = 0.0
    selected_idx: List[int] = []

    # Speed arrays
    idx_vals = df_full.index.values
    client_vals = df_full["client_id"].values
    channel_vals = df_full["canal_id"].values
    cost_vals = df_full["channel_cost"].values

    t_loop0 = time.time()
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
    t_loop = time.time() - t_loop0
    print(
        f"[optimizer] merge={t_merge:.3f}s rows_before={before_filter} filter={t_filter:.3f}s rows_after={after_filter} "
        f"sort={t_sort:.3f}s loop={t_loop:.3f}s selected={len(res)} spent={spent:.2f} total={time.time()-t_all:.3f}s"
    )
    return res

def _solve_mip(df_base: pd.DataFrame, channels: Dict[str, Tuple[int, float, float]], budget: float) -> pd.DataFrame:
    """
    Новая версия MIP-алгоритма (по syper_new.py), адаптированная под наши структуры.
    - Переменная на каждую тройку (client, product, channel)
    - Целевая: maximize (expected_revenue - cost)
    - Ограничения: бюджет (по cost), не более 1 оффера на клиента, лимиты по каналам
    - Сохраняем лимит по времени и логи.
    """
    if not _MIP_AVAILABLE:
        print("[optimizer][warn] mip is not available; falling back to greedy")
        return _solve_greedy(df_base, channels, budget)

    channel_names = list(channels.keys())
    ch_cost = {k: float(v[1]) for k, v in channels.items()}
    ch_prob = {k: float(v[2]) for k, v in channels.items()}
    ch_limit = {k: int(v[0]) for k, v in channels.items()}

    m = Model(sense=MAXIMIZE, solver_name='CBC')
    m.verbose = 0

    # Переменные решения и коэффициенты
    # ключ будет (row_index, channel)
    x_vars: Dict[Tuple[int, str], any] = {}
    rev_coeff: Dict[Tuple[int, str], float] = {}
    cost_coeff: Dict[Tuple[int, str], float] = {}

    total_candidates = len(df_base) * len(channel_names)
    for row in df_base.itertuples(index=True):
        ridx = int(row.Index)
        p_aff = float(getattr(row, 'affinity_prob'))
        price = float(getattr(row, 'product_revenue'))
        if p_aff <= 0:
            continue
        for ch in channel_names:
            exp_rev = price * p_aff * ch_prob[ch]
            cost = ch_cost[ch]
            key = (ridx, ch)
            x_vars[key] = m.add_var(var_type=BINARY)
            rev_coeff[key] = exp_rev
            cost_coeff[key] = cost

    if not x_vars:
        return pd.DataFrame(columns=["client_id", "product_id", "canal_id", "cost", "expected_revenue"])  # empty

    # Целевая: maximize sum(x * (rev - cost))
    m.objective = xsum(x_vars[k] * (rev_coeff[k] - cost_coeff[k]) for k in x_vars)
    # Бюджет: sum(x * cost) <= budget
    m += xsum(x_vars[k] * cost_coeff[k] for k in x_vars) <= float(budget)

    # Не более 1 оффера на клиента
    from collections import defaultdict
    client_to_keys: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for (ridx, ch) in x_vars.keys():
        client_to_keys[str(df_base.loc[ridx, 'client_id'])].append((ridx, ch))
    for keys in client_to_keys.values():
        m += xsum(x_vars[k] for k in keys) <= 1

    # Лимиты каналов
    ch_to_keys: Dict[str, List[Tuple[int, str]]] = {ch: [] for ch in channel_names}
    for k in x_vars.keys():
        ch_to_keys[k[1]].append(k)
    for ch in channel_names:
        m += xsum(x_vars[k] for k in ch_to_keys.get(ch, [])) <= ch_limit.get(ch, 0)

    # Лимит времени
    try:
        m.max_seconds = 120
    except Exception:
        pass
    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0
    print(f"[optimizer][mip] solved in {elapsed:.3f}s status={getattr(m, 'status', None)} obj={getattr(m, 'objective_value', None)}")

    # Сбор результата
    rows: List[Dict[str, any]] = []
    ok = hasattr(m, 'status') and m.status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
    if ok:
        for (ridx, ch), var in x_vars.items():
            try:
                val = float(var.x)
            except Exception:
                val = 0.0
            if val >= 0.99:
                rows.append({
                    'client_id': df_base.loc[ridx, 'client_id'],
                    'product_id': df_base.loc[ridx, 'product_id'],
                    'canal_id': ch,
                    'cost': float(cost_coeff[(ridx, ch)]),
                    'expected_revenue': float(rev_coeff[(ridx, ch)]),
                })
    else:
        print("[optimizer][mip][warn] no feasible/timeout -> fallback to greedy")
        return _solve_greedy(df_base, channels, budget)

    return pd.DataFrame(rows)

def run_optimizer(frontend_model: str, budget: float, enable_rr: bool, channels: Dict[str, List[float]], products: Dict[str, float], advanced: bool = False):
    """
    Returns dict(summary, channels_usage, products_distribution) compatible with OptimizeResponse.
    """
    # normalize channels to (limit, cost, rr)
    # when enable_rr is False, use per-channel defaults from JSON config with fallback
    st = get_settings()
    default_rr_map = {
        "sms": 0.15,
        "push": 0.15,
        "phone": 0.05,
        "email": 0.025,
        "social": 0.025,
        "web_banner": 0.025,
    }
    try:
        with open(st.rr_defaults_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                # normalize keys to lower-case
                default_rr_map.update({str(k).lower(): float(v) for k, v in loaded.items()})
    except Exception as e:
        print(f"[optimizer][warn] cannot load rr defaults from {st.rr_defaults_path}: {e}; using built-ins")
    ch3: Dict[str, Tuple[int, float, float]] = {}
    rr_debug: List[str] = []
    for k, arr in channels.items():
        limit = int(arr[0]) if len(arr) >= 1 else 0
        cost = float(arr[1]) if len(arr) >= 2 else 0.0
        if enable_rr and len(arr) >= 3:
            rr = float(arr[2])
            src = "user"
        else:
            rr = float(default_rr_map.get(k, 0.025))
            src = "default"
        rr = max(0.0, min(1.0, rr))
        ch3[k] = (max(0, limit), max(0.0, cost), rr)
        rr_debug.append(f"{k}={rr}({src})")
    print(f"[optimizer] resolved channel rr: {', '.join(rr_debug)}")

    t0 = time.time()
    df_base = _build_base_df(products, frontend_model)
    df_res = _solve_mip(df_base, ch3, float(budget)) if advanced else _solve_greedy(df_base, ch3, float(budget))
    print(f"[optimizer] total optimization time={time.time()-t0:.3f}s (base+solve)")

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


def run_optimizer_csv(frontend_model: str, budget: float, enable_rr: bool, channels: Dict[str, List[float]], products: Dict[str, float], advanced: bool = False) -> str:
    """
    Same inputs as run_optimizer. Returns CSV text for the selected offers.
    Columns: client_id, product_id, canal_id, cost, expected_revenue
    """
    # resolve rr and build as in run_optimizer
    st = get_settings()
    default_rr_map = {
        "sms": 0.15,
        "push": 0.15,
        "phone": 0.05,
        "email": 0.025,
        "social": 0.025,
        "web_banner": 0.025,
    }
    try:
        with open(st.rr_defaults_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                default_rr_map.update({str(k).lower(): float(v) for k, v in loaded.items()})
    except Exception:
        pass
    ch3: Dict[str, Tuple[int, float, float]] = {}
    for k, arr in channels.items():
        limit = int(arr[0]) if len(arr) >= 1 else 0
        cost = float(arr[1]) if len(arr) >= 2 else 0.0
        if enable_rr and len(arr) >= 3:
            rr = float(arr[2])
        else:
            rr = float(default_rr_map.get(k, 0.025))
        rr = max(0.0, min(1.0, rr))
        ch3[k] = (max(0, limit), max(0.0, cost), rr)

    df_base = _build_base_df(products, frontend_model)
    df_res = _solve_mip(df_base, ch3, float(budget)) if advanced else _solve_greedy(df_base, ch3, float(budget))
    if df_res.empty:
        return "client_id,product_id,canal_id,cost,expected_revenue\n"
    # ensure column order and export
    export_cols = ["client_id", "product_id", "canal_id", "cost", "expected_revenue"]
    present = [c for c in export_cols if c in df_res.columns]
    csv_text = df_res[present].to_csv(index=False)
    return csv_text
