from fastapi import APIRouter, HTTPException
from app.schemas.optimize import OptimizeRequest, OptimizeResponse

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    # Early validation heuristics
    if not req.channels:
        raise HTTPException(status_code=400, detail={"error": "validation_failed", "messages": ["no_channels"]})
    if not req.products:
        raise HTTPException(status_code=400, detail={"error": "validation_failed", "messages": ["no_products"]})
    min_cost = min((arr[1] for arr in req.channels.values() if len(arr) >= 2 and arr[1] > 0), default=0)
    if min_cost > 0 and req.budget > 0 and req.budget < min_cost:
        raise HTTPException(status_code=400, detail={"error": "validation_failed", "messages": ["budget_too_small_min_cost"]})

    # Deterministic stubbed optimizer (proportional budget allocation)
    budget = float(req.budget)
    avg_ltv = 0.0
    if req.products:
        vals = [float(x) for x in req.products.values() if float(x) >= 0]
        avg_ltv = sum(vals) / len(vals) if vals else 0.0

    # Base response rate by model when RR is disabled
    base_rr = 0.02 if req.model == "model1" else (0.018 if req.model == "model2" else 0.017)

    # Weights for allocation
    weights = {}
    for ch, arr in req.channels.items():
        rr = float(arr[2]) if (req.enable_rr and len(arr) == 3) else base_rr
        weights[ch] = max(rr, 1e-9)
    total_w = sum(weights.values()) or 1.0

    channels_usage = {}
    offers_total = 0
    spend_total = 0.0
    revenue_total = 0.0

    # First pass: allocate spend, then compute planned offers
    planned_by_ch = {}
    spend_by_ch = {}
    revenue_by_ch = {}

    for ch, arr in req.channels.items():
        max_offers = max(0, int(arr[0]))
        cost = max(0.0, float(arr[1]))
        rr = float(arr[2]) if (req.enable_rr and len(arr) == 3) else base_rr

        alloc = budget * (weights[ch] / total_w) if budget > 0 else 0.0
        # planned limited by both budget and max_offers
        affordable = int(alloc // cost) if cost > 0 else max_offers
        planned = min(max_offers, max(0, affordable))
        spend = float(planned) * cost
        conversions = planned * rr
        revenue = conversions * avg_ltv

        planned_by_ch[ch] = planned
        spend_by_ch[ch] = spend
        revenue_by_ch[ch] = revenue

        offers_total += planned
        spend_total += spend
        revenue_total += revenue

    roi = (revenue_total - spend_total) / spend_total * 100 if spend_total > 0 else 0.0
    summary = (
        budget,
        spend_total,
        (spend_total / budget * 100.0) if budget > 0 else 0.0,
        revenue_total,
        roi,
        int(sum((planned_by_ch[ch] * (float(req.channels[ch][2]) if (req.enable_rr and len(req.channels[ch]) == 3) else base_rr) for ch in planned_by_ch))),
    )

    channels_usage = {ch: (planned_by_ch[ch], spend_by_ch[ch], revenue_by_ch[ch]) for ch in req.channels.keys()}

    # products_distribution stub: равномерно делим офферы и возвращаем LTV как avg_affinity_revenue
    prod_count = max(len(req.products), 1)
    per_prod = int(offers_total // prod_count)
    products_distribution = {pid: (per_prod, float(ltv)) for pid, ltv in req.products.items()}

    return OptimizeResponse(
        summary=summary,
        channels_usage=channels_usage,
        products_distribution=products_distribution,
    )
