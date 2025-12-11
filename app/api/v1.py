from fastapi import APIRouter, HTTPException
from app.schemas.optimize import OptimizeRequest, OptimizeResponse
from app.services.optimizer import run_optimizer
import time

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

    t0 = time.time()
    result = run_optimizer(
        frontend_model=req.model,
        budget=req.budget,
        enable_rr=req.enable_rr,
        channels=req.channels,
        products=req.products,
    )
    dt = time.time() - t0
    print(f"/optimize done in {dt:.3f}s; channels={len(req.channels)} products={len(req.products)} budget={req.budget}")

    return OptimizeResponse(
        summary=tuple(result["summary"]),
        channels_usage={k: tuple(v) for k, v in result["channels_usage"].items()},
        products_distribution={k: tuple(v) for k, v in result["products_distribution"].items()},
    )
