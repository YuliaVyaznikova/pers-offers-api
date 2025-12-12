from fastapi import APIRouter, HTTPException
from app.schemas.optimize import OptimizeRequest, OptimizeResponse
from fastapi.responses import Response
import time
import asyncio
from typing import Dict, Any

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    # Lazy import to avoid heavy module import at cold start
    from app.services.optimizer import run_optimizer
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
        advanced=req.advanced,
    )
    dt = time.time() - t0
    print(f"/optimize done in {dt:.3f}s; channels={len(req.channels)} products={len(req.products)} budget={req.budget}")

    return OptimizeResponse(
        summary=tuple(result["summary"]),
        channels_usage={k: tuple(v) for k, v in result["channels_usage"].items()},
        products_distribution={k: tuple(v) for k, v in result["products_distribution"].items()},
    )


@router.post("/optimize_csv")
async def optimize_csv(req: OptimizeRequest):
    # Lazy import to avoid heavy module import at cold start
    from app.services.optimizer import run_optimizer_csv
    if not req.channels or not req.products:
        raise HTTPException(status_code=400, detail={"error": "validation_failed", "messages": ["no_channels_or_products"]})
    csv_text = run_optimizer_csv(
        frontend_model=req.model,
        budget=req.budget,
        enable_rr=req.enable_rr,
        channels=req.channels,
        products=req.products,
        advanced=req.advanced,
    )
    return Response(content=csv_text, media_type="text/csv")


# ---- Jobs storage: in-memory single-process queue (use workers=1 for MIP service) ----
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_COUNTER = 0
_MIP_SEM = asyncio.Semaphore(1)


async def _run_job(job_id: str, req: OptimizeRequest) -> None:
    global _JOBS
    started = time.time()
    _JOBS[job_id]["status"] = "running"
    _JOBS[job_id]["started_at"] = started
    try:
        # Lazy import
        from app.services.optimizer import run_optimizer
        if req.advanced:
            # Single concurrent MIP and run in background thread to avoid blocking event loop
            async with _MIP_SEM:
                t0 = time.time()
                res = await asyncio.to_thread(
                    run_optimizer,
                    frontend_model=req.model,
                    budget=req.budget,
                    enable_rr=req.enable_rr,
                    channels=req.channels,
                    products=req.products,
                    advanced=True,
                )
                solve_ms = int((time.time() - t0) * 1000)
        else:
            res = run_optimizer(
                frontend_model=req.model,
                budget=req.budget,
                enable_rr=req.enable_rr,
                channels=req.channels,
                products=req.products,
                advanced=False,
            )
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = res
        if req.advanced:
            _JOBS[job_id]["solve_ms"] = solve_ms
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(e)
    finally:
        _JOBS[job_id]["finished_at"] = time.time()


@router.post("/optimize_async")
async def optimize_async(req: OptimizeRequest):
    """Create a job and return job_id immediately. The client should poll /jobs/{id}."""
    global _JOB_COUNTER, _JOBS
    _JOB_COUNTER += 1
    job_id = f"job-{_JOB_COUNTER}-{int(time.time())}"
    _JOBS[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "advanced": bool(req.advanced),
    }
    # fire-and-forget background task
    asyncio.create_task(_run_job(job_id, req))
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    if job.get("status") == "done" and isinstance(job.get("result"), dict):
        rdata = job["result"]
        return {
            "job_id": job_id,
            "status": job["status"],
            "result": {
                "summary": tuple(rdata.get("summary", ())),
                "channels_usage": {k: tuple(v) for k, v in rdata.get("channels_usage", {}).items()},
                "products_distribution": {k: tuple(v) for k, v in rdata.get("products_distribution", {}).items()},
            },
            "solve_ms": job.get("solve_ms"),
        }
    return {"job_id": job_id, **job}
