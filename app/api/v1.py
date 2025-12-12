from fastapi import APIRouter, HTTPException
from app.schemas.optimize import OptimizeRequest, OptimizeResponse
from fastapi.responses import Response
import time
import asyncio
from typing import Dict, Any, Optional
import json

from app.core.settings import get_settings
try:
    import redis.asyncio as redis  # type: ignore
except Exception:  # redis is optional; if missing, we will fallback to in-memory
    redis = None  # type: ignore

router = APIRouter()

# ---- Redis client (optional). If API_REDIS_URL is not provided or redis pkg missing, fallback to in-memory ----
_redis_client: Optional["redis.Redis"] = None  # type: ignore

def _get_redis() -> Optional["redis.Redis"]:  # type: ignore
    global _redis_client
    st = get_settings()
    if not st.redis_url or redis is None:
        return None
    if _redis_client is None:
        _redis_client = redis.from_url(st.redis_url, decode_responses=True)
    return _redis_client

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


# ---- Jobs storage: prefer Redis, fallback to in-memory (single-process only) ----
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_COUNTER = 0
_MIP_SEM = asyncio.Semaphore(1)  # local guard when Redis is not available

async def _acquire_global_slot() -> bool:
    r = _get_redis()
    st = get_settings()
    if r is None:
        # no redis -> rely on in-proc semaphore only
        await _MIP_SEM.acquire()
        return True
    # Use a counter mip:running with max from settings
    key = "mip:running"
    maxc = max(1, int(st.mip_max_concurrency))
    # Try to increment and check
    val = await r.incr(key)
    if val == 1:
        await r.expire(key, 600)
    if val > maxc:
        # revert and deny
        await r.decr(key)
        return False
    return True

async def _release_global_slot() -> None:
    r = _get_redis()
    if r is None:
        try:
            _MIP_SEM.release()
        except Exception:
            pass
        return
    try:
        await r.decr("mip:running")
    except Exception:
        pass


async def _run_job(job_id: str, req: OptimizeRequest) -> None:
    global _JOBS
    started = time.time()
    r = _get_redis()
    if r is None:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["started_at"] = started
    else:
        await r.hset(f"jobs:hash:{job_id}", mapping={"status": "running", "started_at": started})
    try:
        # Lazy import
        from app.services.optimizer import run_optimizer
        if req.advanced:
            # Acquire global slot (redis) or local semaphore
            got = await _acquire_global_slot()
            while not got:
                await asyncio.sleep(1.0)
                got = await _acquire_global_slot()
            try:
                t0 = time.time()
                res = run_optimizer(
                    frontend_model=req.model,
                    budget=req.budget,
                    enable_rr=req.enable_rr,
                    channels=req.channels,
                    products=req.products,
                    advanced=True,
                )
                solve_ms = int((time.time() - t0) * 1000)
            finally:
                await _release_global_slot()
        else:
            res = run_optimizer(
                frontend_model=req.model,
                budget=req.budget,
                enable_rr=req.enable_rr,
                channels=req.channels,
                products=req.products,
                advanced=False,
            )
        if r is None:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["result"] = res
            if req.advanced:
                _JOBS[job_id]["solve_ms"] = solve_ms
        else:
            await r.hset(f"jobs:hash:{job_id}", mapping={
                "status": "done",
                "result_json": json.dumps(res),
                **({"solve_ms": solve_ms} if req.advanced else {}),
            })
            await r.expire(f"jobs:hash:{job_id}", 7200)
    except Exception as e:
        if r is None:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(e)
        else:
            await r.hset(f"jobs:hash:{job_id}", mapping={"status": "error", "error": str(e)})
    finally:
        if r is None:
            _JOBS[job_id]["finished_at"] = time.time()
        else:
            await r.hset(f"jobs:hash:{job_id}", mapping={"finished_at": time.time()})


@router.post("/optimize_async")
async def optimize_async(req: OptimizeRequest):
    """Create a job and return job_id immediately. The client should poll /jobs/{id}."""
    global _JOB_COUNTER, _JOBS
    _JOB_COUNTER += 1
    job_id = f"job-{_JOB_COUNTER}-{int(time.time())}"
    r = _get_redis()
    if r is None:
        _JOBS[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "advanced": bool(req.advanced),
        }
    else:
        await r.hset(f"jobs:hash:{job_id}", mapping={
            "status": "queued",
            "created_at": time.time(),
            "advanced": json.dumps(bool(req.advanced)),
        })
        await r.expire(f"jobs:hash:{job_id}", 7200)
    # fire-and-forget background task
    asyncio.create_task(_run_job(job_id, req))
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    r = _get_redis()
    if r is None:
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
    else:
        data = await r.hgetall(f"jobs:hash:{job_id}")
        if not data:
            raise HTTPException(status_code=404, detail={"error": "not_found"})
        status = data.get("status")
        if status == "done" and data.get("result_json"):
            rdata = json.loads(data["result_json"]) if isinstance(data["result_json"], str) else {}
            return {
                "job_id": job_id,
                "status": status,
                "result": {
                    "summary": tuple(rdata.get("summary", ())),
                    "channels_usage": {k: tuple(v) for k, v in rdata.get("channels_usage", {}).items()},
                    "products_distribution": {k: tuple(v) for k, v in rdata.get("products_distribution", {}).items()},
                },
                "solve_ms": int(float(data.get("solve_ms", 0))) if data.get("solve_ms") else None,
            }
        # queued/running/error -> return raw fields
        resp = {"job_id": job_id, "status": status}
        if "error" in data:
            resp["error"] = data["error"]
        return resp
