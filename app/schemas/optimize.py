from typing import Dict, List, Literal, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationInfo

ModelName = Literal["model1", "model2", "model3"]

class OptimizeRequest(BaseModel):
    model: ModelName
    budget: float = Field(ge=0)
    advanced: bool
    enable_rr: bool
    channels: Dict[str, List[float]]
    products: Dict[str, float]

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v, info: ValidationInfo):
        enable_rr = bool((info.data or {}).get("enable_rr", False))
        for k, arr in v.items():
            if not isinstance(arr, list):
                raise ValueError(f"channels[{k}] must be list")
            if enable_rr:
                if len(arr) != 3:
                    raise ValueError(f"channels[{k}] must be [max, cost, rr] when enable_rr=true")
                max_offers, cost, rr = arr
                if rr < 0 or rr > 1:
                    raise ValueError(f"response_rate for channel {k} must be in [0,1]")
            else:
                if len(arr) != 2:
                    raise ValueError(f"channels[{k}] must be [max, cost] when enable_rr=false")
            # non-negative checks
            if arr[0] < 0 or arr[1] < 0:
                raise ValueError(f"negative numbers in channel {k}")
        return v

    @field_validator("products")
    @classmethod
    def validate_products(cls, v):
        if not v:
            raise ValueError("at least one product is required")
        for k, ltv in v.items():
            if ltv < 0:
                raise ValueError(f"ltv for {k} must be non-negative")
        return v

class OptimizeResponse(BaseModel):
    summary: Tuple[float, float, float, float, float, int]
    channels_usage: Dict[str, Tuple[int, float, float]]
    products_distribution: Dict[str, Tuple[int, float]]
