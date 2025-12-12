from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent  # .../pers-offers-api

class Settings(BaseSettings):
    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_prefix="API_",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    data_dir: Path = Field(default=PROJ_ROOT / "df")
    product_csv_map_path: Path = Field(default=PROJ_ROOT / "product_csv_map.json")
    model_ranking_path: Path = Field(default=PROJ_ROOT / "model_ranking.json")
    rr_defaults_path: Path = Field(default=PROJ_ROOT / "rr_defaults.json")

    # Redis URL for distributed queue/locks (e.g., redis://:pass@host:port/0)
    redis_url: str | None = Field(default=None)
    # Max number of concurrent MIP solves cluster-wide
    mip_max_concurrency: int = Field(default=1)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
