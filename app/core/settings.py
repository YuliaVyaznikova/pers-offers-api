from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent  # .../pers-offers-api

class Settings(BaseSettings):
    data_dir: Path = Field(default=PROJ_ROOT / "df")
    product_csv_map_path: Path = Field(default=PROJ_ROOT / "product_csv_map.json")
    model_ranking_path: Path = Field(default=PROJ_ROOT / "model_ranking.json")

    class Config:
        env_prefix = "API_"
        case_sensitive = False

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
