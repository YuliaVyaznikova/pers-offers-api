import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "df"
PRODUCT_MAP_PATH = ROOT / "product_csv_map.json"

def main():
    with open(PRODUCT_MAP_PATH, "r", encoding="utf-8") as f:
        csv_map = json.load(f)

    for product_id, filename in csv_map.items():
        csv_path = (DATA_DIR / filename).resolve()
        if not csv_path.exists():
            print(f"[convert] skip: CSV not found {csv_path}")
            continue

        parquet_path = csv_path.with_suffix(".parquet")
        print(f"[convert] {product_id}: {csv_path.name} -> {parquet_path.name}")

        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c.lower() in {"user_id", "catboost_proba", "lama_proba", "lgbm_proba"}]
        if not cols:
            cols = list(df.columns)
        df[cols].to_parquet(parquet_path, index=False)
        print(f"[convert] written: {parquet_path}")

if __name__ == "__main__":
    main()