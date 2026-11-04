"""Traffic simulator generating live inference traffic with controlled drift injection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.serving.app import app


def run_traffic_simulation(
    num_requests: int = 500,
    perturb_feature: str | None = "monthly_charges",
    perturb_magnitude: float = 1.4,
    api_key: str = "mlforge-dev-secret-key-change-in-prod",
) -> int:
    """Simulates production prediction traffic by sampling holdout data and posting to API."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    ref_file = base_dir / "data" / "reference" / "holdout_reference.parquet"

    if not ref_file.exists():
        raise FileNotFoundError(f"Reference snapshot not found at {ref_file}. Run training first.")

    df_ref = pd.read_parquet(ref_file)
    rng = np.random.default_rng(42)

    sample_indices = rng.choice(len(df_ref), size=num_requests, replace=True)
    df_sim = df_ref.iloc[sample_indices].copy()

    # Controlled drift injection per TRD §2 & §4
    if perturb_feature and perturb_feature in df_sim.columns:
        if perturb_feature == "monthly_charges":
            # Shift continuous charges upwards
            df_sim[perturb_feature] = np.clip(
                df_sim[perturb_feature] * perturb_magnitude, 18.0, 120.0
            )
        elif perturb_feature == "contract":
            # Shift contracts towards Month-to-month
            df_sim[perturb_feature] = "Month-to-month"
        print(f"[Simulator] Injected intentional drift onto '{perturb_feature}'.")

    records = df_sim.to_dict(orient="records")
    successful = 0

    with TestClient(app) as client:
        headers = {"X-API-Key": api_key}
        for rec in records:
            # Handle NaN in total_charges
            if pd.isna(rec.get("total_charges")):
                rec["total_charges"] = None

            resp = client.post("/predict", json=rec, headers=headers)
            if resp.status_code == 200:
                successful += 1

    print(f"[Simulator] Generated {successful}/{num_requests} successful simulated predictions.")
    return successful


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate production prediction traffic.")
    parser.add_argument("--count", type=int, default=500, help="Number of simulated requests.")
    parser.add_argument(
        "--perturb",
        default="monthly_charges",
        help="Feature name to inject synthetic drift into.",
    )
    args = parser.parse_args()

    run_traffic_simulation(num_requests=args.count, perturb_feature=args.perturb)
