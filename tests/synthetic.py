"""Synthetic recorded-run campaigns, used by both the contract tests and the analysis test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from contract.v0 import COLUMNS

TRUTH_SIGMA = np.array([0.0, 0.0, 0.0])
TRUTH_OMEGA = np.array([0.017, 0.017, 0.017])  # ~1 deg/s per axis
ERROR_AXIS = np.array([1.0, 0.0, 0.0])


def _run_rows(n_steps: int, dt: float, drift_rate_deg_s: float, rng: np.random.Generator):
    """One run's rows. Truth attitude is held at the identity MRP; the estimate drifts away from
    it such that the attitude error angle grows at drift_rate_deg_s (deg/s)."""
    t = np.arange(n_steps) * dt

    theta_rad = np.deg2rad(drift_rate_deg_s * t)
    mrp_mag = np.tan(theta_rad / 4.0) + rng.normal(0, 1e-5, size=n_steps)
    est_sigma = mrp_mag[:, None] * ERROR_AXIS[None, :]

    truth_sigma = np.tile(TRUTH_SIGMA, (n_steps, 1))
    truth_omega = np.tile(TRUTH_OMEGA, (n_steps, 1))
    est_dv = np.zeros((n_steps, 3))
    gyro = truth_omega + rng.normal(0, 1e-5, size=(n_steps, 3))
    accel = rng.normal(0, 1e-6, size=(n_steps, 3))

    return np.column_stack([t, truth_sigma, truth_omega, est_sigma, est_dv, gyro, accel])


def build_campaign(
    out_dir: Path,
    n_runs: int = 5,
    n_steps: int = 60,
    dt: float = 1.0,
    drift_rate_deg_s: float = 2.0,
    seed: int = 0,
    producer: str = "synthetic-test",
) -> Path:
    """Write a contract-v0-conformant campaign of n_runs runs into out_dir, each with attitude
    error growing at drift_rate_deg_s. Returns out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    runs = []
    for i in range(n_runs):
        data = _run_rows(n_steps, dt, drift_rate_deg_s, rng)
        file_name = f"run_{i:04d}.csv"
        pd.DataFrame(data, columns=COLUMNS).to_csv(out_dir / file_name, index=False)
        runs.append({"run": i, "file": file_name, "seeds": {"rng": seed + i}, "dispersions": {}})

    manifest = {
        "contract_version": 0,
        "producer": producer,
        "created_utc": "2026-10-31T00:00:00Z",
        "runs": runs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
