"""Attitude, position and velocity error analysis against the recorded-run contract v0.

For every run: attitude error e(t) = 4*atan|sigma_truth (-) sigma_est| in degrees, position error
|r_truth - r_est| in m, velocity error |v_truth - v_est| in m/s. Across the ensemble: a summary
and a plot per metric. No dependency on WiseGncDesign or Basilisk; see mrp.py for the local MRP
math.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from contract.v0 import COLUMNS as CONTRACT_COLUMNS  # noqa: E402
from gnc_analysis import mrp  # noqa: E402


def _axis_cols(prefix: str) -> list[str]:
    """The three per-axis columns for prefix (e.g. "truth_sigma"), picked out of the v0
    contract's own column list — contract.v0.COLUMNS is the only place these names are spelled,
    so a rename there cannot silently drift out of sync here."""
    cols = [c for c in CONTRACT_COLUMNS if c.startswith(f"{prefix}_")]
    if len(cols) != 3:
        raise ValueError(
            f"expected 3 columns with prefix {prefix!r} in the v0 contract, found {cols}"
        )
    return cols


SIGMA_TRUTH_COLS = _axis_cols("truth_sigma")
SIGMA_EST_COLS = _axis_cols("est_sigma")
R_TRUTH_COLS = _axis_cols("truth_r")
R_EST_COLS = _axis_cols("est_r")
V_TRUTH_COLS = _axis_cols("truth_v")
V_EST_COLS = _axis_cols("est_v")


def load_campaign(campaign_dir: Path) -> list[pd.DataFrame]:
    """Load every run named in campaign_dir/manifest.json, in run order. Loading by the
    manifest rather than a directory glob means a stray or stale file on disk that the
    manifest doesn't name is never pulled into the analysis."""
    campaign_dir = Path(campaign_dir)
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path}: manifest.json does not exist")

    manifest = json.loads(manifest_path.read_text())
    runs_meta = manifest.get("runs")
    if not runs_meta:
        raise ValueError(f"{manifest_path}: 'runs' is missing or empty")

    runs = []
    for entry in sorted(runs_meta, key=lambda e: e["run"]):
        run_path = campaign_dir / entry["file"]
        if not run_path.is_file():
            raise ValueError(
                f"{manifest_path}: run {entry['run']} names missing file {entry['file']}"
            )
        runs.append(pd.read_csv(run_path))

    lengths = {len(df) for df in runs}
    if len(lengths) != 1:
        raise ValueError(f"runs under {campaign_dir} have differing sample counts: {lengths}")

    reference_t_s = runs[0]["t_s"].to_numpy()
    for i, df in enumerate(runs[1:], start=1):
        if not np.allclose(df["t_s"].to_numpy(), reference_t_s, rtol=0, atol=1e-9):
            raise ValueError(
                f"runs under {campaign_dir} do not share a common t_s time base "
                f"(run index {i} differs from run 0)"
            )

    return runs


def attitude_error_deg(df: pd.DataFrame) -> np.ndarray:
    """Per-sample attitude error angle, in degrees, for one run."""
    truth = df[SIGMA_TRUTH_COLS].to_numpy()
    est = df[SIGMA_EST_COLS].to_numpy()

    err = mrp.subtract(truth, est)
    over_one = np.linalg.norm(err, axis=-1) > 1.0
    err[over_one] = mrp.shadow(err[over_one])

    return np.degrees(mrp.angle(err))


def _vector_error(df: pd.DataFrame, truth_cols: list[str], est_cols: list[str]) -> np.ndarray:
    truth = df[truth_cols].to_numpy()
    est = df[est_cols].to_numpy()
    return np.linalg.norm(truth - est, axis=-1)


def position_error_m(df: pd.DataFrame) -> np.ndarray:
    """Per-sample position error magnitude, in m, for one run."""
    return _vector_error(df, R_TRUTH_COLS, R_EST_COLS)


def velocity_error_m_s(df: pd.DataFrame) -> np.ndarray:
    """Per-sample velocity error magnitude, in m/s, for one run."""
    return _vector_error(df, V_TRUTH_COLS, V_EST_COLS)


def growth_rate(t_s: np.ndarray, rms: np.ndarray) -> float:
    """Slope of a linear fit of ensemble RMS error vs time."""
    slope, _ = np.polyfit(t_s, rms, 1)
    return float(slope)


@dataclass(frozen=True)
class Metric:
    label: str
    unit: str
    rate_unit: str
    compute: Callable[[pd.DataFrame], np.ndarray]
    figure_name: str


METRICS = [
    Metric("attitude error", "deg", "deg/s", attitude_error_deg, "attitude_error.png"),
    Metric("position error", "m", "m/s", position_error_m, "position_error.png"),
    Metric("velocity error", "m/s", "m/s^2", velocity_error_m_s, "velocity_error.png"),
]


def _plot_metric(
    out_dir: Path,
    metric: Metric,
    t_s: np.ndarray,
    errors: np.ndarray,
    mean_t: np.ndarray,
    std_t: np.ndarray,
    n_runs: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in errors:
        ax.plot(t_s, e, color="0.75", linewidth=0.5)
    ax.plot(t_s, mean_t, color="C0", linewidth=2, label="ensemble mean")
    ax.plot(t_s, mean_t + 3 * std_t, color="C0", linewidth=1.5, linestyle="--", label="±3σ")
    ax.plot(t_s, mean_t - 3 * std_t, color="C0", linewidth=1.5, linestyle="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(f"{metric.label} [{metric.unit}]")
    ax.set_title(f"{metric.label.capitalize()}, {n_runs} runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / metric.figure_name, dpi=150)
    plt.close(fig)


def analyze(campaign_dir: Path, out_dir: Path) -> None:
    """Load every run in campaign_dir, score attitude/position/velocity error, and write
    summary.md and one figure per metric into out_dir."""
    campaign_dir = Path(campaign_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_campaign(campaign_dir)
    t_s = runs[0]["t_s"].to_numpy()
    duration_s = float(t_s[-1] - t_s[0])

    summary_lines = [
        "# Attitude, position and velocity error summary",
        "",
        f"- Runs: {len(runs)}",
        f"- Duration: {duration_s:.1f} s",
    ]

    for metric in METRICS:
        errors = np.stack([metric.compute(df) for df in runs], axis=0)  # (n_runs, n_steps)
        mean_t = np.mean(errors, axis=0)
        std_t = np.std(errors, axis=0)
        rms_t = np.sqrt(np.mean(errors**2, axis=0))

        final = errors[:, -1]
        final_mean = float(np.mean(final))
        final_std = float(np.std(final))
        final_max = float(np.max(final))
        rate = growth_rate(t_s, rms_t)

        summary_lines += [
            "",
            f"## {metric.label.capitalize()}",
            f"- Final mean: {final_mean:.4f} {metric.unit}",
            f"- Final std: {final_std:.4f} {metric.unit}",
            f"- Final 3-sigma: {3 * final_std:.4f} {metric.unit}",
            f"- Final max: {final_max:.4f} {metric.unit}",
            f"- Growth rate (linear fit of ensemble RMS vs time): {rate:.6f} {metric.rate_unit}",
        ]

        _plot_metric(out_dir, metric, t_s, errors, mean_t, std_t, len(runs))

    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a recorded-run campaign's attitude, position and velocity error "
        "against the v0 contract."
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.campaign_dir, args.out)


if __name__ == "__main__":
    main()
