"""Attitude-error analysis against the recorded-run contract v0: for every run,
e(t) = 4*atan|sigma_truth (-) sigma_est| in degrees; across the ensemble, a summary and a plot.
No dependency on WiseGncDesign or Basilisk; see mrp.py for the local MRP math.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from gnc_analysis import mrp  # noqa: E402

SIGMA_TRUTH_COLS = ["truth_sigma_1", "truth_sigma_2", "truth_sigma_3"]
SIGMA_EST_COLS = ["est_sigma_1", "est_sigma_2", "est_sigma_3"]


def load_campaign(campaign_dir: Path) -> list[pd.DataFrame]:
    """Load every run_*.csv in campaign_dir, sorted by filename."""
    paths = sorted(Path(campaign_dir).glob("run_*.csv"))
    if not paths:
        raise ValueError(f"no run_*.csv files found in {campaign_dir}")
    runs = [pd.read_csv(p) for p in paths]
    lengths = {len(df) for df in runs}
    if len(lengths) != 1:
        raise ValueError(f"runs under {campaign_dir} have differing sample counts: {lengths}")
    return runs


def attitude_error_deg(df: pd.DataFrame) -> np.ndarray:
    """Per-sample attitude error angle, in degrees, for one run."""
    truth = df[SIGMA_TRUTH_COLS].to_numpy()
    est = df[SIGMA_EST_COLS].to_numpy()

    err = mrp.subtract(truth, est)
    over_one = np.linalg.norm(err, axis=-1) > 1.0
    err[over_one] = mrp.shadow(err[over_one])

    return np.degrees(mrp.angle(err))


def growth_rate_deg_s(t_s: np.ndarray, rms_deg: np.ndarray) -> float:
    """Slope of a linear fit of ensemble RMS error vs time, in deg/s."""
    slope, _ = np.polyfit(t_s, rms_deg, 1)
    return float(slope)


def analyze(campaign_dir: Path, out_dir: Path) -> None:
    """Load every run in campaign_dir, score attitude error, and write summary.md and
    attitude_error.png into out_dir."""
    campaign_dir = Path(campaign_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_campaign(campaign_dir)
    t_s = runs[0]["t_s"].to_numpy()
    errors = np.stack([attitude_error_deg(df) for df in runs], axis=0)  # (n_runs, n_steps)

    mean_t = np.mean(errors, axis=0)
    std_t = np.std(errors, axis=0)
    rms_t = np.sqrt(np.mean(errors**2, axis=0))

    final = errors[:, -1]
    final_mean = float(np.mean(final))
    final_std = float(np.std(final))
    final_max = float(np.max(final))
    rate = growth_rate_deg_s(t_s, rms_t)
    duration_s = float(t_s[-1] - t_s[0])

    summary = (
        "# Attitude-error summary\n"
        "\n"
        f"- Runs: {len(runs)}\n"
        f"- Duration: {duration_s:.1f} s\n"
        f"- Final error mean: {final_mean:.4f} deg\n"
        f"- Final error std: {final_std:.4f} deg\n"
        f"- Final error 3-sigma: {3 * final_std:.4f} deg\n"
        f"- Final error max: {final_max:.4f} deg\n"
        f"- Error growth rate (linear fit of ensemble RMS vs time): {rate:.6f} deg/s\n"
    )
    (out_dir / "summary.md").write_text(summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    for e in errors:
        ax.plot(t_s, e, color="0.75", linewidth=0.5)
    ax.plot(t_s, mean_t, color="C0", linewidth=2, label="ensemble mean")
    ax.plot(t_s, mean_t + 3 * std_t, color="C0", linewidth=1.5, linestyle="--", label="±3σ")
    ax.plot(t_s, mean_t - 3 * std_t, color="C0", linewidth=1.5, linestyle="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("attitude error [deg]")
    ax.set_title(f"Attitude error, {len(runs)} runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "attitude_error.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a recorded-run campaign's attitude error against the v0 contract."
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.campaign_dir, args.out)


if __name__ == "__main__":
    main()
