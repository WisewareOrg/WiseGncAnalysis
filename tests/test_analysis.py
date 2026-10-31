import h5py
import pytest
from gnc_analysis.imu_dead_reckoning import analyze

from tests.synthetic import build_campaign


def test_load_runs_rejects_mismatched_time_base(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=2, n_steps=10)
    # Same sample count as every other run, but a different t_s grid — the length check alone
    # would miss this; analyze() must also check the time base itself.
    with h5py.File(path, "a") as f:
        dataset = f["runs/0001/t_s"]
        dataset[...] = dataset[()] + 0.5

    with pytest.raises(ValueError, match="time base"):
        analyze(path, tmp_path / "out")


def test_fitted_growth_rates_match_injected_drift(tmp_path):
    attitude_drift_rate_deg_s = 2.0
    position_drift_rate_m_s = 0.5
    campaign_path = build_campaign(
        tmp_path / "campaign.h5",
        n_runs=5,
        n_steps=60,
        attitude_drift_rate_deg_s=attitude_drift_rate_deg_s,
        position_drift_rate_m_s=position_drift_rate_m_s,
    )
    out_dir = tmp_path / "out"

    analyze(campaign_path, out_dir)

    for figure in ("attitude_error.png", "position_error.png", "velocity_error.png"):
        assert (out_dir / figure).is_file()

    summary = (out_dir / "summary.md").read_text()
    fitted_attitude_rate = _growth_rate(summary, "Attitude error")
    fitted_position_rate = _growth_rate(summary, "Position error")

    assert fitted_attitude_rate == pytest.approx(attitude_drift_rate_deg_s, rel=0.1)
    assert fitted_position_rate == pytest.approx(position_drift_rate_m_s, rel=0.1)


def _growth_rate(summary: str, heading: str) -> float:
    lines = summary.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"## {heading}")
    rate_line = next(line for line in lines[start:] if line.startswith("- Growth rate"))
    return float(rate_line.split(":")[1].strip().split(" ")[0])
