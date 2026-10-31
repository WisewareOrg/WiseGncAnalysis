import pytest
from gnc_analysis.imu_dead_reckoning import analyze

from tests.synthetic import build_campaign


def test_fitted_growth_rate_matches_injected_drift(tmp_path):
    drift_rate_deg_s = 2.0
    campaign_dir = build_campaign(
        tmp_path / "campaign", n_runs=5, n_steps=60, drift_rate_deg_s=drift_rate_deg_s
    )
    out_dir = tmp_path / "out"

    analyze(campaign_dir, out_dir)

    summary = (out_dir / "summary.md").read_text()
    assert (out_dir / "attitude_error.png").is_file()

    rate_line = next(
        line for line in summary.splitlines() if line.startswith("- Error growth rate")
    )
    fitted_rate = float(rate_line.split(":")[1].strip().split(" ")[0])

    assert fitted_rate == pytest.approx(drift_rate_deg_s, rel=0.1)
