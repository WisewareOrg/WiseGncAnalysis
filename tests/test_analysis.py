import pandas as pd
import pytest
from gnc_analysis.imu_dead_reckoning import analyze, load_campaign

from tests.synthetic import build_campaign


def test_load_campaign_ignores_file_not_in_manifest(tmp_path):
    campaign_dir = build_campaign(tmp_path / "campaign", n_runs=2, n_steps=10)
    # A stray file the manifest never named, carrying a contract defect, sitting right next to
    # the real runs. load_campaign must read the manifest, not glob the directory, so this file
    # is never pulled into the analysis.
    df = pd.read_csv(campaign_dir / "run_0000.csv")
    df.loc[0, "t_s"] = 1
    df.to_csv(campaign_dir / "run_0002.csv", index=False)

    runs = load_campaign(campaign_dir)

    assert len(runs) == 2


def test_load_campaign_rejects_mismatched_time_base(tmp_path):
    campaign_dir = build_campaign(tmp_path / "campaign", n_runs=2, n_steps=10)
    # Same sample count as every other run, but a different t_s grid — the length check alone
    # would miss this; load_campaign must also check the time base itself.
    run_path = campaign_dir / "run_0001.csv"
    df = pd.read_csv(run_path)
    df["t_s"] = df["t_s"] + 0.5
    df.to_csv(run_path, index=False)

    with pytest.raises(ValueError, match="time base"):
        load_campaign(campaign_dir)


def test_fitted_growth_rates_match_injected_drift(tmp_path):
    attitude_drift_rate_deg_s = 2.0
    position_drift_rate_m_s = 0.5
    campaign_dir = build_campaign(
        tmp_path / "campaign",
        n_runs=5,
        n_steps=60,
        attitude_drift_rate_deg_s=attitude_drift_rate_deg_s,
        position_drift_rate_m_s=position_drift_rate_m_s,
    )
    out_dir = tmp_path / "out"

    analyze(campaign_dir, out_dir)

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
