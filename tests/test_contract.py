import pandas as pd

from contract.v0 import check_campaign, check_run_file
from tests.synthetic import build_campaign


def test_good_campaign_conforms(tmp_path):
    build_campaign(tmp_path, n_runs=3)
    assert check_campaign(tmp_path) == []


def test_good_run_file_conforms(tmp_path):
    build_campaign(tmp_path, n_runs=1)
    assert check_run_file(tmp_path / "run_0000.csv") == []


def test_wrong_column_order_fails(tmp_path):
    build_campaign(tmp_path, n_runs=1)
    run_path = tmp_path / "run_0000.csv"
    df = pd.read_csv(run_path)
    cols = df.columns.tolist()
    cols[1], cols[2] = cols[2], cols[1]
    df[cols].to_csv(run_path, index=False)

    problems = check_run_file(run_path)
    assert problems
    assert any("out of order" in p for p in problems)


def test_nan_fails(tmp_path):
    build_campaign(tmp_path, n_runs=1)
    run_path = tmp_path / "run_0000.csv"
    df = pd.read_csv(run_path)
    df.loc[2, "truth_sigma_1"] = float("nan")
    df.to_csv(run_path, index=False)

    problems = check_run_file(run_path)
    assert problems
    assert any("NaN" in p for p in problems)


def test_missing_file_named_in_manifest_fails(tmp_path):
    build_campaign(tmp_path, n_runs=2)
    (tmp_path / "run_0001.csv").unlink()

    problems = check_campaign(tmp_path)
    assert problems
    assert any("run_0001.csv" in p for p in problems)
