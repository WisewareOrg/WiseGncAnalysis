import h5py
import numpy as np

from contract.v0 import check_campaign
from tests.synthetic import build_campaign


def test_good_campaign_conforms(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=3)
    assert check_campaign(path) == []


def test_missing_dataset_fails(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=2)
    with h5py.File(path, "a") as f:
        del f["runs/0000/truth/r_N"]

    problems = check_campaign(path)
    assert problems
    assert any("truth/r_N" in p for p in problems)


def test_nan_fails(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=2)
    with h5py.File(path, "a") as f:
        dataset = f["runs/0000/truth/sigma_BN"]
        data = dataset[()]
        data[2, 0] = np.nan
        dataset[...] = data

    problems = check_campaign(path)
    assert problems
    assert any("NaN" in p for p in problems)


def test_shuffled_t_s_fails(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=2)
    with h5py.File(path, "a") as f:
        dataset = f["runs/0000/t_s"]
        rng = np.random.default_rng(1)
        dataset[...] = rng.permutation(dataset[()])

    problems = check_campaign(path)
    assert problems
    assert any("strictly increasing" in p for p in problems)


def test_extra_uncounted_run_group_fails(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=2)
    with h5py.File(path, "a") as f:
        f.copy("runs/0000", "runs/0002")
        # root 'runs' attr is left at 2, so group 0002 is on disk but uncounted.

    problems = check_campaign(path)
    assert problems
    assert any("'runs' attr is 2" in p for p in problems)


def test_wrong_units_fails(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=2)
    with h5py.File(path, "a") as f:
        f["runs/0000/truth/r_N"].attrs["units"] = "ft"

    problems = check_campaign(path)
    assert problems
    assert any("units" in p for p in problems)


def test_mismatched_t_s_between_runs_fails(tmp_path):
    path = build_campaign(tmp_path / "campaign.h5", n_runs=2)
    with h5py.File(path, "a") as f:
        dataset = f["runs/0001/t_s"]
        dataset[...] = dataset[()] + 0.5

    problems = check_campaign(path)
    assert problems
    assert any("t_s differs" in p for p in problems)
