"""v0 recorded-run contract: one HDF5 file per campaign, nothing else.

See `contract/README.md` for the documented layout. Provisional — the format trade becomes an
ADR here when it matters, not at v0.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

CONTRACT_VERSION = 0

ROOT_ATTRS = [
    "contract_version",
    "producer",
    "created_utc",
    "seed",
    "runs",
    "versions_bsk",
    "versions_gnc_algorithms",
    "versions_wise_dynamics",
]

# Dataset path within a run group -> (expected ndim, expected size of the last axis or None for
# a 1-D dataset, expected `units` attribute).
DATASETS: dict[str, tuple[int, int | None, str]] = {
    "t_s": (1, None, "s"),
    "truth/sigma_BN": (2, 3, "MRP"),
    "truth/omega_BN_B": (2, 3, "rad/s"),
    "truth/r_N": (2, 3, "m"),
    "truth/v_N": (2, 3, "m/s"),
    "est/sigma_BN": (2, 3, "MRP"),
    "est/r_N": (2, 3, "m"),
    "est/v_N": (2, 3, "m/s"),
    "imu/gyro_P": (2, 3, "rad/s"),
    "imu/accel_P": (2, 3, "m/s^2"),
}


@dataclass(frozen=True)
class RunArrays:
    """One run's datasets, exactly as named under its /runs/<NNNN> group."""

    t_s: np.ndarray
    truth_sigma_BN: np.ndarray
    truth_omega_BN_B: np.ndarray
    truth_r_N: np.ndarray
    truth_v_N: np.ndarray
    est_sigma_BN: np.ndarray
    est_r_N: np.ndarray
    est_v_N: np.ndarray
    imu_gyro_P: np.ndarray
    imu_accel_P: np.ndarray


def _check_run_group(group: h5py.Group, run_name: str) -> tuple[list[str], np.ndarray | None]:
    """Check one /runs/<NNNN> group against the v0 contract. Returns the problems found and
    that run's t_s array (None if it could not be read)."""
    problems: list[str] = []
    t_s: np.ndarray | None = None

    for rel_path, (ndim, last_dim, units) in DATASETS.items():
        if rel_path not in group:
            problems.append(f"run {run_name}: missing dataset '{rel_path}'")
            continue

        dataset = group[rel_path]
        data = dataset[()]

        if dataset.dtype != np.float64:
            problems.append(
                f"run {run_name}: dataset '{rel_path}' has dtype {dataset.dtype}, expected float64"
            )

        shape_ok = data.ndim == ndim and (last_dim is None or data.shape[-1] == last_dim)
        if not shape_ok:
            expected = "(N,)" if last_dim is None else f"(N, {last_dim})"
            problems.append(
                f"run {run_name}: dataset '{rel_path}' has shape {data.shape}, expected {expected}"
            )

        actual_units = dataset.attrs.get("units")
        if actual_units != units:
            problems.append(
                f"run {run_name}: dataset '{rel_path}' has units {actual_units!r}, "
                f"expected {units!r}"
            )

        if np.isnan(data).any():
            problems.append(f"run {run_name}: dataset '{rel_path}' contains NaN")

        if rel_path == "t_s" and shape_ok:
            t_s = data
            not_increasing_from_zero = (
                data.size == 0 or data[0] != 0 or (data.size > 1 and (data[1:] <= data[:-1]).any())
            )
            if not_increasing_from_zero:
                problems.append(f"run {run_name}: t_s is not strictly increasing from 0")

    return problems, t_s


def check_campaign(path: str | Path) -> list[str]:
    """Check one campaign.h5 against the v0 contract. Returns the problems found; an empty list
    means the campaign conforms."""
    path = Path(path)
    if not path.is_file():
        return [f"{path}: file does not exist"]

    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        return [f"{path}: could not be opened as HDF5 ({exc})"]

    with handle as f:
        problems: list[str] = []

        for attr in ROOT_ATTRS:
            if attr not in f.attrs:
                problems.append(f"{path}: missing root attr '{attr}'")

        if "runs" not in f:
            problems.append(f"{path}: missing '/runs' group")
            return problems

        runs_group = f["runs"]
        run_names = sorted(runs_group.keys())

        runs_attr = f.attrs.get("runs")
        if runs_attr is not None:
            runs_attr = int(runs_attr)
            if len(run_names) != runs_attr:
                problems.append(
                    f"{path}: 'runs' attr is {runs_attr}, but {len(run_names)} run "
                    f"group(s) exist on disk: {run_names}"
                )
            expected_names = {f"{i:04d}" for i in range(runs_attr)}
            for missing_name in sorted(expected_names - set(run_names)):
                problems.append(f"{path}: missing run group '{missing_name}'")

        reference_t_s = None
        reference_run = None
        for run_name in run_names:
            group_problems, t_s = _check_run_group(runs_group[run_name], run_name)
            problems.extend(group_problems)
            if t_s is not None:
                if reference_t_s is None:
                    reference_t_s, reference_run = t_s, run_name
                elif not np.array_equal(t_s, reference_t_s):
                    problems.append(
                        f"{path}: run '{run_name}' t_s differs from run '{reference_run}' t_s"
                    )

    return problems


def load_campaign(path: str | Path) -> Iterator[RunArrays]:
    """Yield one RunArrays per run group in campaign.h5, in run order."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        runs_group = f["runs"]
        for run_name in sorted(runs_group.keys()):
            group = runs_group[run_name]
            yield RunArrays(
                t_s=group["t_s"][()],
                truth_sigma_BN=group["truth/sigma_BN"][()],
                truth_omega_BN_B=group["truth/omega_BN_B"][()],
                truth_r_N=group["truth/r_N"][()],
                truth_v_N=group["truth/v_N"][()],
                est_sigma_BN=group["est/sigma_BN"][()],
                est_r_N=group["est/r_N"][()],
                est_v_N=group["est/v_N"][()],
                imu_gyro_P=group["imu/gyro_P"][()],
                imu_accel_P=group["imu/accel_P"][()],
            )
