"""v0 recorded-run contract: schema and conformance checks.

See `contract/README.md` for the documented format. Provisional — the format trade becomes an
ADR here when it matters, not at v0.
"""

import json
from pathlib import Path

import pandas as pd

CONTRACT_VERSION = 0

COLUMNS = [
    "t_s",
    "truth_sigma_1",
    "truth_sigma_2",
    "truth_sigma_3",
    "truth_omega_1",
    "truth_omega_2",
    "truth_omega_3",
    "truth_r_1",
    "truth_r_2",
    "truth_r_3",
    "truth_v_1",
    "truth_v_2",
    "truth_v_3",
    "est_sigma_1",
    "est_sigma_2",
    "est_sigma_3",
    "est_r_1",
    "est_r_2",
    "est_r_3",
    "est_v_1",
    "est_v_2",
    "est_v_3",
    "gyro_1",
    "gyro_2",
    "gyro_3",
    "accel_1",
    "accel_2",
    "accel_3",
]


def check_run_file(path: str | Path) -> list[str]:
    """Check one run_<NNNN>.csv against the v0 contract. Returns the problems found; an empty
    list means the file conforms."""
    path = Path(path)

    if not path.is_file():
        return [f"{path}: file does not exist"]

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [f"{path}: could not be parsed as CSV ({exc})"]

    problems: list[str] = []

    actual = df.columns.tolist()
    if actual != COLUMNS:
        missing = [c for c in COLUMNS if c not in actual]
        extra = [c for c in actual if c not in COLUMNS]
        if missing:
            problems.append(f"{path}: missing columns {missing}")
        if extra:
            problems.append(f"{path}: unexpected columns {extra}")
        if not missing and not extra:
            problems.append(
                f"{path}: columns are present but out of order, expected {COLUMNS}, got {actual}"
            )
        return problems

    if df.empty:
        return [f"{path}: has no data rows"]

    t = df["t_s"].to_numpy()
    if t[0] != 0:
        problems.append(f"{path}: t_s does not start at 0 (starts at {t[0]})")
    if (t[1:] <= t[:-1]).any():
        problems.append(f"{path}: t_s is not strictly increasing")

    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        problems.append(f"{path}: NaN values in column(s) {nan_columns}")

    return problems


def check_campaign(campaign_dir: str | Path) -> list[str]:
    """Check a campaign directory's manifest against the files it names, and each named run file
    against the v0 contract. Returns the problems found; an empty list means the campaign
    conforms."""
    campaign_dir = Path(campaign_dir)
    manifest_path = campaign_dir / "manifest.json"

    if not manifest_path.is_file():
        return [f"{manifest_path}: manifest.json does not exist"]

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{manifest_path}: could not be parsed as JSON ({exc})"]

    problems: list[str] = []

    if manifest.get("contract_version") != CONTRACT_VERSION:
        problems.append(
            f"{manifest_path}: contract_version is {manifest.get('contract_version')!r}, "
            f"expected {CONTRACT_VERSION}"
        )

    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        problems.append(f"{manifest_path}: 'runs' is missing or empty")
        return problems

    named_files: set[str] = set()
    for entry in runs:
        run_id = entry.get("run")
        file_name = entry.get("file")
        if not file_name:
            problems.append(f"{manifest_path}: run {run_id} has no 'file' entry")
            continue
        named_files.add(file_name)
        run_path = campaign_dir / file_name
        if not run_path.is_file():
            problems.append(f"{manifest_path}: run {run_id} names missing file {file_name}")
            continue
        problems.extend(check_run_file(run_path))

    on_disk = {p.name for p in campaign_dir.glob("run_*.csv")}
    orphans = sorted(on_disk - named_files)
    if orphans:
        problems.append(
            f"{campaign_dir}: file(s) present but not referenced by any run in "
            f"manifest.json: {orphans}"
        )

    return problems
