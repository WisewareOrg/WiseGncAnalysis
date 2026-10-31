# Contract

The recorded-run data contract every simulation exporter must satisfy, and the conformance
checks in [`v0.py`](v0.py). **The format is provisional** — the format trade becomes an ADR here
when it matters, not at v0.

## v0

A campaign is a directory holding one `manifest.json` and one `run_<NNNN>.csv` per run.

### Run file — `run_<NNNN>.csv`

A header row followed by one data row per recorded sample, columns in exactly this order:

| Column | Units | Meaning |
| --- | --- | --- |
| `t_s` | s | Sample time, strictly increasing, starting at 0 |
| `truth_sigma_1`, `truth_sigma_2`, `truth_sigma_3` | MRP (dimensionless) | True attitude, body wrt inertial, σ_BN |
| `truth_omega_1`, `truth_omega_2`, `truth_omega_3` | rad/s | True body-frame angular rate |
| `est_sigma_1`, `est_sigma_2`, `est_sigma_3` | MRP (dimensionless) | Estimated attitude, body wrt inertial, σ_BN |
| `est_dv_1`, `est_dv_2`, `est_dv_3` | m/s | Estimated accumulated ΔV, inertial frame |
| `gyro_1`, `gyro_2`, `gyro_3` | rad/s | Recorded gyro output, body frame |
| `accel_1`, `accel_2`, `accel_3` | m/s² | Recorded accelerometer output, body frame |

A column present under the wrong name, a different column order, a non-monotonic or
non-zero-starting `t_s`, or a `NaN` anywhere in the file is a contract violation.

### Manifest — `manifest.json`

```json
{
  "contract_version": 0,
  "producer": "<string>",
  "created_utc": "<ISO 8601>",
  "runs": [
    {
      "run": 0,
      "file": "run_0000.csv",
      "seeds": {"<task or module>": 0},
      "dispersions": {"<dotted.name>": [0.0, 0.0, 0.0]}
    }
  ]
}
```

- `contract_version` — always `0` for this version.
- `producer` — a free-form string identifying what wrote the campaign.
- `created_utc` — an ISO 8601 timestamp.
- `runs` — one entry per run, naming the run's index, its file, the RNG seeds it used (keyed by
  whichever task or module drew them), and the dispersed values it was given (keyed by a dotted
  name, scalar or per-axis).

### Conformance checks

`check_run_file(path) -> list[str]` checks one run file: the file exists and parses as CSV, its
columns match the table above by name and order, it has at least one data row, `t_s` starts at 0
and is strictly increasing, and no column holds a `NaN`. It returns the list of problems found;
an empty list means the file conforms.

`check_campaign(campaign_dir) -> list[str]` checks a whole campaign: `manifest.json` exists and
parses, its `contract_version` is `0`, every run it names has a file in the directory, and every
such file passes `check_run_file`. It returns the combined list of problems; an empty list means
the campaign conforms.
