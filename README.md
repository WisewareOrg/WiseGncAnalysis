# WiseGncAnalysis

WiseGncAnalysis is the analysis home for Wiseware's GNC work. It reads run files written by a
simulation; its loader, under `analysis/gnc_analysis/runs.py`, is the only statement of what a
run file must contain, and it knows nothing else about the producer.

Flight, simulation and analysis are split into three homes so any one can be replaced without
touching the other two: framework-free flight algorithms live in
[WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign); simulation plants and Monte Carlo
campaigns that import those algorithms and write run files live in
[WiseDynamics](https://github.com/WisewareOrg/WiseDynamics); analysis over those run files lives
here, in WiseGncAnalysis.

## Layout

- `analysis/gnc_analysis/runs.py` — what a run file must contain (see "What a run file must
  contain" below).
- `docs/decisions/` — this repository's ADRs (none yet).

## What a run file must contain

A run is one HDF5 file, `run_<NNNN>.h5` (zero-padded to 4 digits); a campaign is a directory of
them. The loader (`analysis/gnc_analysis/runs.py`) reads these datasets, each `float64` with a
string `units` attribute matching the value shown:

| Dataset | Shape | Units |
| --- | --- | --- |
| `t_s` | `(N,)` | `s` — strictly increasing, starting at 0 |
| `truth/sigma_BN` | `(N, 3)` | `MRP` |
| `truth/omega_BN_B` | `(N, 3)` | `rad/s` |
| `truth/r_N` | `(N, 3)` | `m` |
| `truth/v_N` | `(N, 3)` | `m/s` |
| `est/sigma_BN` | `(N, 3)` | `MRP` |
| `est/r_N` | `(N, 3)` | `m` |
| `est/v_N` | `(N, 3)` | `m/s` |
| `imu/gyro_P` | `(N, 3)` | `rad/s` (sensor frame) |
| `imu/accel_P` | `(N, 3)` | `m/s^2` (sensor frame) |

`load_run` reads each of these by name, checks its `units` attribute, rejects a `NaN` anywhere,
and requires `t_s` to be strictly increasing from 0; `load_campaign` requires every run in a
directory to share the same `t_s`. Extra datasets and attributes are ignored. There is no schema
file and no separate checker — this loader, and the behavior above, is the authority on what a
run file must contain.
