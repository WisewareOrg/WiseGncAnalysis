# Contract

The recorded-run data contract every simulation exporter must satisfy, and the conformance
check in [`v0.py`](v0.py). **The layout is provisional** — the format trade becomes an ADR here
when it matters, not at v0.

## v0

A campaign is exactly one HDF5 file, `campaign.h5`. No CSV, no manifest, no sidecar files —
everything the contract describes lives inside this one file (owner ruling, 2026-10-31).

```
campaign.h5
  attrs (root): contract_version=0 (int), producer (str), created_utc (ISO 8601 str), seed (int),
                runs (int), versions_bsk, versions_gnc_algorithms, versions_wise_dynamics (str)
  /runs/<NNNN>/            one group per case, zero-padded 4
     attrs: case (int), seed (int), and one attr per drawn quantity named draw_<name> (float array)
     datasets, float64, with a string attr `units`:
       t_s             (N,)    "s"        strictly increasing, t_s[0] == 0
       truth/sigma_BN  (N,3)   "MRP"
       truth/omega_BN_B(N,3)   "rad/s"
       truth/r_N       (N,3)   "m"
       truth/v_N       (N,3)   "m/s"
       est/sigma_BN    (N,3)   "MRP"
       est/r_N         (N,3)   "m"
       est/v_N         (N,3)   "m/s"
       imu/gyro_P      (N,3)   "rad/s"    sensor frame
       imu/accel_P     (N,3)   "m/s^2"    sensor frame
```

### Root attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `contract_version` | int | Always `0` for this version |
| `producer` | str | Free-form string identifying what wrote the campaign |
| `created_utc` | str | ISO 8601 timestamp |
| `seed` | int | The campaign-level seed the producer drew everything else from |
| `runs` | int | The number of `/runs/<NNNN>` groups; must equal the number present |
| `versions_bsk` | str | Basilisk version the producer ran |
| `versions_gnc_algorithms` | str | `gnc-algorithms` version/revision the producer ran |
| `versions_wise_dynamics` | str | `wise-dynamics` version/revision that produced the file |

### Run groups — `/runs/<NNNN>`

One group per case, named by its zero-padded 4-digit case index (`0000`, `0001`, ...). Each
group's own attributes carry `case` (int) and `seed` (int), plus one `draw_<name>` attribute
(a float array) per dispersed quantity the producer drew for that case.

Each group holds the ten datasets in the layout above, all `float64`, each carrying a string
`units` attribute with the value shown. `t_s` must start at 0 and be strictly increasing; every
run in a campaign shares the same `t_s`. Extra datasets and attributes beyond these are allowed —
a study may add its own channels — but nothing documented above may be missing, misshapen,
mistyped, or NaN.

### Conformance check

`check_campaign(path) -> list[str]` checks a campaign.h5 file and returns the problems found; an
empty list means the file conforms. It checks: every root attribute above is present; the `runs`
attribute equals the number of `/runs/<NNNN>` groups actually present, and every group that
number implies exists; every dataset in the layout above is present in every run group, with the
right shape and dtype and the right `units` attribute; no dataset holds a `NaN`; every run's
`t_s` starts at 0 and is strictly increasing; and every run's `t_s` matches every other run's.

`load_campaign(path)` yields one `RunArrays` per run group, in run order, with the ten datasets
above as plain numpy arrays.
