# WiseGncAnalysis

WiseGncAnalysis is the analysis home for Wiseware's GNC work. It reads recorded simulation runs
through a data contract it owns, under `contract/`, and knows nothing about which simulation
produced them — only the contract's shape.

Flight, simulation and analysis are split into three homes so any one can be replaced without
touching the other two: framework-free flight algorithms live in
[WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign); simulation plants and Monte Carlo
campaigns that import those algorithms and export to the data contract live in
[WiseDynamics](https://github.com/WisewareOrg/WiseDynamics); analysis against that contract lives
here, in WiseGncAnalysis.

## Layout

- `contract/` — the recorded-run data contract every simulation exporter must satisfy (see its
  own `README.md`).
- `docs/decisions/` — this repository's ADRs (none yet).
