# Bulk service queueing for transit resilience

Code and numerical data for the paper

> Baichuan Mo, Zhengzhong Ricky You, Li Jin, Zuo-Jun Max Shen, Haris N. Koutsopoulos and Jinhua Zhao.
> *Bulk Service Queueing for Transit Resilience under Short Random Service Suspensions.*

The repository contains the physical route simulator, the joint queue–phase calculation, the direct
two-station joint calculation, the iid root and contour benchmarks, every experiment driver, the
replication-level outputs behind each table and figure, and 35 regression tests. All experiments use
synthetic routes, so no external data are needed. A standard CPU suffices; no GPU or commercial solver
is used.

## Setup

Use Python 3.13 in a fresh virtual environment and run every command from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The stored runs used NumPy 2.5.4, SciPy 1.18.1 and one BLAS thread per numerical process
(`runs/rr-runtime-versions.json`).

## Regression tests

```bash
python -m unittest discover -s . -p 'test_*.py'
```

The 35 tests cover FIFO service conservation, calendar-cohort matching, destination inputs, phase
moments, sparse transitions, root safeguards, the Lambert-W special case and a dense enumeration of
the two-station joint operator.

## Reproducing the tables and figures

Each command below reads the stored outputs in `runs/` and writes to the directory it names. Section,
table and figure numbers refer to the paper.

| Paper item | Command |
|---|---|
| Table 1 | inputs defined in `revision_inputs.py` |
| Figure 1 | `python plot_revision_model.py` (writes `Figures/rr2_model.pdf`) |
| Sections 5.2 and 5.6–5.8, Figure 2, Table 7 | `python summarize_revision_evidence.py --route-run runs/rr-final-common-cohort-20261101-v1 --root-run runs/rr-rootfree-final-20261101-v3 --output runs/reproduced-summary` |
| Tables 2–5, Figures 3, 5 and 6 | `python analyze_integrated_revision.py --output runs/reproduced-integrated-summary` |
| Figure 4 | `python plot_service_revision.py runs/rr-service-boundary-20261031 --output runs/reproduced-service` |
| Table 6 | `python summarize_two_stop_validation.py --output runs/reproduced-two-stop` |

`analyze_integrated_revision.py` recomputes all 171 station–profile errors for the four moments,
including the variance errors above 10%, every input-control comparison, the input profiles and the
policy statistics. It also verifies the SHA-256 digests of all 401 frozen calibration and kernel files,
checks irreducibility and strict drift for every fitted kernel and reports its mean-interval error.
`analysis/integrated/` stores its outputs for the paper, together with the two-station summaries. The iid benchmark of Section 5.9 and
Section EC.9 is stored in `runs/rr-rootfree-final-20261101-v3` and
`runs/rr2-deterministic-roots-20261101-v1`.

## Rerunning the experiments

Every driver refuses to overwrite an existing output directory. The large experiments used 16–24
parallel CPU processes; the 4,800,000-vehicle two-station validation took about ten minutes with
eight processes on a laptop.

| Experiment | Command |
|---|---|
| Calibration and independent validation: 19 profiles, 39 physical inputs, 20 replications each (Sections 5.3 and 5.5) | `python run_revision_scan.py --output runs/reproduced-expanded --workers 24` |
| Small pilot of the same scan | `python run_revision_scan.py --output runs/reproduced-pilot --workers 4 --pilot` |
| Operating sample: 41 settings, 20 replications each (Sections 5.2, 5.6 and 5.7) | `python run_revision_final_suite.py --output runs/reproduced-physical` |
| Passenger-dwell reconstruction and all-station diagnostics (Section 5.6) | `python run_revision_dwell_joint.py --previous runs/rr-final-common-cohort-20261101-v1 --direct runs/rr2-two-stop-direct-20261101-v1 --output runs/reproduced-dwell-joint --workers 24` |
| Direct two-station calculation (Section 5.4) | `python joint_two_stop.py --output runs/reproduced-two-stop-direct` |
| Two-station physical validation, 4,800,000 vehicles (Table 6) | `python run_revision_dwell_joint.py --previous runs/rr-final-common-cohort-20261101-v1 --direct runs/rr2-two-stop-direct-20261101-v1 --output runs/reproduced-two-stop-4800k --skip-dwell --toy-vehicles 4800000 --workers 8` |
| Fixed-capacity service diagnostics (Figure 4) | `python validate_service_revision.py --output runs/reproduced-service-boundary` |
| Iid root and contour benchmark | `python benchmark_rootfree.py --output runs/reproduced-roots --workers 1 --root-limit 30` |
| Deterministic full-capacity roots | `python benchmark_deterministic_bulk.py --output runs/reproduced-deterministic-roots` |

Timings depend on hardware; numerical results do not.

## Repository layout

Core modules:

- `service_dynamics.py`: propagated FIFO vehicle service clock.
- `route_validation.py`: physical route simulation that tracks passenger destinations, vehicle
  identities, dwell and fleet returns.
- `delay_stationary.py`, `sparse_phase_queue.py`: joint stationary calculation of the residual queue,
  displacement and available capacity.
- `joint_two_stop.py`: direct joint calculation for the two-boarding-station route.
- `rootfree_bulk.py`, `root_solver.py`: contour and root calculations for the iid bulk-service
  benchmark.
- `revision_inputs.py`: route inputs and validation profiles.

The remaining scripts are experiment drivers (`run_*.py`, `validate_*.py`, `benchmark_*.py`,
`refine_route_phase.py`, `evaluate_frozen_route_phase.py`, `repair_revision_scan.py`) and analysis or
plotting scripts (`analyze_*.py`, `summarize_*.py`, `plot_*.py`).

Stored runs in `runs/`:

| Directory | Contents |
|---|---|
| `rr2-scan-20261101-v1` | Validation sample. Frozen calibration inputs (400,000 vehicles per profile, first 100,000 discarded), kernels and predictions for all 171 station–profile cells, and 20 independent replications of each of the 39 physical inputs. Observations use calendar gates (192000, 384000] and the arrival cohort (192000, 288000] minutes. |
| `rr2-scan-repair-20261101-v1` | The 20 validation outputs of profile `original-demand1`, recomputed with identical frozen inputs and seeds. In the original run a near-duplicate policy input with the same file name overwrote them; the analysis labels that policy input `original-policy-demand1` and never pools the two. |
| `rr-final-common-cohort-20261101-v1` | Operating sample. 41 settings × 20 replications of 80,000 scheduled vehicles: stability boundary, passenger dwell, finite fleet, incident duration laws, destination changes and passing. |
| `rr2-dwell-joint-20261101-v1` | Reconstruction of the 280 dwell cases from the same primitive arrays, with all-station gate, departure, blocking and boarding diagnostics; also the two-station validation at 120,000 vehicles. |
| `rr2-two-stop-direct-20261101-v1` | Direct two-station joint calculation with queue caps 32 to 256. |
| `rr2-two-stop-long-20261101-v1`, `rr2-two-stop-4800k-20261101-v1` | Two-station physical validation at 480,000 and 4,800,000 vehicles, 20 passenger streams per input with seeds 2026112100–2026112119. |
| `rr2-deterministic-roots-20261101-v1` | Lambert-W and contour comparison for deterministic headways and full capacity, 16 settings. |
| `rr-rootfree-final-20261101-v3` | Iid benchmark, 32 settings with three timing repetitions each. |
| `rr-service-boundary-20261031` | Fixed-capacity service diagnostics behind Figure 4. |
| `rr-FINAL-summary-20261101` | Stored output of `summarize_revision_evidence.py`. |
| Other `rr-*` directories | Earlier calibration, refinement, frozen-evaluation and fixed-capacity runs on the operating sample, kept for provenance and described in `README-previous-cohort.md`. |

The validation sample and the operating sample are independent and are never pooled; neither are runs
that share seeds but differ in horizon.

## Representative trajectories

The `traces/` folders of `rr2-scan-20261101-v1`, `rr2-scan-repair-20261101-v1` and
`rr-final-common-cohort-20261101-v1` hold complete vehicle- and passenger-level trajectories for
selected replications (374 MB). No command above needs them. They are attached to the `v1.0` release as
`representative-trajectories.zip`; extract it in the repository root to restore the complete package.

## Data integrity

- `MANIFEST.json` lists the size and SHA-256 digest of every package file, including the
  trajectories in the release archive.
- Frozen calibration and kernel files are hashed before any evaluation.
- Each run directory keeps its protocol, random seeds, completion status and a snapshot of the source
  that produced it. The top-level code is the current implementation.

## Quantities

Q is the queue observed just before a vehicle's boarding gate, R is the residual queue after admission
and W is the time from a passenger's arrival to boarding admission. Calendar-time and service-epoch
moments differ. With positive dwell, gate intervals and departure intervals differ. Finite-horizon
waits in overloaded cases are not stationary moments and are excluded from stationary summaries.

## License

The code and data are released under the MIT License; see `LICENSE`.

## Citation

If you use this code, please cite the paper above.
