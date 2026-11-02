# Earlier operating-sample runs

This note documents runs from the operating-sample analysis that preceded the expanded validation in
`runs/rr2-scan-20261101-v1`. They are kept for provenance. Their single-reference accuracy results
and the earlier count of 23 tests are superseded by `analyze_integrated_revision.py` and the
35-test suite described in `README.md`.

The numerical runs used Windows with Python 3.13.12, NumPy 2.5.4, SciPy 1.18.1 and one BLAS thread
per numerical process; plotting used Python 3.13 on macOS.

## Runs

- `runs/rr-final-common-cohort-20261101-v1`: operating sample, 41 settings × 20 independent
  replications (seeds 2026104200–2026104219), 80,000 scheduled vehicles each.
- `runs/rr-frozen-phase-final-20261101-v1`: evaluation of the frozen phase model on those seeds,
  with paired replay and control outcomes and all four moment errors.
- `runs/rr-route-phase-refinement-20261101-v1`: frozen joint displacement–capacity phase,
  transition matrix, numerical refinement and conditional stationary law.
- `runs/rr-rootfree-final-20261101-v3`: 32 prescribed-capacity iid benchmark settings, three timing
  repetitions each. Timings and time-limit outcomes depend on hardware.
- `runs/rr-FINAL-summary-20261101` and `runs/rr-FINAL-dependence-20261101`: derived statistics,
  paired intervals and joint-dependence checks.
- `runs/rr-delay-independent-20261031` and its validation and boundary companions: a separate
  fixed-available-capacity experiment with its own seeds and protocol.
- `runs/rr-route-phase-20261101-v1`: independent upstream calibration and an earlier
  displacement-only model. Its evaluation with seeds 2026103600–2026103619 is a development run
  and is never pooled with the operating sample; the frozen displacement-only prediction is compared with
  the operating sample as an ablation.

Final waiting cohorts are the same local calendar arrivals in (0.5Lh, 0.75Lh] within each demand
setting and replication, followed until final service; every cohort member completed and no waits
were imputed. Queue measures use the last half of the stated vehicle horizon. Only the explicit
finite-fleet variants simulate incident-affected empty returns and vehicle reuse; the open-route
`returned` placeholders are unused and are not reported as fleet cycle measurements.

## Verification and analysis

From the repository root:

```bash
python -m unittest test_service_dynamics test_service_validation test_delay_stationary test_route_validation test_rootfree_bulk test_root_verification test_sparse_phase_queue test_reproduction_paths
python summarize_revision_evidence.py --route-run runs/rr-final-common-cohort-20261101-v1 --root-run runs/rr-rootfree-final-20261101-v3 --output runs/reproduced-summary
python analyze_joint_dependence.py --route-run runs/rr-final-common-cohort-20261101-v1 --phase-run runs/rr-route-phase-refinement-20261101-v1 --output runs/reproduced-dependence
python evaluate_frozen_route_phase.py --route-run runs/rr-final-common-cohort-20261101-v1 --phase-run runs/rr-route-phase-refinement-20261101-v1 --edge-phase-run runs/rr-route-phase-20261101-v1 --output runs/reproduced-frozen-evaluation
```

Frozen kernel hashes are checked. Path separators in Windows manifests are normalized for POSIX
verification; this does not alter simulations or predictions.

To rerun the complete operating sample (24 CPU processes; allow adequate memory):

```bash
python run_revision_final_suite.py --output runs/reproduced-physical
```

This uses the prescribed cases and random seeds and the supplied frozen kernel hashes; the
independent donor traces are regenerated before evaluation.

To recompute the calibration, grid and cutoff checks:

```bash
python refine_route_phase.py --calibration runs/rr-route-phase-20261101-v1/calibration_inputs.npz --output runs/reproduced-refinement
```

Reproductions of the reported evaluation should use the original frozen kernels. A refitted kernel
defines a new numerical artifact and should be compared with the original arrays before use.

To rerun the iid benchmark:

```bash
python benchmark_rootfree.py --output runs/reproduced-roots --workers 1 --root-limit 30
```

Successful numerical times exclude interpreter startup for both methods, while the root time limit
includes it. The stored run used 16 parallel cases on a 384-logical-processor server. These results
are finite-budget numerical evidence, not exact certificates or hardware-independent completion
guarantees.

## Notes

All data are simulated. The phase is an empirical first-order closure and generally not an exact
Markov projection of the full route. It is independent of the local Poisson stream by construction,
and the residual queue and phase are solved jointly. Calibration uses simulated upstream service and
load, never local queue or waiting targets. Physical route stability rests on the exact cut-load
theorem; a capped numerical queue is never used as evidence of stability in an overloaded case.

Host and process metadata were removed from the run records. Numeric results, arrays, seeds, model
settings and source snapshots are unchanged.
