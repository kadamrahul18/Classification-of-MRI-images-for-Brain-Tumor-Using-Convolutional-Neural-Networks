# Release Checklist (v1.0.0)

1) Run baseline:
   ```bash
   bash scripts/run_baseline_3d.sh
   ```
2) Summarize + figures:
   ```bash
   python scripts/summarize_run.py --run-dir outputs/runs/<run_id>
   python scripts/make_readme_figures.py --run-dir outputs/runs/<run_id>
   ```
3) Verify artifacts:
   - `outputs/baseline_v1/<timestamp>/` contains config, metrics, logs, and checkpoint.
   - `docs/assets/baseline_examples.png` and `docs/assets/baseline_curves.png` updated.
4) Update README table with values from `outputs/baseline_metrics.json`.
5) Tag and release:
   ```bash
   git tag -a v1.0.0 -m "v1.0.0 baseline"
   git push origin v1.0.0
   ```
6) Create GitHub Release notes (copy from CHANGELOG).
