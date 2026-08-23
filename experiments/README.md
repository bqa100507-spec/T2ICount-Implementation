# Experiment provenance

Keep one `manifest.yaml` per controlled run. The manifest is a small,
human-readable record of what changed, what was intentionally held constant,
where large artifacts live, and which results were observed.

## Recording a run

1. Copy `manifest.template.yaml` into `experiments/<run-name>/manifest.yaml`.
2. Record the exact intervention under `loss` and the controlled setup under
   `training`.
3. State the invariants explicitly. Do not rely on a run name to imply that
   architecture, RRC, optimizer, transforms, or inference stayed fixed.
4. Store checkpoints and full logs on Drive. Keep only the small manifest in
   Git.
5. Fill a provenance field only when it is supported by Git history, notebook
   output, a committed log, repository documentation, or another retained
   record. Never reconstruct a missing SHA, date, dirty state, or artifact name
   from memory. Use `null` (or `unknown` where a string is required) and explain
   the gap in `notes`.

Artifact names inferred from a tracked code convention should be identified as
such in `notes`; they are not a substitute for checking that the file still
exists on Drive.

## Metric names

Use `val_mae`, `val_rmse`, `test_mae`, and `test_rmse` in manifests. The legacy
training output labels the root-mean-square error value as `MSE`; the manifest
uses `RMSE` so the field describes the metric that the code actually computes.

## Scope of the DUMLO records

These manifests describe a DUMLO-based T2ICount integration (the operational
`DUMLO-s10` implementation in this repository). They must not be described as
paper-exact reproduction evidence unless a separate audit establishes that
claim.

