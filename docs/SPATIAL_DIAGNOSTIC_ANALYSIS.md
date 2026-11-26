# DIAG-002 - CPU post-hoc spatial analysis

Entry points: `tools/analyze_spatial_diagnostic.py` and
`notebooks/spatial_diagnostic_analysis.ipynb`. This analyzes completed DIAG-001
artifacts; it never imports T2ICount/torch, initializes a model, opens a checkpoint,
runs inference/training, generates prompts, or changes source artifacts. Nothing
updates PROJECT_STATE.md or EXPERIMENT_LEDGER.md. No automatic commit/push.

## Research protocol

- **Questions:** Is RP-009's FSC147-S gain consistent with reduced overcounting
  and/or density redistribution? Are HSCM differences more associated with counting
  gain than attention changes? What changes with detailed prompts within a model?
- **Hypotheses:** H1a magnitude/calibration and H1b redistribution can both occur.
  HSCM/attention associations are descriptive hypotheses, not causal attribution.
- **Changed variable:** No new inference condition. Only approved post-hoc
  transformations, paired comparisons and statistical adjustment are added.
- **Controls:** Exact existing 230 image IDs, BASE-002/RP-003/RP-009, class/detailed
  text, saved predictions/maps, original 20 qualitative images and patch geometry.
- **Metrics:** Counting errors/bias/distributions; paired gains and image bootstrap;
  saved per-map normalized L1/Pearson; Pearson/Spearman and transparent residualized
  associations; nonnegative unit-mass density shape comparisons on the saved 20.
- **Decision rule:** Report evidence, associations, plausible interpretations,
  alternatives and uncertainty for researcher review. Do not automatically select
  an architecture or generalize the FSC147-S ranking to standard FSC147 or elsewhere.

Class-name inference is primary. Detailed prompting is secondary. No verified
FSC147-S target dot annotations exist, so no localization-accuracy metrics or
pseudo ground truth are introduced.

## Inputs and validation

Default input: `/content/drive/MyDrive/T2ICount-assets/outputs/spatial_diagnostic_v2_all230/`.
Required: manifest.json, summary.json, results.csv, map_comparisons.csv, and exactly
120 native NPZs (original 20 x three checkpoints x two prompt types).

Validation requires 230 unique images, 1,380 exact counting keys and 14,490 exact
map keys, six designated-image capture checks, compatible schema/preprocessing,
saved prompt/class/GT consistency, and agreement with source aggregate metrics.
Missing, duplicate, nonfinite or incompatible records fail explicitly; no rows
are silently repaired. Only empty Pearson CSV fields represent undefined r.

Joins use `image/model/prompt_type` and exact left/right condition identities.
CSV row order is irrelevant. Each NPZ is opened with `allow_pickle=False` and
checked against the actual DIAG-001 schema:

- tokens_12/24/48: patches x active BPE positions x native height x width;
- attention_12/24/48: patches x 1 x native height x width;
- fused_attention, sim_x2, sim_x1, density_native: patches x 1 x 48 x 48;
- density_full: 1 x 1 x original image H x W;
- patch_xywh: original row-major shifted-edge patch coordinates.

Content-token means, coordinates and full-density sum versus saved predicted
count are checked (sum tolerance rtol=2e-6, atol=1e-5 allows float32 reduction).
No alternative patch interpolation/reassembly is performed. Similarity maps are
the existing decoder cosine outputs, not newly computed semantic similarities.

Qualitative context requires the original 20 RGBs matching manifest SHA256s.
Use `--image-root` for a local copy. The notebook finds existing original paths,
an existing asset directory, or extracts **only those 20 JPEGs** from the existing
asset ZIP into private local scratch. It never extracts/loads checkpoint weights.

## Statistical definitions

- Signed error = prediction - GT. Over/under counts use tolerance 1e-9. GT strata
  are exactly 1, 2-5, 6-20 and >20; n is always reported.
- Paired checkpoint gain = reference absolute error - candidate absolute error.
  Positive means candidate better. Wins/losses/ties are paired; model_winners.csv
  separately reports sole-best/shared-best/not-best among all three models.
- Absolute error = positive overcount error + positive undercount error. Their
  paired mean reductions decompose the MAE gain arithmetically, not causally.
- The 95% image-bootstrap percentile interval resamples paired gains, 2,000 times
  with seed 3407. It does not estimate training-seed/checkpoint variability,
  independence of semantically related images, or unrelated-dataset performance.
- Existing normalized L1 retains DIAG-001's per-native-patch minmax (eps=1e-7)
  and concatenated patch-cell convention. Overlapping cells remain separate patch
  observations. This is not raw magnitude, localization accuracy or segmentation.
- Spatial difference features are normalized L1 and 1-spatial-Pearson. Pearson
  and Spearman associate them with counting gain; within-checkpoint prompt pairs
  use delta_pred and delta_abs_error. Better/worse/tie distributions are separate.
- Partial Pearson regresses x and y separately on an intercept, log1p(GT) and
  log1p(reference absolute error), then correlates residuals. Partial Spearman
  first average-ranks x/y/controls and applies the same OLS residualization.
  Rank and valid n are recorded; constant/insufficient cases return NA explicitly.
  Raw associations also appear per GT stratum. Residualization cannot eliminate
  all confounding or the mathematical coupling of reference error with gain.
- No p-value screening or automatic best-map selection. Many correlated
  exploratory comparisons are shown; isolated large coefficients are not proof.

## Density magnitude versus shape (original 20 only)

Raw magnitude uses saved `density_full` in count/pixel units, with total mass
shown alongside predicted count. Color limits are shared across all 20 class
images and three models. A fixed square-root display norm improves visibility;
the colorbar remains in raw units and no values are changed. Negative values,
if present, are retained using a shared linear diverging scale instead.

P = D/sum(D) is valid only for finite, nonnegative D with mass > 1e-12. Epsilon
is a validity threshold, not an additive mass or clipping operation. Zero/near-zero
or negative maps are reported explicitly and excluded from probability comparisons.
Normalized-shape color limits are shared across the three models of each image.

Shape metrics: total variation = 0.5 sum(abs(P-Q)), and pixelwise shape Pearson.
TV=0 means identical probability distributions; scalar multiplication cannot
change P. Shape differences establish redistribution, not improved target localization.
There is no raw-map reconstruction for the other 210 images.

Full RGB/density/P panels use the saved full-image coordinates. Attention/HSCM
panels show each native 48x48 map at its explicit original patch xywh. Patch maps
are not independently resized and presented as a reconstructed full image.

## Running and rerunning

The notebook runs completely on a CPU kernel. It embeds the identical reviewed
analysis script as a hash-verified fallback, so it works before DIAG-002 is
committed/pushed and needs no source ZIP, Git checkout, CLIP or torch. It prefers
an identical local helper when available. Set INPUT_DIR/IMAGE_ROOT for a local
kernel; Colab defaults read the existing Drive artifacts.

```powershell
& C:\Users\MSl\miniconda3\envs\t2itest\python.exe tools/analyze_spatial_diagnostic.py --input-dir outputs/diag002_inputs/spatial_diagnostic_v2_all230 --image-root D:/T2ICount-assets/datasets/FSC147/images_384_VarV2
```

Default output is `INPUT_DIR/analysis_cpu_v1/`. Existing directories are refused
unless `--reuse` explicitly validates complete status plus source/config/input
and every output hash. Reuse writes nothing. A partial, modified, or incompatible
analysis is preserved: choose a new directory. No overwrite/delete/resume option.
`--validate-only` validates tables/raw maps/RGBs without creating output files.

Dependencies are CPU numerical/plotting packages: numpy, pandas, scipy,
matplotlib, Pillow. Notebook setup only installs missing CPU packages.

## Outputs and interpretation

- analysis_manifest.json: exact input/code/output fingerprints, CPU execution
  boundary, base checkout commit, environment and immutability verification.
- input_validation.json and density_normalization.csv: source consistency and
  every invalid/valid normalization status.
- counting_summary.csv, paired_counting.csv, paired_counting_summary.csv,
  model_winners.csv and prompt_sensitivity.csv: magnitudes, bias, bin sample counts,
  paired gains, bootstrap intervals, wins/losses/ties and within-model changes.
- spatial_joined.csv, spatial_associations.csv, spatial_groups.csv and
  spatial_extremes.csv: exact-key joins, raw/partial correlations, distribution
  summaries and fixed extreme-selection rules (top three L1 / lowest Pearson).
- density_shape_pairs.csv: 20 x nine paired conditions, raw mass and normalized shape.
- figures/: 13 aggregate plots, 20 full-density/shape panels and 40 native-patch
  panels for this dataset (73 figures). figure_qa.json records render sizes,
  clipping/blank checks and hashes; plots are also visually reviewed.
- summary.json and research_report.md: four findings organized into evidence,
  association, plausible interpretation, alternatives and uncertainty, ending
  with a researcher decision table that selects no architecture automatically.

All four input files and all 120 NPZ hashes are checked before/after analysis;
matching RGBs are checked too. Original Drive artifacts are only read. Downloaded
copies, generated analysis, contact sheets and temporary build scripts live under
ignored outputs/, never Git. Source files are uncommitted until researcher review.
