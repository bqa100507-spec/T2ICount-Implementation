## Research Experiment Protocol

1. Before any research experiment or architecture/loss modification, read
   `PROJECT_STATE.md` and the relevant entries in `EXPERIMENT_LEDGER.md`.

2. Before implementation, explicitly define all of the following:

   - Question
   - Hypothesis
   - Changed variable
   - Controls
   - Metrics
   - Decision rule

3. If any of those definitions are missing, stop and ask the user/researcher
   instead of implementing a speculative change.

4. Prefer changing one research variable at a time.

5. Do not silently change any of the following:

   - dataset subset
   - random seeds
   - prompt bank
   - evaluator
   - data split
   - preprocessing
   - optimizer
   - loss definition
   - model architecture
   - augmentation behavior
   - checkpoint selection rule

6. Do not repeat a historical negative experiment unless a new hypothesis
   explicitly requires it.

7. Never delete a failed experiment from `EXPERIMENT_LEDGER.md`; the ledger is
   append-only.

8. If a conclusion becomes invalid, mark it `OBSOLETE` and explain which later
   evidence superseded it.

9. Agents may record raw metrics, configurations, and artifact paths.

10. Agents must not automatically finalize scientific conclusions after
    training.

11. The user/researcher must explicitly review scientific interpretation before
    `PROJECT_STATE.md` or `EXPERIMENT_LEDGER.md` receives a final conclusion.

12. After an experiment is reviewed, update `EXPERIMENT_LEDGER.md` first, then
    update `PROJECT_STATE.md`.

13. Do not use scalar loss alone to declare an alignment experiment better.
    When available, consider:

    - positive distance
    - negative distance
    - separation gap
    - margin violation rate
    - pairwise accuracy
    - retrieval R@1
    - class-aware R@1
    - sample-aware R@1
    - per-prompt behavior

14. Clearly distinguish paper facts, this repository's adaptation,
    experimental observations, hypotheses, and proposed future work.

15. Never describe the current RichCount-inspired prototype as a faithful
    RichCount reproduction unless the implementation actually matches the
    paper.

16. Repository `Stage-A` and `Stage-B` are internal sub-stages of the
    RichCount-inspired visual-text alignment stage:
    Stage-A = visual FFN training;
    Stage-B = text Adapter training.
    Do not confuse them with RichCount paper Stage 1 / Stage 2.

17. The primary objective is improved T2ICount zero-shot counting with short
    class-name inference. Evaluate research changes primarily through class-name
    counting MAE/RMSE on the appropriate fixed datasets and protocols. Detailed
    and generalized rich prompts are primarily auxiliary training information;
    treat rich-prompt inference and alignment metrics as diagnostics unless a
    new research objective is explicitly approved.
