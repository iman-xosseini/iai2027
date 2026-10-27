# IAI 2027 — Stage 5 Coding Agent Plan

**Study:** Mask-Aware Temporal World Modeling for Robust Industrial Condition Monitoring  
**Stage:** 5 — Final Held-Out Evaluation, Failure-Event Analysis, and Abstract-Ready Evidence  
**Prepared:** 27 October 2026  
**Conference abstract deadline:** 1 November 2026  
**Status:** Instructions for implementation, **not** a claim that Stage 5 has already been executed.

> **Agent's central instruction:** Freeze the models, configurations, evaluation rules, and checkpoint-selection decisions made in Stages 1–4. Run a credible final evaluation on the previously reserved chronological **test** partition. Do not retrain, tune on test results, change thresholds after inspection, or attempt to make Model C outperform Model B. Document negative and mixed findings faithfully.

---

## 0. Scientific context and what Stage 4 actually established

We trained a shared, 32-dimensional latent encoder and a temporal predictor on the **Microsoft Azure Predictive Maintenance** benchmark (simulated industrial telemetry). Four preceding 6-hour blocks (24 hours) are used to predict the representation of the next 6-hour block. Historical inputs contain voltage, rotation, pressure, and vibration. The learning objective is **next-latent prediction MSE + 0.1 × SIGReg**.

All three models have the same architecture (**36,224 parameters**):

| Model | Training inputs | Historical availability channel |
|---|---|---|
| **A — Standard** | Clean data only | All ones, including at deliberately removed test readings |
| **B — Augmented** | Clean + zero-filled corrupted contexts | All ones, including at missing readings |
| **C — Mask-aware** | The **same** clean/corrupted contexts as B | Correct binary availability mask |

Models B and C were trained with identical architectures, seeds, minibatch/corruption decisions, and objectives; B versus C is the principal test of the *incremental* value of explicit availability indicators **on top of zero-fill corruption augmentation**.

### Key Stage 4 validation findings (do not describe these as test results)

At independent dropout levels of 10%, 20%, and 40%, the relative growth in prediction MSE was approximately:

| Model | 10% missing | 20% missing | 40% missing |
|---|---:|---:|---:|
| A | 1.05% | 2.51% | 7.21% |
| B | **0.31%** | **0.97%** | 3.21% |
| C | 0.53% | 1.21% | **3.06%** |

At 40% dropout, clean-calibrated **window alert-rate** increases were approximately **+2.51, +0.54, +0.89 percentage points** for A, B, C respectively. Model C **did not consistently outperform** Model B. Model C also showed more variable representations: selected validation effective rank ~**25.17, 23.60, and 16.66** for seeds 11, 22, and 33 (seed 33 checkpoint from epoch **3**). All three passed the prespecified Stage 4 noncollapse criteria, but they are not equally well regularized.

These facts set the honest interpretation: **corruption augmentation has promising robustness benefits; the added value of an explicit binary mask is small or inconsistent in the tested setup**. Do not silently change this framing because of a favored hypothesis.

---

## 1. Scope, boundaries, and authoritative inputs

### Required existing project inputs

- `configs/base.yaml` — shared model architecture, optimizer, SIGReg, seeds, diagnostics.
- `configs/stage4.yaml` — frozen corruption patterns, severities, model mask policies, and proxy-normal rules.
- `notebooks/04_world_model_stage_2_3_4.ipynb` (or the current equivalent) — executed implementation and Stage 4 methodology.
- `src/data/azure.py`, `src/data/corruption.py` — loaders, paired corruption, `WindowTimeline`, schedules, model input policies.
- `src/models/` — the existing shared encoder and temporal predictor.
- `src/training/trainer.py`, `src/training/robust.py`, `src/training/losses.py` — model/checkpoint loading and training definitions. **Do not retrain.**
- `src/evaluation/robustness.py` — reuse `predict`, `window_scores`, `paired_metrics`, `alert_threshold`, `alert_rate`, `proxy_normal_windows`, and machine-bootstrap helpers.
- `data/processed/` — **test** windows and `window_metadata.csv`, together with existing **validation** windows for threshold calibration.
- Original Azure `PdM_failures.csv`, `PdM_maint.csv` (only for evaluation metadata and failure-event analyses).
- **Nine frozen checkpoints:** `model_{a,b,c}_best_seed{11,22,33}.pt`.
- `results/stage4_summary.json`, `results/stage4_robustness_metrics.csv`, and Stage 4 tables for cross-checking definitions and trends.

### Verified data sizes reported by Stage 1

| Partition | Windows | Machines | Recorded failure rows | Machines with recorded failures |
|---|---:|---:|---:|---:|
| Train | 101,800 | 100 | 555 | 98 |
| Validation | 21,500 | 100 | 98 | 69 |
| **Test** | **21,500** | **100** | **108** | **70** |

**Failure rows are not necessarily independent failure episodes.** Deduplicate machine–timestamp records and state the number of distinct events before reporting event-recall denominators.

### Strict restrictions

1. **No retraining or replacing checkpoints.** Retain all seeds 11, 22, and 33, including Model C seed 33. Any diagnostic excluding a seed must be labeled supplementary, not substituted for the main analysis.
2. **No access to test model performance before locking the Stage 5 protocol.** Create and save `configs/stage5.yaml` first; use its frozen contents throughout.
3. **No test-set-based choice of threshold, horizon, corruption severity, scenario, model, seed, or favorable figure.**
4. **No direct claims from raw MSE comparisons across models** because their encoders learn different latent spaces. Make paired *within-model* comparisons and use calibrated outcomes for between-model interpretation.
5. **No leakage of failure/maintenance labels into the encoder/predictor.** Such data are used only for retrospective evaluation and eligibility rules.
6. Future *target* measurements stay clean, exactly as in Stage 4. This is a **controlled dropout study**; it is not a realistic guarantee of performance when the target itself is unavailable.
7. Stage 1 included some failure-centered exploratory plots across the full dataset; avoid claiming the test period was *never viewed in any exploratory capacity*. What is essential now is that no Stage 5 model or evaluation rule is chosen from final test outcomes.
8. No point adjustment, automatic credit for every timestamp in a failure event, or other scoring procedure that inflates detection metrics.

---

## 2. Organization and workflow

**Preferred implementation:** a separate `notebooks/05_final_evaluation.ipynb`, so running or inspecting the final test does not accidentally execute Stage 4 retraining cells. Reuse modules instead of copying large notebook cells.

Suggested additions:

```text
configs/stage5.yaml
notebooks/05_final_evaluation.ipynb
src/evaluation/final.py                 # optional reusable Stage 5 logic
src/evaluation/failure_events.py        # optional event matching and lead time
src/evaluation/__init__.py              # export new functions if useful
tests/test_stage5_protocol.py
results/stage5/
figures/stage5/
```

**Ordered sections in the new notebook:**

- **9.1** Freeze protocol and verify test isolation.
- **9.2** Load all nine checkpoints, calibration data, and test data.
- **9.3** Recreate fixed paired test dropout schedules.
- **9.4** Final test prediction-robustness comparison.
- **9.5** Clean-calibrated surprise thresholds and alert stability.
- **9.6** Machine-level uncertainty and seed consistency.
- **9.7** Failure-event detection and optional early-warning lead times.
- **9.8** Sensitivity, negative controls, and limitations.
- **9.9** Tables, plots, audit results, and abstract evidence ledger.

---

## 3. Step 5.1 — Pre-register/freeze the Stage 5 protocol **before evaluating test**

Create `configs/stage5.yaml` containing at minimum:

| Field | Fixed setting |
|---|---|
| Checkpoint seeds | `[11, 22, 33]` for A, B, C |
| Test partition | `test` exclusively for final metrics |
| Calibration partition | `val` exclusively for thresholds |
| Independent dropout rates | `0.0`, `0.1`, `0.2`, `0.4` |
| Held-out patterns | `burst_6h`, `burst_12h`, `outage`, with **unchanged Stage 4 definitions** |
| Test corruption random seed | Fixed in advance (suggest **5201**), independent of validation seed 4201 |
| Model input masks | A/B all ones; C actual historical availability |
| Target handling | Always clean, target mask all ones |
| Surprise | `||z_pred - z_target||_2` |
| Alarm threshold | Each **model–seed** clean-validation 95th percentile |
| Main comparisons | A vs B, A vs C, **B vs C primary** |
| Primary outcomes | Relative MSE degradation; excess alert fraction at dropout rates |
| Confidence interval | Machine-level paired bootstrap, 2,000 replicates; fixed seed |
| Uncertainty reporting | Paired-seed results plus machine-level interval, not a single pooled uncertainty number |
| Pre-failure horizon | **24 hours** primary *exploratory* horizon, fixed before test |
| Optional event sensitivity | **48 hours**, explicitly secondary and fixed before test |
| Alarm clustering | Consecutive positive 6-hour predictions grouped; 24-hour cooldown for new alarm episodes (if used) |

Write this file to disk and save its SHA-256 hash plus a timestamp in `results/stage5/protocol_manifest.json` **before loading test predictions**. If the agent discovers a genuine implementation bug later, record the correction, date, and whether any test results had already been inspected. Do not quietly overwrite the protocol.

**Deliverable:** Frozen protocol and manifest.

**Acceptance:** A second researcher can identify exactly what was planned before final testing.

---

## 4. Step 5.2 — Checkpoint and dataset integrity

Load A/B/C for all three seeds using the existing checkpoint loader. Verify:

- Exactly **9** checkpoints, correct model key and seed, unique paths, load without exceptions.
- Checkpoint architecture remains 36,224 parameters, latent dimension 32, history length 4, prediction horizon 1.
- Saved encoder and predictor weights are finite.
- Selected epochs agree with Stage 4 summary (A: 16/20/21; B: 26/21/24; C: 26/22/**3**).
- Input-mask policy matches the model identity; no accidental use of true masks for A/B.
- Stage 5 code calls `model.eval()` and performs **no optimizer step**.
- Test tensors reproduce Stage 1 dimensions; 21,500 windows × 100 machines expected, unless a clearly documented dataset/version difference exists.
- Every window has consistent machine ID, 24-hour historical context, a directly following 6-hour target, chronological timestamps, and no crossing of the dataset partition boundary.
- Scaler is the **training-fitted** scaler from Stage 1; do not fit a new scaler on test data.
- No test labels occur in feature inputs.
- Checkpoints are associated with the correct saved configuration and available dataset version; retain SHA-256 hashes if possible.

**Optional but recommended:** For each model–seed, recompute one small sample of *clean-validation* forward outputs and compare with checkpoint-reload diagnostics from Stage 4 to detect an accidental implementation change.

**Deliverable:** `results/stage5/integrity_checks.json` and `checkpoint_manifest.csv`.

**Acceptance:** No mismatched checkpoint, shape, split, or preprocessing provenance.

---

## 5. Step 5.3 — Reuse paired dropout generation on the test timeline

Use Stage 4's existing `WindowTimeline`, `scenario_schedule`, and `window_masks` utilities.

**Scenarios (must not be altered in response to test results):**

| Scenario | Meaning | Training exposure |
|---|---|---|
| `clean` | 0% loss | A, B, C |
| `independent_10` | Approximately 10% randomly missing sensor-hour readings | B/C augmented |
| `independent_20` | Approximately 20% randomly missing readings | B/C augmented |
| `independent_40` | Approximately 40% randomly missing readings | B/C augmented |
| `burst_6h` | One sensor loses 6 consecutive hours per 30-hour cycle | **Held-out pattern** |
| `burst_12h` | One sensor loses 12 consecutive hours per 30-hour cycle | **Held-out pattern** |
| `outage` | One sensor absent for 1–2 aligned 6-hour blocks per 30-hour cycle | **Held-out pattern** |

Generate **one scenario availability schedule per test scenario**, independently of the model; reuse it for all models and all seeds. Because historical windows overlap, the same machine/hour/sensor must have consistent availability across its appearances in different windows.

For each scenario, create three model inputs from the **same** original context and actual availability:

- A: zero-filled missing readings + all-ones historical mask.
- B: **identical** zero-filled readings + all-ones historical mask.
- C: **identical** zero-filled readings + true binary mask.
- All: unchanged next six-hour target and all-ones target mask.

Record exact removal share, fraction of windows affected, scenario seed, and deterministic checksum of schedule. At 0% dropout, corrupt and clean inputs must be bit-for-bit identical.

**Deliverable:** `results/stage5/corruption_manifest.json` with checksums and measured rates.

**Acceptance:** For each scenario, A/B/C observe the same *sensor values*; the only intended policy difference is C's extra availability information.

---

## 6. Step 5.4 — Calculate the primary final test outcomes

For every **model × seed × scenario**, compute predictions and targets on the same test windows.

Per-window quantities:

- Prediction MSE: `mean_j[(z_pred[j] - z_target[j])²]` across 32 dimensions.
- Surprise: `sqrt(sum_j[(z_pred[j] - z_target[j])²])`.
- Clean/corrupted paired MSE difference and affected-window flag.

The target embedding remains identical between clean and corrupted input evaluations **within a given model/seed**.

Primary robustness metric:

\[
D_{m,s,p}=\frac{\operatorname{MSE}_{m,s,p}}{\operatorname{MSE}_{m,s,0}}-1.
\]

Here `m` is model, `s` is seed and `p` is corruption scenario. A lower `D` indicates less loss of predictive fidelity under corruption. Use consistent **ratios of mean per-window errors**, not silently interchangeable means of per-window ratios.

**Primary between-model effect:** paired difference `D_B - D_C` at each prespecified independent dropout level. **Positive** favors C; negative favors B. Also report `D_A - D_B` and `D_A - D_C` to quantify benefits of corruption augmentation against the clean-trained reference.

Additional reporting:

- Relative MSE for **affected windows only**, where appropriate.
- Mean and median paired error change and fraction of affected windows whose error increased.
- Clean prediction MSE **plus** latent target variance/normalized MSE for diagnostic context; do **not** rank models using raw MSE alone.
- Results for held-out burst/outage patterns reported separately from independent dropout.

**Deliverables:**

- `results/stage5/test_robustness_per_seed.csv`
- `results/stage5/test_robustness_aggregate.csv`
- `results/stage5/per_window_scores.parquet` (or compressed CSV if Parquet is unavailable)

Include model, seed, scenario, `machine_id`, `target_end`, test sample ID, dropout checksum, and score columns so metrics can be independently recomputed. Preserve the paired clean/corrupted identity.

**Acceptance:** All nine model–seed combinations evaluated on the exact same set of test windows/scenarios; independent recalculation from per-window records matches summary within numerical tolerance.

---

## 7. Step 5.5 — Clean-calibrated alert stability

Calibration is **only on clean validation data**; do not re-estimate thresholds on the test partition.

For each model and seed, recover or recompute a fixed clean-validation threshold:

\[
T_{m,s}=Q_{0.95}\{\operatorname{surprise}^{\text{clean,val}}_{m,s,i}\}.
\]

Apply the **same** `T_{m,s}` to clean and corrupted **test** windows for that model–seed. Do not recalibrate separately for 10%, 20%, 40%, bursts, or outage. Do not use failure labels to fit this percentile.

Calculate:

1. Clean-test alert rate (it **need not equal 5%**; it was calibrated on *validation*).
2. Alert rate under each corruption scenario.
3. **Excess alert fraction** = corrupted-test alert rate − clean-test alert rate, in **percentage points**.
4. Alert rate restricted to a prespecified **failure-negative proxy subset** using the exact Stage 4 rule:
   - No machine failure within `[context_start − 48 h, target_end + 48 h]`.
   - No maintenance within `[context_start − 24 h, target_end]`.
5. Number of additional alert **windows** per machine-day, explicitly labeled as a window-based rate rather than distinct incidents.

**Important distinction:** The percentage above threshold on every window is an *alert fraction*. It is **not necessarily a confirmed false-alarm rate**, and a threshold calibrated on all validation windows does not guarantee 5% on clean test or on the proxy-normal subset.

If computing distinct alarm incidents, group consecutive flagged windows and use a **fixed** 24-hour cooldown. Report that rule separately; do not mix window alerts and distinct alarms.

**Deliverables:** `test_alert_metrics.csv`, `test_thresholds.csv`, and a per-machine alert summary.

**Acceptance:** Thresholds are copied from/recomputed only on validation; fixed-threshold test results are reproducible; the denominator and units are explicit.

---

## 8. Step 5.6 — Statistical uncertainty and robustness of conclusions

The 21,500 test windows overlap and are **not independent**. The 100 machines constitute the natural primary resampling units. Also, three seeds provide only limited information about training stochasticity.

Implement:

- **Paired machine-level bootstrap**, 2,000 replicates with a fixed seed. Resample machines, retaining all their windows, and recompute **ratios of weighted sums** within each draw; calculate B−C and A−B/A−C using aligned resamples.
- Per-seed B−C differences and direction of effect: e.g. "C better in 1 of 3 seeds".
- Mean and sample standard deviation across the three seeds, **separately** from bootstrap intervals.
- Where feasible, per-machine distribution/heterogeneity of benefit, without claiming 100 independent training runs.
- Confidence intervals for a small set of **predeclared primary contrasts**, particularly 40% independent dropout and the trajectory across 10/20/40; make secondary patterns explicitly exploratory. Avoid uncorrected significance claims across numerous post hoc comparisons.

**Do not pool all 21,500 windows as if they were independent** when calculating a tiny standard error. Do not present a machine-only bootstrap interval as accounting for uncertainty across new training seeds, new datasets, or operating sites.

If the **test** B−C effect is negligible, inconsistent, or contrary to the hypothesis, record that result as the primary conclusion. Negative results are acceptable.

**Deliverables:** `test_paired_differences.csv`, `test_machine_bootstrap_intervals.csv`, and `test_seed_consistency.csv`.

**Acceptance:** Reported intervals can be traced to machine-level resampling and are interpreted with the proper limitations.

---

## 9. Step 5.7 — Failure-event and advance-warning analysis (**exploratory; do not confuse it with the main Stage 4 hypothesis**)

This is the most interpretation-sensitive part of the project. Finish the primary robustness evaluation first. Only run this analysis if the needed event metadata and time budget are available.

### 9.1 Define what an alarm timestamp means

For a window with historical blocks ending at `context_end` and a future target block ending at `target_end`, the score compares `z_pred` with the **actual observed target representation**.

**Therefore surprise becomes available only after `target_end`.** It is not available at the start of that future six-hour block. No lead-time calculation may use `context_end` or `target_start` as the alarm time.

If a recorded failure occurs at time `f`, define a positive **prospective warning lead time** only when:

\[
0 < f-\texttt{target\_end} \le 24\ \mathrm{hours}
\]

for the **primary exploratory** 24-hour warning horizon. A 48-hour horizon, if computed, is supplementary and must have been frozen in Stage 5 YAML **before** checking test labels/results.

### 9.2 Prepare events and denominators

- Filter failure metadata to the **test chronological partition** for the final failure-event table.
- Combine records with the same `(machine_id, failure_timestamp)` into **one event**, with a list of failed components; separately report counts of rows and distinct machine–timestamp events.
- If multiple events occur close together on one machine, apply one fixed episode-grouping rule (e.g. 24-hour separation) that is recorded before evaluation; otherwise explicitly keep them separate but explain any overlapping warning windows.
- Exclude failure events for which the **entire required pre-event monitoring window** is not observable inside the test partition or flag them as partially observable. Do not count an unobservable event as missed silently.
- Maintenance and error-event records may be used for sensitivity/exclusion analyses, never as inputs to the self-supervised model.

### 9.3 Fixed-threshold detection metrics

Compute on **clean test contexts first**, then optionally re-run the exact metric under each corruption scenario:

- **Event recall:** number of eligible failure events with ≥1 preceding alarm in `(f−24 h, f)` divided by number of eligible events. Use only scores available at `target_end < f`.
- **Lead time:** for detected events, report the lead of the **earliest eligible warning** in the fixed horizon; state the six-hour sampling resolution and median/interquartile range.
- **Alarm burden:** distinct alarm incidents per eligible monitored machine-day using a fixed grouping/cooldown rule.
- **Window alert rate** and failure-negative proxy alert rate, for comparison with Stage 4.
- **Precision of distinct alarm incidents**, only if a defensible event-to-alarm matching rule and exposure denominator exist; otherwise omit rather than fabricate.

Optional: event recall at a fixed *alarm burden* established on validation, but do not tune the burden on test.

A single long alert period must **not** receive automatic credit for every point or every subsequent failure. State the one-to-one or earliest-warning event-matching policy clearly. Distinct failure components at the same timestamp must not multiply the apparent event recall.

### 9.4 Baselines and caveats

- Use the clean validation-calibrated thresholds; do not optimize failure detection for any model on test labels.
- Compare A/B/C as monitoring systems at their **calibrated thresholds**, not by raw latent distances.
- If practical, include a simple random/constant-rate alert control to show what event recall could occur by chance, without fitting anything on test.
- A lack of predictive failure association is a valid finding. Do not claim early failure prediction from lower latent MSE or stable alert fractions alone.
- If the failures are too sparse or results vary across seeds, classify the entire failure component as *exploratory* and omit numerical early-warning claims from the abstract.

**Deliverables:** `test_failure_events.csv`, `test_failure_event_metrics.csv`, `test_event_matching_audit.csv`; optional illustrative time traces showing historical signal, score availability time, failure timestamps, and fixed alarms.

**Acceptance:** Every reported lead time is computed from `target_end`, every counted event has sufficient monitoring coverage, and duplicate failures/alerts cannot inflate recall.

---

## 10. Step 5.8 — Short diagnostic/validity checks (no retraining)

These are quick analyses to qualify the results, not excuses to optimize after viewing test performance.

1. **Implicit missingness cue:** On **train and/or validation** normalized raw observed sensor readings, calculate how frequently a genuine reading is *exactly zero*. If exact zero is rare, zero-filled Model B may already have an implicit missingness cue; report this as a limitation of claiming masks are unhelpful in general.
2. **Model C diagnostics:** Preserve the reported variability in C's selected epochs and effective ranks, especially seed 33. Optionally examine how conclusions differ using only seeds 11/22, but **retain all three seeds for the primary results**.
3. **Clean performance versus robust performance:** Confirm that reduced alert sensitivity under corruption has not merely become reduced responsiveness to failure-related deviations. Interpret any failure metrics together with alarm burden.
4. **Domain limitations:** The Azure dataset is simulated; there is no verification on physical industrial machines. Corruptions are synthetically imposed only on **historical context**; target availability is assumed complete.
5. **Temporal baseline:** Retain the Stage 3/4 evidence that predicted next embeddings respond to historical ordering and mismatched history. Do not repeat lengthy baseline experiments or turn them into a new hyperparameter search.
6. **Data-leakage audit:** Confirm chronology, training-only normalization, no test labels in training/thresholds, and the Stage 1 cross-partition exploration caveat.

**Deliverable:** `results/stage5/limitations_and_sensitivity.md`.

---

## 11. Step 5.9 — Mandatory outputs for scientific review and abstract writing

### Main publication-ready figures

| Figure | Content | Interpretation guardrail |
|---|---|---|
| **F1 — Dropout robustness** | Test relative MSE degradation at 0/10/20/40%, A/B/C, across seeds | No cross-model raw MSE ranking |
| **F2 — Excess alerts** | Corruption-induced change in clean-calibrated test alert rate | Distinguish window-alert fraction and false alarms |
| **F3 — Stress patterns** | Burst 6h, burst 12h, and outage B−C/A−B comparison | Report actual removed shares because scenarios have different severity |
| **F4 — Failure warning (optional)** | Per-event recall versus incident burden or representative alarm/failure traces | Alarm availability timestamp is `target_end` |

### Main tables

- **T1:** A/B/C final test performance across independent dropout rates; three-seed mean ± SD, paired effects, machine-bootstrap intervals.
- **T2:** Alert rates, excess alerts, and proxy-normal subset metrics on test.
- **T3:** Selected checkpoint summary, effective rank and variance diagnostics carried from Stage 4; do not substitute new checkpoints.
- **T4 (optional):** Eligible failure-event counts, event recall, false-alert burden, warning lead times.

### Evidence ledger for abstract

Create `results/stage5/abstract_claims.md`, with one row per possible claim:

| Proposed claim | Quantitative evidence | Partition | Can we state it? | Required limitation |
|---|---|---|---|---|
| Corruption augmentation improves robustness | A–B paired degradation and alert differences | Test | Yes, **only if test confirms** | Synthetic missingness |
| Explicit masks outperform augmentation | B–C paired results, seed consistency | Test | Only if supported; currently **not supported by validation** | Zero-fill may imply missingness |
| Model predicts failures early | Fixed-threshold event recall/lead and alarm burden | Test | Only if adequate positive evidence | Score available after target block |
| SIGReg prevents collapse | Latent diagnostics across checkpoints | Validation + optional test diagnostics | Describe as noncollapsed representations, not proven physical semantics | C's lower effective rank |

Never manufacture figures or narrative to produce a positive result. Where the evidence is negative or mixed, state the mixed result explicitly.

**Final deliverables:** `results/stage5/summary.json`, `results/stage5/abstract_claims.md`, test CSVs, figures, and a concise written **Stage 5 conclusion** with clear support/limits.

---

## 12. Implementation tests and acceptance gate

Add automated tests in `tests/test_stage5_protocol.py` (or the project's established test structure):

| Test | Required expected behavior |
|---|---|
| Test-set isolation | No test tensors/labels are used for training, checkpoint selection, calibration, or metric-definition changes |
| Frozen settings | Evaluator reads fixed `stage5.yaml`, and writes its hash |
| Checkpoint loading | Exactly 9 correct model–seed checkpoints, no optimizer activity |
| Clean scenario | Context values unchanged and all-ones availability |
| Paired inputs | Zero-filled context values identical for A/B/C for the same test sample/scenario |
| Mask policy | A/B all-ones; C actual availability |
| Target preservation | Every target and target mask identical to original Stage 1 test dataset |
| Temporal schedule consistency | Overlapping windows agree on availability for the same machine/hour/sensor |
| Reproducible corruption | Same frozen seed yields same checksum and masks |
| Threshold integrity | Only validation is used for thresholds; different model–seed thresholds do not get mixed |
| Within-model MSE denominator | Relative degradation uses same model/seed clean MSE |
| Bootstrap clustering | Repeated/overlapping windows are resampled as whole machines |
| Event grouping | Same-machine, same-time multi-component records count once per event |
| Lead-time causality | A warning is counted only after its target ended and strictly before failure |
| Boundary eligibility | Events without enough preceding test coverage identified, not silently called misses |
| Baseline/units | Alarm windows and distinct alarm incidents never confused |
| No post hoc selection | All models, all seeds, all scenarios reported; failures/negative results not filtered |

### Stage 5 *go/no-go* decision

**Complete and proceed to abstract drafting if:**

- The frozen protocol and all integrity tests pass.
- Final test dropout comparison for A/B/C is computed and independently reproducible.
- Clean-validation thresholds are used unchanged on clean/corrupted test data.
- Paired B–C effects, seed variability and machine-level uncertainty are reported.
- Any implemented failure-event claims satisfy timestamp and denominator audits.
- The evidence ledger identifies exactly what can and cannot be claimed.

**Fix narrowly and repeat the affected calculation if:** missing masks are routed incorrectly, target samples are corrupted, checkpoint identities are mixed, thresholds are accidentally fit on test, timing/episode matching is wrong, or summary files cannot be reproduced from the per-window scores. Log the correction and any test exposure.

**Do not block completion merely because:** Model C loses to B, results are statistically inconclusive, predictive failure association is weak, or the untrained failure-related component is infeasible before the deadline. Such outcomes change **conclusions**, not whether a valid evaluation is complete.

---

## 13. Deadline-aware order of execution

Today is **27 October 2026**. The conference abstract deadline is **1 November 2026**. Keep the core final-test pipeline ahead of optional event work.

| Priority | Work | Target |
|---|---|---|
| **P0** | Freeze Stage 5 YAML; verify all nine checkpoints, data and metadata | First |
| **P0** | Final test dropout robustness and clean-calibrated excess alerts | Next |
| **P0** | B–C seed/paired uncertainty, main tables and plots, abstract evidence ledger | Next |
| **P1** | Failure-event counts, 24-hour recall, timestamp-valid lead times, alarm burden | If feasible |
| **P1** | Zero-fill implicit-cue diagnostic and C seed-variability commentary | Lightweight |
| **P2** | More detailed 48-hour sensitivity, further illustrations, other optional baselines | After abstract if necessary |

The required core outcome is a **rigorous final test of dropout robustness**. The failure-warning analysis is scientifically interesting but is not a prerequisite for a truthful robustness-focused abstract.

---

## 14. Reading linked to implementation

Use the existing `IAI2027_Complete_Scientific_Reading_Guide.md` as the reading baseline. Stage 5 requires especially:

1. **Kim et al.**, *Towards a Rigorous Evaluation of Time-series Anomaly Detection* (AAAI 2022): **§2.3**, **§3.2–3.3** and **§4.2**. Avoid inflated anomaly metrics / overly permissive point adjustment. https://arxiv.org/html/2109.05257
2. **Sørbø and Ruocco**, *Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series*: **§5.1–5.2**, especially point-wise versus event-wise metrics. https://doi.org/10.1007/s10618-023-00988-8
3. **LeWorldModel**: **§5.2 (Violation-of-Expectation Framework)** for surprise-based analysis, while recognizing that our actual future target must be observed first. https://arxiv.org/html/2603.19312v2
4. **scikit-learn, Common Pitfalls**, **§12.2–12.2.1** for information leakage and train/validation/test separation. https://scikit-learn.org/stable/common_pitfalls.html

No additional architecture or training literature is required before the frozen evaluation.

---

## 15. Handoff package for independent scientific review

When finished, provide the following (combine into ZIP if attachment limits apply):

**Essential:**

- `notebooks/05_final_evaluation.ipynb` **with executed outputs**.
- `configs/stage5.yaml` and `results/stage5/protocol_manifest.json`.
- New or changed `src/evaluation/` and `tests/` code, plus existing source version used.
- `results/stage5/summary.json`, test robustness CSV, alert CSV, paired-differences/CI CSV.
- `results/stage5/abstract_claims.md` and Stage 5 conclusion.

**For fuller independent reproduction:**

- Relevant A/B/C checkpoints (at least one complete paired seed if all nine cannot be attached at once).
- Fixed test-corruption scenario manifest/checksums.
- Per-window score archive and figures.
- Failure-event tables, episode-matching audit and original event metadata if a failure-warning claim is made.

### Final scientific question

> **Does the held-out test period confirm that corruption augmentation reduces degradation and dropout-induced alerts, and does an explicit availability mask add a consistent, practically meaningful benefit beyond augmentation?**

Answer that question **before** deciding the conference abstract's principal claim. Early-failure results are a secondary, separately substantiated contribution—not an assumption or automatic consequence of next-latent prediction.
