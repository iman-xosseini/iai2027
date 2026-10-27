# Coding Agent — Stage 4 Implementation Specification

## Project

**IAI 2027: Mask-Aware Temporal World Modeling for Robust Industrial Condition Monitoring**

**Task:** Implement Stage 4 — Historical Telemetry Dropouts and Three-Model Robustness Comparison.

**Primary notebook:** `notebooks/02_world_model.ipynb`

**Deadline:** November 1, 2026, for conference abstract submission.

## 1. Objective

Extend the completed Stage 3 implementation to test the following hypothesis:

**Explicit availability masks improve the robustness of temporal latent-state prediction under intermittent historical sensor dropouts, compared with corruption augmentation alone.**

Implement a rigorous comparison between three models:

- Model A: Standard temporal LeWorldModel, trained on clean data.
- Model B: Corruption-augmented temporal LeWorldModel, without informative availability masks.
- Model C: Mask-aware temporal LeWorldModel, with informative availability masks.

The primary comparison is Model B versus Model C.

Use clean and artificially corrupted Azure Predictive Maintenance telemetry.

Do not redesign the encoder, predictor, SIGReg objective, or temporal window construction.

## 2. Inspect the existing project first

Before modifying the code:

1. Inspect the completed Stage 1 data pipeline.
2. Inspect Stage 2 encoder and predictor classes.
3. Inspect the Stage 3 training and checkpointing code.
4. Load and verify the three Model A checkpoints.
5. Inspect `configs/base.yaml`.
6. Identify which existing functions should be reused.

The current Stage 3 notebook imports:

- `Encoder`
- `TemporalPredictor`
- `TemporalWorldModel`
- `SIGReg`
- `build_model`
- `build_sigreg`
- `fit`
- `world_model_loss`
- `encode_windows`
- `latent_statistics`
- `load_checkpoint`

Reuse these components whenever possible.

Do not duplicate existing working training code.

Preserve all Stage 1, 2 and 3 outputs.

## 3. Preserve the original architecture

The architecture remains:

**Encoder:** 48 → 64 → 64 → 32

**Predictor:** 128 → 128 → 64 → 32

**Historical context:** `[B, 4, 6, 4]`

**Target:** `[B, 6, 4]`

**Historical embeddings:** `[B, 4, 32]`

**Predicted embedding:** `[B, 32]`

**Actual target embedding:** `[B, 32]`

All three models must have identical architectures and parameter counts.

Preserve shared-encoder training without EMA teachers or stop-gradients.

Preserve the existing SIGReg implementation, projection configuration, and loss reduction.

## 4. Implement a reusable historical corruption module

Implement three corruption strategies.

### 4.1 Independent dropout

Generate a Bernoulli availability mask for historical sensor measurements.

Required severity levels:

- 0%
- 10%
- 20%
- 40%

At 0%, all measurements remain available.

At other rates, individual historical measurements are independently selected for removal.

### 4.2 Consecutive packet-loss bursts

Select a sensor and remove a consecutive sequence of historical observations.

Use a small number of predefined durations, such as 6 and 12 hours.

### 4.3 Temporary sensor outage

Remove one sensor for one or two consecutive 6-hour historical blocks.

Keep the dropout generator independent from the neural network.

### Corruption invariants

The module must guarantee:

- Historical tensor shapes remain unchanged.
- Only historical context measurements can be corrupted.
- Removed standardized measurements are replaced with zero.
- Original uncorrupted sensor values are never overwritten.
- Target values remain exactly unchanged.
- Target masks remain all ones.
- Masks contain only zeros and ones.
- Random seeds reproduce the same corruption.
- All models are evaluated using identical corrupted samples.

Provide unit tests.

For independent dropout, verify that the observed dropout proportions are approximately equal to the requested probabilities on sufficiently large batches.

For burst and outage evaluation, pay attention to overlapping temporal windows. Prefer a consistent machine–timestamp corruption schedule where appropriate, so the same sensor-hour is not arbitrarily described as available in one window and unavailable in an adjacent window. Explicitly document that keeping every prediction target clean is a controlled evaluation assumption.

## 5. Implement the three-model input policy

Use the following rules.

### Model A

- Load existing Stage 3 checkpoint.
- Do not retrain.
- Training data originally clean.
- For corrupted evaluation, replace selected historical values with zero.
- Pass all-ones mask channels.

### Model B

- Initialize a new model using the existing architecture.
- Train on clean and corrupted historical contexts.
- Replace missing values with zero.
- Pass all-ones mask channels even at missing positions.

### Model C

- Initialize a new model using the existing architecture.
- Train on exactly the same clean and corrupted contexts as Model B.
- Replace missing values with zero.
- Pass actual availability masks.

For Models B and C, initialize from identical random weights for each paired seed.

Verify experimentally that B and C receive identical value tensors before the encoder and that their only intended input difference is the availability-mask information.

## 6. Training protocol for Models B and C

Reuse the Stage 3 optimizer and training objective:

```
total_loss = prediction_loss + 0.1 * SIGReg
```

Use the same initial training configuration:

- Adam optimizer.
- Learning rate: 0.001.
- Batch size: 128.
- Maximum epochs: 30.
- Seeds: 11, 22, 33.
- Stage 3 SIGReg implementation and hyperparameters.

Use a mixed training distribution:

- 50% clean context windows.
- 50% corrupted context windows.

For corrupted windows, sample the dropout severity from the predefined training rates: 10%, 20%, or 40%.

Use independent dropout as the primary training corruption.

The burst and temporary-outage patterns should initially be held-out stress tests, allowing assessment of generalization to missingness patterns not explicitly used during training.

### Fairness requirements

For Models B and C:

- Use the same initial weights.
- Use the same minibatch order.
- Use the same corruption decisions.
- Use the same corruption severity distribution.
- Use the same target values.
- Use the same optimizer settings and update budgets.
- Use the same SIGReg configuration.
- Use the same validation and checkpoint-selection protocol.

Use a separate, reproducible random-number stream for corruption generation, independent of model initialization and SIGReg projections.

Do not rely on Python's process-randomized built-in `hash()` for reproducible sample-based seeds.

## 7. Validation and checkpointing

Reuse Stage 1 validation windows.

Prepare two validation components:

1. Clean validation contexts.
2. A fixed validation corruption set generated using predefined seeds.

Use a predefined mixture of clean and corrupted validation total losses for B/C checkpoint selection, with equal weighting as an initial choice.

Maintain the existing noncollapse criteria:

- Minimum effective rank.
- Minimum latent standard deviation.
- Finite losses and gradients.

Use identical criteria for Models B and C.

Model A retains its already selected Stage 3 checkpoint.

Save Model B and Model C separately for each seed.

Example checkpoint names:

- `model_b_best_seed11.pt`
- `model_b_best_seed22.pt`
- `model_b_best_seed33.pt`
- `model_c_best_seed11.pt`
- `model_c_best_seed22.pt`
- `model_c_best_seed33.pt`

Save corruption configuration, random seeds, training settings, and validation metrics inside each checkpoint.

Verify that saved checkpoints reload correctly.

Do not use the test partition for model selection.

## 8. Evaluation of clean and corrupted validation windows

Evaluate all three models using identical validation samples and evaluation corruption masks.

Required scenarios:

- Clean, 0% dropout.
- Independent dropout, 10%.
- Independent dropout, 20%.
- Independent dropout, 40%.
- Six-hour dropout burst.
- Twelve-hour dropout burst.
- One-sensor outage.

Calculate:

1. Clean prediction MSE.
2. Corrupted prediction MSE.
3. Relative MSE degradation.
4. Per-window prediction-error change.
5. Latent surprise-score changes.

Use the same target embedding for a clean/corrupted pair within each model.

Do not compare absolute MSE across separately trained latent spaces without normalization or calibration.

For each model, report the relative degradation compared with that model's own clean MSE.

Calculate mean and standard deviation over the three seeds.

For scientific uncertainty, distinguish variation over random seeds from variation over machines or overlapping windows.

## 9. Surprise-score and alert analysis

For each model, define surprise as Euclidean distance between predicted and actual target embeddings.

Use clean validation data to establish a model-specific alert threshold.

Initially select the 95th percentile of the clean-validation surprise distribution.

Hold that threshold fixed when assessing corrupted validation windows.

Report:

- Percentage of corrupted windows exceeding the clean-calibrated threshold.
- Change in alert rate between clean and corrupted contexts.
- Alert-rate variation by dropout severity and dropout pattern.

This is a controlled robustness experiment.

Do not claim that every alert is a machine failure or that every window without recorded failure is guaranteed normal.

For failure-negative proxy analysis, define a fixed exclusion rule around failure and maintenance events. The choice of horizon and exclusion interval must be recorded before evaluation.

Do not tune thresholds or horizons using test results.

## 10. Additional temporal-information checks

Perform two lightweight diagnostics on the trained models using validation data:

**A. Historical order perturbation**

Reorder the four historical blocks without changing the target.

Observe whether prediction MSE increases.

**B. Same-machine mismatched history**

Replace the context with a different valid context from the same machine, selected sufficiently far away in time and without overlapping the target window.

Observe whether prediction MSE increases.

Keep these tests distinct from the dropout-robustness experiment.

Do not retrain models for these diagnostics.

Record the limitations: changes in prediction error under perturbation are diagnostic evidence of information use, not proof that the latent model has learned physical causal dynamics.

## 11. Required figures and tables

Generate the following.

### Figure A — Relative prediction-error degradation

X-axis: Dropout severity.

Y-axis: Prediction MSE divided by clean MSE.

Separate curves for A, B and C.

Show variation across seeds.

### Figure B — Alert-rate increase

X-axis: Dropout severity.

Y-axis: Fraction of windows exceeding each model's fixed clean-validation threshold.

Separate curves for A, B and C.

### Figure C — Corruption-pattern comparison

Compare random dropout, consecutive packet loss, and temporary sensor outages.

### Table A — Main three-model comparison

Include:

- Clean validation MSE.
- MSE at 10%, 20%, and 40% dropout.
- Relative degradation.
- Clean-calibrated alert-rate change.
- Mean and standard deviation across seeds.

### Table B — Training diagnostics

Include:

- Selected epoch.
- Validation objective.
- SIGReg loss.
- Effective rank.
- Minimum latent standard deviation.
- Parameter count.

Prepare figures and tables suitable for the conference abstract and future paper.

## 12. Output files

Continue in:

```
notebooks/02_world_model.ipynb
```

Add Section 8:

- 8.1 Historical dropout generator
- 8.2 Dropout validation tests
- 8.3 Model comparison protocol
- 8.4 Training Models B and C
- 8.5 Validation and checkpointing
- 8.6 Robustness evaluation
- 8.7 Surprise-score and alert analysis
- 8.8 Temporal-information checks
- 8.9 Figures, tables and Stage 4 validation

Implement reusable code under:

- `src/data/` for corruption utilities.
- `src/training/` for training adaptations.
- `src/evaluation/` for robustness metrics if a new module is useful.

Suggested generated files:

- `results/stage4_training_metrics.csv`
- `results/stage4_robustness_metrics.csv`
- `results/stage4_summary.json`
- `figures/stage4_robustness_curves.png`
- `figures/stage4_alert_rates.png`
- `figures/stage4_corruption_patterns.png`

Preserve existing Stage 3 checkpoints and results.

## 13. Acceptance tests

Stage 4 is complete only when:

- Model A reloads and retains its recorded behavior.
- Corruption affects historical context only.
- Corruption generation is reproducible.
- Models B and C receive exactly paired training corruptions.
- Models A, B and C have identical architectures.
- Models B and C have been trained successfully.
- Their representations satisfy the predefined noncollapse criteria.
- All three models are evaluated on identical validation corruption scenarios.
- Model-specific clean baselines and relative degradation are reported.
- Alert thresholds are calibrated without using test data.
- The trained checkpoints can be reloaded.
- Comparison plots and results tables have been generated.
- All limitations, unexpected findings and failed checks are reported transparently.

Do not force Model C to outperform B. A negative or mixed result is scientifically valid and must be reported.

## 14. Scope restrictions

Do not:

- Add a GRU or Transformer.
- Modify the Stage 3 SIGReg implementation.
- Change sensor-block size or prediction horizon.
- Train using machine failure labels.
- Add an LLM or explainability module.
- Introduce additional model architectures.
- Tune results against the held-out test partition.
- Implement the complete final failure-detection study from Stage 5.

The objective is a controlled, reproducible, lightweight robustness comparison that directly tests the mask-awareness hypothesis.

## 15. Scientific references

**LeWorldModel — Maes et al. (2026)**

https://arxiv.org/html/2603.19312v2

Read Section 3.1 (Model Architecture and Training Objective), Section 4.3 (Ablations), and Section 5.2 (Violation-of-expectation Framework).

**GRU-D — Che et al. (2018)**

https://arxiv.org/html/1606.01865v2

Read Section 2 (Mask Representation) and Section 3.2 (Methods and Implementation Details).

**Kim et al. (2022), Towards a Rigorous Evaluation of Time-Series Anomaly Detection**

https://ojs.aaai.org/index.php/AAAI/article/view/20680

Read the evaluation methodology and discussion of inflated anomaly-detection performance.

The purpose is to understand the scientific foundations of the experiment, not to reproduce those papers' complete models.

**Deliverable:** A fully tested Stage 4 implementation, trained Models B and C, and validation results that can be independently reviewed before proceeding to Stage 5.