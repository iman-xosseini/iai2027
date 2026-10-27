# Coding Agent — Stage 3 Implementation Specification

## Project

**IAI 2027: Mask-Aware Temporal World Modeling for Robust Industrial Condition Monitoring**

**Task:** Implement Stage 3 — SIGReg and End-to-End Training.

**Primary notebook:** `notebooks/02_world_model.ipynb`

**Existing implementation:** Stage 1 Azure data pipeline and Stage 2 shared encoder/predictor have been completed and tested.

## 1. Objective

Extend the existing Stage 2 model with a self-supervised training pipeline. Train the encoder and temporal predictor jointly on clean Azure Predictive Maintenance telemetry.

The result must be a trained and reproducible **Model A (Standard Temporal LeWorldModel)** that will serve as the reference model in Stage 4.

Do not implement artificial telemetry corruption, Models B/C, or final failure-detection evaluation in this stage.

## 2. Existing code and architecture

Inspect the existing project before making changes.

The Stage 2 notebook imports:

- `src.models.Encoder`
- `src.models.TemporalPredictor`
- `src.models.TemporalWorldModel`
- `src.models.mask_conditioned_input`
- `src.data.load_azure_dataset`

Locate and inspect the actual implementations, particularly `src/models/encoder.py`, `predictor.py`, and `world_model.py`.

Retain the current architecture:

| Component             | Architecture                                      |
| --------------------- | ------------------------------------------------- |
| Encoder               | 48 → 64 → 64 → 32, with GELU hidden activations   |
| Predictor             | 128 → 128 → 64 → 32, with GELU hidden activations |
| Context               | `[B, 4, 6, 4]`                                    |
| Context mask          | `[B, 4, 6, 4]`                                    |
| Target                | `[B, 6, 4]`                                       |
| Target mask           | `[B, 6, 4]`                                       |
| Historical embeddings | `[B, 4, 32]`                                      |
| Predicted embedding   | `[B, 32]`                                         |
| Target embedding      | `[B, 32]`                                         |

There must be one shared trainable encoder.

Do not redesign the neural networks, introduce a separate target encoder, or change the data windowing.

## 3. Implement the prediction loss

Define mean-squared error between `z_pred` and `z_target`.

Use the mean reduction over the batch and all 32 latent dimensions.

Requirements:

- Output a scalar differentiable tensor.
- Verify input/output shapes.
- Confirm finite numerical values.
- Verify that the encoder and predictor receive gradients.
- Do not detach `z_target`.

Log the unweighted prediction loss separately.

## 4. Reuse and validate SIGReg

Locate the working SIGReg implementation from the previous CIRP study if it is available in the project.

Reuse the actual Epps–Pulley / empirical-characteristic-function formulation, not a replacement based merely on mean and variance matching.

Verify:

1. Gaussian random projection directions are normalized to unit length.
2. The empirical characteristic function is compared against that of a standard normal distribution.
3. The selected numerical integration method is correct.
4. The loss is scalar, finite, and differentiable.
5. The statistical and numerical scaling is explicitly documented.
6. The number of random projections is configurable.
7. Gradients reach the encoder embeddings.

Use initial `sigreg_weight = 0.1`.

Use 1,024 projections as the LeWorldModel reference configuration, but benchmark computational cost first. For rapid debugging, fewer projections may be used temporarily; document and report the final choice.

Construct a tensor of shape `[5, B, 32]` containing the four historical embedding positions and the target embedding.

Compute SIGReg across the batch independently for each temporal position, then average the five losses.

Perform diagnostic tests with:

- Random standard-Gaussian embeddings.
- Constant or collapsed embeddings.
- Low-variance embeddings.
- Normal encoder outputs.

Check that the behavior is consistent with distribution matching and that gradients remain finite.

Do not use stop-gradients, EMA teachers, or additional variance/covariance regularizers.

## 5. Implement the combined objective

Use the LeWorldModel-style objective:

```
total_loss = prediction_loss + lambda * sigreg_loss
```

Do not silently replace it with the differently weighted LeJEPA multiview objective.

Return and log:

- `prediction_loss`
- `sigreg_loss`
- `weighted_sigreg_loss`
- `total_loss`

Check loss scales before running full training. An incorrect SIGReg reduction or scaling can make `lambda = 0.1` inappropriate.

## 6. Verify gradient propagation

Before full training, run explicit gradient tests.

**Prediction loss alone:**

- Encoder receives finite, nonzero gradients.
- Predictor receives finite, nonzero gradients.
- Historical and target branches of the shared encoder remain differentiable.

**SIGReg alone:**

- Encoder receives finite, nonzero gradients.
- Predictor is not required to receive SIGReg gradients directly.

**Combined objective:**

- Encoder and predictor both receive finite gradients.
- A single optimizer step changes both modules' trainable parameters.
- No NaNs or infinite values occur.

Retain the Stage 2 causality test: modifying target inputs must not alter the forward prediction `z_pred`.

## 7. Implement training

Use the processed Stage 1 training dataset and existing DataLoader.

Starting parameters:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Maximum epochs: 30
- Initial seed: 11
- Additional seeds: 22 and 33, if computationally feasible
- Encoder and predictor trained jointly
- Device: CUDA when available, otherwise CPU

For every training batch:

1. Load and move tensors to the selected device.
2. Reset gradients.
3. Execute the Stage 2 model forward pass.
4. Calculate prediction loss.
5. Calculate SIGReg.
6. Calculate total loss.
7. Backpropagate.
8. Inspect or validate gradients.
9. Apply optimizer updates.
10. Record the loss components.

First run a short training sanity test on a small subset. Then benchmark one full epoch before launching the full training run.

Do not introduce artificial sensor dropouts in this stage. Model A uses clean input windows and all-ones availability masks.

Do not use failure records or maintenance records as training labels.

## 8. Implement validation

Use only the existing validation partition.

At the end of every epoch:

- Put the model in evaluation mode.
- Disable gradient updates.
- Compute prediction loss.
- Compute SIGReg and total loss.
- Record validation metrics.
- Monitor representation collapse.

For validation SIGReg, use a reproducible projection-sampling policy so different epochs can be meaningfully compared.

Select the checkpoint using validation total loss, subject to predefined noncollapse checks.

Do not evaluate or tune using the test partition.

## 9. Implement representation diagnostics

Monitor the following:

- Mean and standard deviation of latent features.
- Per-dimension latent variance.
- Covariance matrix.
- Effective rank of encoder embeddings.
- Encoder gradient norm.
- Predictor gradient norm.
- Training and validation losses.
- Prediction errors for selected held-out windows.

Calculate embedding statistics on a consistent validation subset, with the model in evaluation mode.

Record signs of representation collapse or dimensional collapse.

Avoid treating a decreasing prediction loss as sufficient evidence of successful representation learning.

## 10. Implement a persistence baseline

Using the trained encoder, calculate a latent persistence prediction:

```
z_persistence = z_history[:, -1, :]
```

Compare this with `z_target` using exactly the same mean-squared-error reduction as the temporal predictor.

Report:

- Trained predictor validation MSE.
- Persistence predictor validation MSE.
- Relative improvement or degradation.

Both comparisons must use the same trained encoder.

Do not compare prediction-error values across independently trained models as if their latent representations were identical.

## 11. Generate training results

Create:

1. Training versus validation total-loss curve.
2. Prediction-loss curve.
3. SIGReg-loss curve.
4. Latent variance and/or effective-rank diagnostic plot.
5. Predictor versus persistence validation comparison.
6. A compact summary of training hyperparameters and checkpoint selection.

Produce clear scientific figures, not decorative plots.

## 12. Checkpoints and reproducibility

Save the best selected checkpoint with:

- Encoder parameters.
- Predictor parameters.
- Optimizer state.
- Epoch.
- Seed.
- Hyperparameters.
- Validation metrics.
- SIGReg implementation settings.

Save metrics in a structured CSV or JSON format.

Ensure the checkpoint can be loaded in a fresh session and reproduces the expected forward outputs.

Suggested outputs:

- `checkpoints/model_a_best_seed11.pt`
- `results/stage3_training_metrics.csv`
- `results/stage3_summary.json`
- `figures/stage3_training_curves.png`
- `figures/stage3_latent_diagnostics.png`
- `figures/stage3_persistence_comparison.png`

## 13. Notebook organization

Continue inside `02_world_model.ipynb`, adding the following sections after Stage 2:

- **7. SIGReg and End-to-End Training**
    - 7.1 Training configuration
    - 7.2 Prediction loss
    - 7.3 SIGReg implementation and tests
    - 7.4 Combined objective
    - 7.5 Gradient verification
    - 7.6 Training loop
    - 7.7 Validation and checkpoint selection
    - 7.8 Representation diagnostics
    - 7.9 Persistence baseline
    - 7.10 Results and Stage 3 completion checks

Put reusable loss and training code in appropriate `src/` modules. Use the notebook for explanation, execution, inspection, and scientific results.

Do not duplicate existing Stage 2 neural-network code.

## 14. Acceptance criteria

Stage 3 is complete only when:

- Both modules train end-to-end.
- The shared encoder and target branches remain differentiable.
- SIGReg has been validated against its mathematical definition.
- Training and validation losses are recorded separately.
- The trained representation is demonstrably noncollapsed.
- Validation results are available.
- Persistence comparison is available.
- The best checkpoint is saved and can be reloaded.
- The implementation is reproducible.
- No artificial telemetry corruption or test-set tuning has been introduced.

If any condition fails, report the issue and evidence rather than forcing a passing result or silently changing the architecture.

## 15. Scientific references

**LeWorldModel — Maes et al. (2026)**

https://arxiv.org/html/2603.19312v2

Read Section 3.1, Training Objective; Appendix A, SIGReg; and Section 4.3, Training Curves.

**LeJEPA — Balestriero and LeCun**

https://arxiv.org/html/2511.08544v3

Read Section 4.2.3, Characteristic Functions; and Section 5.1, SIGReg + Prediction Loss.

Implement only Stage 3. Leave Stage 4 sensor corruption and three-model robustness experiments for the next implementation phase.