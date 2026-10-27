# Robustness and Responsiveness of Temporal Joint-Embedding Models under Missing Industrial Telemetry

![Research status](https://img.shields.io/badge/Status-Conference_Abstract_Prepared-orange)

[![Target venue: IAI 2027](https://img.shields.io/badge/Target-IAI_2027-blue)](https://www.ltu.se/en/research/research-subjects/operation-and-maintenance-engineering/conferences/iai2027)

![Python](https://img.shields.io/badge/Python-PyTorch-3776AB)

> **Research status:** The implementation and three-seed, held-out evaluation are complete. This repository documents an IAI 2027 conference study; it does **not** claim that the paper has been accepted or published. The main results concern robustness and responsiveness on **simulated industrial telemetry**.

## Abstract

Industrial condition monitoring relies on continuous sensor measurements, yet missing telemetry can undermine monitoring reliability. Training models to tolerate incomplete observations may improve robustness while reducing sensitivity to meaningful changes in machine behavior. This study investigates this trade-off using a self-supervised temporal joint-embedding predictive architecture adapted from LeWorldModel. A shared encoder represents six-hour blocks of voltage, rotation, pressure, and vibration measurements, while a temporal predictor estimates the next latent state from four historical blocks. Training combines latent prediction loss with SIGReg regularization.

Using the simulated Microsoft Azure Predictive Maintenance dataset comprising 100 machines, we compare three approaches: a clean-trained baseline, corruption-augmented training with zero-filled missing values, and equivalent augmentation with explicit binary availability masks. Experiments use chronologically held-out data, three random seeds, and independent sensor dropout, consecutive data loss, and temporary sensor outages.

At 40% historical dropout, relative prediction-error degradation was 7.04%, 3.63%, and 3.29%, respectively. Corresponding increases in calibrated alert rates were 2.44, 0.72, and 0.99 percentage points. Corruption augmentation substantially improved robustness, whereas explicit availability masks provided no consistent additional benefit. Exploratory failure-event analysis further indicated that increased robustness can coincide with reduced sensitivity to simulated pre-failure operating changes.

The findings demonstrate the importance of evaluating both tolerance to missing telemetry and responsiveness to genuine operating changes when developing reliable AI-based condition-monitoring systems. This trade-off has practical implications for sensor-data quality management, monitoring alerts, and maintenance decision support.

**Keywords:** Industrial AI · Condition monitoring · Self-supervised learning · Joint-embedding predictive architecture · Missing sensor data · System resilience

## Overview

This repository accompanies research prepared for the **9th International Conference on Industrial AI (IAI 2027)** in Luleå, Sweden, within operation and maintenance engineering. It studies a practical question:

> Does training a temporal self-supervised model to tolerate missing historical sensor readings make its predictions more robust **without** reducing its responsiveness to changes in machine operation?

The work adapts **LeWorldModel**, a latent-state predictive framework, to multivariate hourly sensor telemetry. Unlike supervised failure prediction, the core model learns from sequences of measurements **without using failure labels during training**. After training, latent prediction errors are evaluated for robustness under missing data and, separately, for associations with recorded failure events.

### Main contributions

1. **Industrial adaptation:** A lightweight, mask-conditioned, temporal joint-embedding architecture using a shared encoder, a temporal predictor, and SIGReg regularization.
2. **Controlled missingness study:** Three architecture-matched training policies compared under identical historical telemetry corruptions, including held-out burst and sensor-outage patterns.
3. **Robustness and responsiveness analysis:** Separate measurement of prediction-error degradation, calibrated alert changes, and exploratory sensitivity to simulated pre-failure operating changes.
4. **Reproducible evaluation:** Chronological data partitions, three training seeds, paired corruptions, frozen test protocol, and machine-level uncertainty analysis.

The novelty claimed here is **the controlled empirical evaluation in this particular temporal joint-embedding industrial setting**, not the invention of sensor masking, corruption augmentation, SIGReg, or world models.

## Proposed Architecture

```text
AZURE PREDICTIVE MAINTENANCE TELEMETRY
Four hourly sensors: voltage, rotation, pressure, vibration
                             |
              Construct 6-hour observation blocks
                             |
             +---------------+----------------+
             |                                |
      HISTORICAL CONTEXT                 FUTURE TARGET
      4 blocks = 24 hours                Next 6 hours
      [4, 6, 4] values                    [6, 4] values
             |                                |
  Zero-fill missing readings             Always observed
  + availability-mask input             + all-ones mask
             |                                |
      4 x 48 inputs                       48 inputs
             |                                |
             +-------- SHARED ENCODER --------+
                       MLP: 48 -> 64 -> 64 -> 32
             |                                |
      4 historical latents               Target latent
        [4, 32]                              [32]
             |                                |
         Concatenate                           |
           [128]                               |
             |                                 |
       TEMPORAL PREDICTOR                       |
       MLP: 128 -> 128 -> 64 -> 32             |
             |                                 |
      Predicted latent ----------------- Target latent
             |               |                 |
             +---------------+-----------------+
                             |
              MSE prediction loss + λ SIGReg
                             |
                  TRAINED WORLD MODEL
                             |
           Latent surprise = ||z_pred - z_target||₂
                             |
              Robustness and alert evaluation
```

### Representation learning

| Component                    | Study configuration                                          |
| ---------------------------- | ------------------------------------------------------------ |
| Sampling interval            | 1 hour                                                       |
| Historical observation block | 6 hours × 4 sensors = 24 readings                            |
| Temporal context             | 4 blocks (24 hours)                                          |
| Target                       | Immediately following 6-hour block                           |
| Shared encoder input         | 24 masked readings + 24 availability indicators = 48 values per block |
| Encoder                      | MLP `48 → 64 → 64 → 32`, GELU hidden activations             |
| Predictor input              | Four ordered 32-dimensional historical embeddings = 128 values |
| Temporal predictor           | MLP `128 → 128 → 64 → 32`, GELU hidden activations           |
| Trainable parameters         | 36,224                                                       |
| Learning objective           | Latent prediction MSE + `0.1 × SIGReg`                       |
| SIGReg settings              | Epps–Pulley-based statistic; 1,024 random projections; 17 integration nodes |
| Optimizer                    | Adam, learning rate `0.001`                                  |
| Training                     | Up to 30 epochs; batch size 128; seeds 11, 22, 33            |

Both historical and target blocks use **one shared, trainable encoder**: there is no frozen target encoder, stop-gradient operation, or EMA teacher. The future target is **not** an input to the temporal predictor. The model contains no action inputs, reconstruction decoder, or supervised failure classifier.

The objective is conceptually:

```text
L_total = mean((z_pred - z_target)^2) + λ * SIGReg(Z)
```

SIGReg encourages noncollapsed, statistically distributed embeddings. It does **not** directly identify failure classes or guarantee physically meaningful latent dimensions.

## Dataset and Preprocessing

The experiments use the **Microsoft Azure Predictive Maintenance dataset** (a simulated benchmark, not measurements collected from a physical factory). The source provides hourly telemetry along with separate failure, maintenance, error, and machine-information tables.

The Stage 1 notebook expects these files:

```text
PdM_telemetry.csv
PdM_failures.csv
PdM_maint.csv
PdM_errors.csv
PdM_machines.csv
```

The four model inputs are `volt`, `rotate`, `pressure`, and `vibration`, renamed consistently to `voltage`, `rotation`, `pressure`, and `vibration` during preprocessing.

| Property                               |    Value |
| -------------------------------------- | -------: |
| Machines                               |      100 |
| Hourly telemetry records               |  876,100 |
| Recorded failure rows (all partitions) |      761 |
| Historical context                     | 24 hours |
| Next-block target                      |  6 hours |
| Train windows                          |  101,800 |
| Validation windows                     |   21,500 |
| Test windows                           |   21,500 |

**Leakage safeguards:** A common chronological split is applied to raw telemetry before window construction (approximately 70% / 15% / 15%). Windows do not cross machines, temporal gaps, or partition boundaries. The `StandardScaler` is fitted on training telemetry only. Failure and maintenance information is kept out of self-supervised model inputs and training objectives.

The preprocessing notebook writes model-ready tensors, event metadata, a fitted scaler, and window metadata to `data/processed/`.

**Source and availability:** See Microsoft's [Azure Predictive Maintenance example](https://github.com/Azure-Samples/MachineLearningSamples-PredictiveMaintenance). Obtain the five CSVs from their applicable distribution source. Raw telemetry and generated checkpoint files should not be assumed to be included in this repository.

## Experimental Design: Models A, B, and C

All three models have **identical neural-network architectures** and use the same joint-embedding learning objective.

| Model                        | Training context                                     | Input availability channels                | Interpretation                                               |
| ---------------------------- | ---------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| **A — Standard**             | Clean observations only                              | All ones                                   | Clean-trained baseline                                       |
| **B — Corruption-augmented** | Clean + artificially corrupted observations          | All ones, even where readings were removed | Tests exposure to missingness without explicit mask information |
| **C — Mask-aware**           | Exactly the same clean + corrupted observations as B | True binary availability mask              | Tests the incremental benefit of explicit missingness indicators |

For B and C, removed historical values are replaced by **zero after training-set standardization**. B and C share paired random initialization, minibatch order, corruption realizations, and training settings for each seed. Targets remain clean in all models.

**Training corruption:** 50% clean contexts and 50% contexts with independent dropout at 10%, 20%, or 40% missingness. The primary comparison is **B versus C**. Comparing A with B/C measures the effect of corruption exposure, which is a different question.

**Held-out corruption scenarios:**

- Independent missing observations at 0%, 10%, 20%, and 40%.
- Consecutive loss of one sensor for 6 or 12 hours.
- A temporary single-sensor outage across one or two six-hour blocks.

For the test evaluation, corruption locations are paired across models. The *future target is always clean*: this is a controlled robustness assumption, not a claim that production telemetry will always have complete future measurements.

### Evaluation protocol

1. **Prediction robustness:** Compare corrupted MSE against clean MSE **within each trained model's own latent space**; report `100 × (MSE_corrupted / MSE_clean − 1)`.
2. **Alert stability:** Compute surprise as `||z_pred − z_target||₂`. Each model and seed uses a threshold fixed at the 95th percentile of **clean validation** surprise. Report excess test alert-window fraction in percentage points.
3. **Uncertainty:** Report variation across three training seeds and paired machine-level bootstrap intervals; overlapping windows are not treated as independent machines.
4. **Exploratory failure analysis:** Evaluate recorded failure episodes with event-level matching and explicit alarm times. A surprise score is available **only once its six-hour target block has been observed**, not at the beginning of the predicted interval.

The final evaluation uses a frozen `configs/stage5.yaml` protocol and existing checkpoints. **There is no Stage 5 retraining.**

## Main Results — Chronologically Held-Out Test Set

Values below are **means across seeds 11, 22, and 33**.

### Prediction-error degradation

Lower is better; the percentages are increases relative to each model's own clean MSE.

| Independent historical dropout | A — Standard | B — Augmented | C — Mask-aware |
| ------------------------------ | -----------: | ------------: | -------------: |
| 10%                            |        0.93% |     **0.39%** |          0.56% |
| 20%                            |        2.41% |     **1.11%** |          1.23% |
| 40%                            |        7.04% |         3.63% |      **3.29%** |

At 40% dropout, the increase in prediction error is approximately **48.5% lower for B** and **53.3% lower for C** than for A. These refer to relative reductions in **degradation**, not improvements in failure-detection accuracy.

### Dropout-induced excess alerts

The following are changes in test alert-window rate compared with the corresponding clean test condition, under fixed, model-specific validation-calibrated thresholds.

| Independent historical dropout |            A — Standard |               B — Augmented |          C — Mask-aware |
| ------------------------------ | ----------------------: | --------------------------: | ----------------------: |
| 40%                            | +2.44 percentage points | **+0.72 percentage points** | +0.99 percentage points |

**Interpretation:** Training with corrupted examples substantially improves robustness over clean-only training. Explicit masks give a small advantage over B at 40% dropout in prediction-error degradation, but **no consistent advantage across dropout rates or alert stability**. In this setup, B has the lowest additional alert rate.

### Robustness versus responsiveness

Exploratory clean-condition **24-hour pre-failure episode recall** was approximately **20.4% (A)**, **5.8% (B)**, and **13.3% (C)**, across 98 eligible simulated failure episodes. These results do **not** establish reliable 24-hour early failure warning. A separate 48-hour analysis is exploratory and appears strongly affected by a synthetic pre-failure step change in the benchmark's signal-generation mechanism.

A key observation is therefore a **robustness–responsiveness trade-off**: the model most stable under missing telemetry (B) is also less sensitive to the simulated pre-failure sensor changes. Low prediction-error degradation alone is not sufficient to demonstrate effective condition monitoring.

## Repository Structure

The following reflects the study's implemented modules, notebooks, configuration, and generated artifacts. Some large data, checkpoints, and outputs may be **generated locally rather than committed**.

```text
.
├── configs/
│   ├── base.yaml
│   ├── stage4.yaml
│   └── stage5.yaml
├── data/
│   ├── raw/                         # Downloaded Azure CSVs (local)
│   └── processed/                   # Train/val/test windows, scaler, metadata
├── notebooks/
│   ├── 01_azure_data.ipynb          # Stage 1: preprocessing and windowing
│   ├── 04_world_model_stage_2_3_4.ipynb
│   │                                # Stages 2–4: architecture, training, dropouts
│   └── 05_final_evaluation.ipynb    # Stage 5: frozen test evaluation
├── src/
│   ├── data/                        # Azure windows and corruption schedules
│   ├── models/                      # Encoder, predictor, integrated model
│   ├── training/                    # SIGReg, training, checkpoints
│   └── evaluation/                  # Robustness and failure-event evaluation
├── checkpoints/                     # Selected A/B/C weights (may be local only)
├── results/
│   ├── stage3_summary.json
│   ├── stage4_summary.json
│   └── stage5/                     # Test metrics, predictions, evidence ledger
├── figures/
│   └── stage5/                     # Scientific figures, if generated locally
└── pyproject.toml                  # Project-root marker used by notebooks
```

If you keep notebooks in the repository root rather than `notebooks/`, adjust the displayed paths accordingly. The notebooks locate the project root by searching for `pyproject.toml`.

## Reproducing the Experiments

### 1. Create a Python environment

Install a Python version supported by your PyTorch distribution, then install the required research packages. For example:

```bash
python -m pip install jupyterlab numpy pandas torch scipy scikit-learn matplotlib pyyaml pytest
```

These are the core dependencies visible in the notebooks and supporting modules; check `pyproject.toml` (if supplied) for any additional pinned versions. Execute commands from the **repository root** so `src` imports resolve.

### 2. Obtain and place the Azure CSV files

Place the five original `PdM_*.csv` files in `data/raw/`. The Stage 1 notebook also recognizes `data/raw/azure/` and `data/Microsoft_Azure_Predictive_Maintenance/`.

### 3. Prepare chronological windows

Launch Jupyter from the repository root:

```bash
jupyter lab
```

Run **`01_azure_data.ipynb`** in cell order. It writes:

```text
data/processed/train_windows.pt
data/processed/val_windows.pt
data/processed/test_windows.pt
data/processed/window_metadata.csv
data/processed/failure_events.csv
data/processed/maintenance_events.csv
data/processed/scaler.pkl
```

### 4. Train or load the models

The Stage 2–4 notebook defines and validates the architecture and performs Model A training. It also evaluates the B/C robustness experiment.

With model-ready windows in place, the repository's CLI can train **Models B and C**:

```bash
python -m src.training.robust --models B C --seeds 11 22 33 --threads 4
```

This command **trains missing B/C checkpoints** and skips existing checkpoints by default. Training can be computationally expensive, especially on CPU. Model A is trained by the Stage 3 notebook workflow; the B/C command does not train Model A.

If all nine selected checkpoints are already present, **do not retrain them merely to reproduce Stage 5 analysis**.

### 5. Run the frozen final evaluation

Run **`05_final_evaluation.ipynb`** in cell order with `configs/stage5.yaml`, Stage 4 configuration, processed windows, and the nine model checkpoints available.

The evaluation loads existing weights, uses `model.eval()` and `torch.inference_mode()`, and generates predictions for the predefined scenarios. It then produces tabular analyses and scientific figures. The results include:

```text
results/stage5/summary.json
results/stage5/T1_test_performance.csv
results/stage5/T2_test_alerts.csv
results/stage5/T4_failure_events.csv
results/stage5/test_robustness_per_seed.csv
results/stage5/test_paired_differences.csv
results/stage5/test_machine_bootstrap_intervals.csv
results/stage5/abstract_claims.md
results/stage5/limitations_and_sensitivity.md
```

The full per-window scores and prediction tensors are optional large artifacts. Preserve the frozen protocol, test-run manifests, and checksum records if publishing a reproducibility package. If summary files are already available, descriptive tables and plots can be reconstructed **without rerunning training**.

## Limitations and Interpretation

- **Simulated benchmark:** Azure telemetry and failure behavior come from a simulator; results cannot be assumed to generalize to real industrial assets.
- **Synthetic missingness:** The training and test missing-data patterns were generated artificially. Real failures of sensors and networks may behave differently.
- **Zero-fill cue:** The removed values are exact zeros after standardization. This can implicitly reveal missingness to Model B, so the experiment does not establish that explicit masks would be unhelpful under other imputation schemes.
- **Clean observed target:** The surprise score requires the target block to be available; this does not address cases where the future target itself is missing.
- **Latent-scale differences:** Raw MSE values across separately trained encoders are not directly comparable as a common measure of prediction quality. The reported robustness comparisons use within-model ratios and calibrated thresholds.
- **Alert interpretation:** A high surprise score indicates latent prediction discrepancy, **not** a calibrated failure probability or confirmed diagnosis.
- **Failure-warning limitations:** The 24-hour analysis did not demonstrate reliable warning. The exploratory 48-hour behavior is associated with simulator-injected shifts, not validated physical degradation mechanisms.
- **Limited training seeds:** Three seeds and one simulated dataset cannot establish general robustness across algorithms, plants, or failure modes.

## Study Status

- [x] Chronological Azure data preprocessing and 6-hour temporal window construction
- [x] Shared encoder and temporal latent predictor
- [x] SIGReg-regularized, end-to-end self-supervised training
- [x] Model A: clean-trained baseline, three seeds
- [x] Models B/C: paired corruption-augmented training, three seeds each
- [x] Validation experiments for independent, burst, and outage missingness
- [x] Frozen held-out test evaluation and uncertainty reporting
- [x] Exploratory failure-event and responsiveness analyses
- [x] Abstract prepared for IAI 2027
- [ ] Conference acceptance / final published citation (not claimed)

## Authors and Affiliations

1. **Seyed Iman Hosseini**<sup>1,2,*</sup>
2. **Marko Grobelnik**<sup>1</sup>
3. **Dunja Mladenić**<sup>1,2</sup>

**Affiliations**

1. Department for Artificial Intelligence, Jožef Stefan Institute, Jamova cesta 39, 1000 Ljubljana, Slovenia
2. Jožef Stefan International Postgraduate School, Ljubljana, Slovenia

<sup>*</sup>Corresponding author: **iman.hosseini@ijs.si**

**Target venue:** [9th International Conference on Industrial AI (IAI 2027), Luleå University of Technology](https://www.ltu.se/en/research/research-subjects/operation-and-maintenance-engineering/conferences/iai2027).

## Background and Related Work

- **LeWorldModel:** [Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312) — conceptual foundation for latent future-state prediction and SIGReg-regularized training.
- **LeJEPA:** [Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544) — theoretical background for SIGReg.
- **GRU-D:** [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://arxiv.org/abs/1606.01865) — precedent for explicitly representing missing observations with masks.
- **Related project:** [CIRP 2027 — Label-Efficient and Explainable Defect Detection in Injection Moulding](https://github.com/iman-xosseini/cirp2027). The CIRP study concerns **non-temporal part-level sensor snapshots and label-efficient inspection**, whereas this IAI 2027 work concerns **temporal latent prediction, missing telemetry, and condition-monitoring robustness**.

## Citation and Data Use

If a proceedings paper becomes available, use its final bibliographic details when citing this work. **No DOI, acceptance status, or license is asserted in this README.** The source data remain subject to their provider's applicable terms. See the repository's `LICENSE` file if one is added later.
