

## Scientific reading for Stage 1

| Reference                                                    | Exact sections to read                                    | What to focus on                                             |
| ------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------ |
| [Microsoft Azure Predictive Maintenance](https://github.com/Azure-Samples/MachineLearningSamples-PredictiveMaintenance?utm_source=chatgpt.com) | Task 1: Prepare your data and Task 2: Feature Engineering | How telemetry, machine IDs, failures and maintenance records are organized; how temporal features are constructed |
| [LeWorldModel — Maes et al. (2026)](https://arxiv.org/html/2603.19312v2?utm_source=chatgpt.com) | §3.1 — Offline Dataset                                    | How observation trajectories are organized for self-supervised future-state prediction |
| [sktime Forecasting Tutorial](https://www.sktime.org/en/latest/examples/01_forecasting.html?utm_source=chatgpt.com) | §1.3 — Basic Evaluation Workflow, especially §1.3.1       | Why temporal partitioning is necessary for forecasting tasks |
| [Scikit-learn — Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html?utm_source=chatgpt.com) | §12.1, §12.2 and §12.2.1                                  | Correct normalization and how to prevent training/test information leakage |



---

---

---



## Scientific reading for Stage 2

### Paper 1 — LeWorldModel

[LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/html/2603.19312v2?utm_source=chatgpt.com)

Maes et al. (2026)

| Exact part                                                   | What you should understand                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Figure 1 — LeWorldModel Training Pipeline                    | The overall relationship between observations, encoder, predictor and future target |
| §3.1 — Model Architecture, first paragraph and encoder/predictor equations | How raw observations become latent representations and how future embeddings are predicted |
| §3.1 — Model Architecture, paragraph beginning “The predictor takes as input a history…” | How historical representations are used without giving the predictor future information |
| Appendix D — Encoder Architecture                            | The original encoder design and why the model works in latent space |
| Appendix D — Predictor Architecture                          | How the original predictor handles sequences and temporal order |



### Paper 2 — GRU-D

[Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://arxiv.org/html/1606.01865v2?utm_source=chatgpt.com)

Che et al. (2018)

| Exact part                                                   | What you should understand                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| §2 — RNN Models for Time Series with Missing Variables, opening paragraphs | How multivariate time series and missing observations are represented |
| Figure 2                                                     | The relationship between observed measurements, availability masks and timestamps |
| §2 — Binary masking vector definition                        | Why a mask entry equals 1 for an observed measurement and 0 for a missing measurement |
| §3.2 — Methods and Implementation Details, paragraph beginning “Similar to RNN baselines…” | How availability indicators can be concatenated with measurements and supplied to a neural model |



---

---

---



## Scientific reading for Stage 3

I recommend focusing on three short parts of two papers:

| Paper                                                        | Exact section                                | What to understand                                           |
| ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| [LeWorldModel](https://arxiv.org/html/2603.19312v2?utm_source=chatgpt.com) | §3.1 — Training Objective, Equations (1)–(3) | Why prediction loss needs SIGReg and how the combined objective trains encoder and predictor jointly |
| [LeWorldModel](https://arxiv.org/html/2603.19312v2?utm_source=chatgpt.com) | Appendix A — SIGReg                          | Random projections, Epps–Pulley test, isotropic Gaussian target |
| [LeJEPA](https://arxiv.org/html/2511.08544v3?utm_source=chatgpt.com) | §4.2.3 — Characteristic Functions            | How SIGReg is computed from empirical characteristic functions and why it is differentiable |
| [LeJEPA](https://arxiv.org/html/2511.08544v3?utm_source=chatgpt.com) | §5.1 — LeJEPA: SIGReg + Prediction Loss      |                                                              |













