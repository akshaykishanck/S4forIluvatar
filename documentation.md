# GPU Latency Prediction for Ilúvatar FaaS

## Introduction

The controller in Ilúvatar's cloud service receives FaaS requests at a high rate (anywhere from 0.2–1.3 seconds between arrivals) and routes each request to either a GPU or a CPU. To decide on the routing, the controller estimates the end-to-end latency (`e2etime`) for both compute paths.

The current GPU latency estimates are inaccurate, which can lead to suboptimal routing decisions. Our objective is to develop an algorithm that can **accurately estimate the latency of an incoming FaaS request at the time of invocation**, enabling the controller to make better scheduling decisions.

---

## Features Affecting GPU Latency

The overall latency is a sum of the **queue waiting time** and the **GPU execution time**. The features below aim to capture the factors that influence these:

| Feature | Description |
|---|---|
| **FQDN** | Unique function identifier. Different functions have different execution characteristics and thus different GPU execution times. |
| **IAT** | Inter-arrival time — time elapsed since the last invocation request (any function). |
| **IAT_FQDN** | Inter-arrival time per FQDN — time since the last invocation of the same function. Acts as a proxy for warm vs. cold start likelihood. |
| **target_queue_len** | Current queue length for this function's FQDN at the time of invocation. |
| **others_len_queue** | Aggregate queue length of all *other* FQDN queues — captures GPU contention. |
| **is_status_Active / Inactive / Throttled** | Queue status at FQDN level — captures whether the queue is actively serving, idle, or throttled. |
| **num_running_funcs_filled** | Number of functions currently executing on the GPU at invocation time. |
| **is_cold_start** | Binary flag indicating whether the execution will be a cold start (new container spin-up) or a warm start (existing container reuse). |
| **gpu_warm_results_sec** | Historical average GPU execution time for warm starts of this FQDN. |
| **gpu_cold_results_sec** | Historical average GPU execution time for cold starts of this FQDN. Together with `gpu_warm_results_sec`, these numerically represent the FQDN's typical execution profile. |

---

## Data & Processing

We have logs from **44+ sessions**, each 30 minutes long. The `s4_feature_pipeline.py` script converts raw `worker1.log` files into structured DataFrames, where each row represents a single FaaS invocation with all features listed above.

**Average invocations per session:** ~1,800

The raw feature columns produced by the pipeline are:
- `timestamp`, `tid`, `fqdn`
- `iat`, `iat_fqdn`
- `num_running_funcs_filled`
- `target_queue_len`, `others_len_queue`
- `is_status_Active`, `is_status_Inactive`, `is_status_Throttled`
- `gpu_warm_results_sec`, `gpu_cold_results_sec`
- `is_cold_start`

> Additional columns are generated as part of feature engineering and may be useful for further analysis.

---

## Current Estimation: Baseline Error Analysis

Before developing any ML model, we evaluated the accuracy of the **current GPU latency estimation strategy** (a static Kalman Filter implemented in the Ilúvatar landlord dispatcher). This provides the lower bound for model performance.

The `evaluate_current_estimate.py` script computes per-session MAE, MSE, and Percentage Relative Errors. Results are saved as a CSV and can be explored interactively via **`analyse_current_estimate.ipynb`**.

**Key finding:** The highest estimation errors were reported for `pyhpc-eos`, `pyhpc-isoneural`, and `torch_rnn` functions (mean actual latency ~3–6 seconds), with static estimates off by **4–7 seconds**.

---

## Baseline ML Model: Random Forest

A Random Forest regressor was trained as a baseline ML model for GPU latency prediction. In addition to the core features from the pipeline, **lag features** (up to 10 previous invocations) of the queue and GPU representation features were added to capture temporal dependencies.

Feature engineering for the RF model is implemented in `rf_feature_pipeline.py`. Training and evaluation is performed in `rf_tuning.py`, which:
- Trains on a subset of sessions, tests on a separate holdout set
- Uses `GridSearchCV` to identify optimal hyperparameters
- **Log-transforms the target (`e2etime`)** to mitigate the heavy right-skew of the latency distribution (where extreme outliers can reach over 20× the mean)

**Training configuration:** 36 sessions, lag window of 10 invocations.

**Results:** The highest errors were still reported for `pyhpc-eos`, `pyhpc-isoneural`, and `torch_rnn` functions — but predictions improved dramatically, now off by only **0.5–2.5 seconds** (compared to 4–7 seconds for the static baseline).

### Feature Importances (RF Model)

| Feature | Importance |
|---|---|
| `is_cold_start` | 0.361 |
| `gpu_warm_results_sec` | 0.232 |
| `others_len_queue` | 0.204 |

The cold start flag is the dominant predictor, followed by the historical warm execution time and the competing queue load — all of which are directly tied to the physical mechanics of GPU scheduling.

---

---

## S4: Structured State Space Sequence Modeling

### Introduction to S4

S4 (Structured State Space Sequences) is a sequence modeling architecture specifically designed to capture **Long-Range Dependencies (LRDs)** in sequential data. Unlike Transformers (which scale quadratically with sequence length) or standard RNNs (which struggle to retain distant history), S4 is built directly on the **continuous-time State Space Model (SSM)** formulation:

```
x'(t) = A·x(t) + B·u(t)
 y(t) = C·x(t) + D·u(t)
```

where:
- `u(t)` — current input (the feature vector at timestep t)
- `x(t)` — the hidden **state vector** (N × 1), continuously updated from history
- `x'(t)` — the derivative / updated state
- `A, B, C, D` — learned parameter matrices

The central idea is that the state matrix `A`, when initialized carefully using **HiPPO** (High-order Polynomial Projection Operators), allows `x(t)` to act as a **compressed memory** of all past inputs. S4 then uses a special parameterization (**NPLR — Normal Plus Low-Rank**) to make this computationally tractable, achieving **O(N log N)** training via convolutions and **O(N)** recurrent inference.

---

### Why S4 Is a Good Fit for FaaS Latency Prediction

The Random Forest baseline, despite incorporating lag features, is fundamentally a **bag-of-rows** model — it treats each invocation as largely independent and cannot explicitly reason about the state of the system as it evolves moment-to-moment through a session.

In contrast, GPU latency in FaaS is an **inherently sequential, stateful system**:

1. **Queue dynamics are global and continuous** — the latency of an incoming request depends on all prior requests that have accumulated in the queue, in order. This is a textbook LRD problem.
2. **Cold vs. warm start is a hidden state** — whether a container is "warm" for a given FQDN depends on the history of recent invocations of that FQDN, not just the current feature snapshot.
3. **GPU contention evolves over time** — a burst of concurrent FQDN requests followed by a quiet period produces very different latency distributions that lag features can only approximate crudely.

S4's state vector `x(t)` can naturally encode all of the above by continuously updating itself as the sequence of FaaS invocations flows through it — making it well suited for this problem.

---

### Session-Based Training Design

A **session** is defined as a single continuous 30-minute trace log (`worker1.log`). The key design decisions for training S4 on this data are:

1. **Each batch = one session.** The S4 model processes all invocations within a session as a single sequence, in chronological order.
2. **State vector is reset to zero at the start of each session.** This correctly models the physical reality that each session begins with a fresh controller state — there is no carryover between independent experiment runs.
3. **`batch_size=1` in the DataLoader** ensures that variable-length sessions (which range from ~1,000 to ~5,500 invocations) are processed without padding or truncation, preserving the true temporal structure of each session.
4. After training, the learned parameter matrices `A`, `B`, `C`, `D` are fixed. During **inference**, the state vector is again reset to zero at the start of a new sequence, replicating the session boundary.

---

### What the State Vector Memorizes

The state vector `x(t)` (dimension N, tunable) acts as a rolling compressed summary of the invocation history within the current session. In the context of FaaS scheduling, it will learn to encode:

- **Queue pressure history**: How long and how heavily queues have been loaded across all FQDNs in the recent past
- **GPU concurrency trends**: Whether the system has been transitioning from low to high concurrency or vice versa
- **Cold/warm start rhythm**: The temporal pattern of cold starts for a given FQDN as its container warms up and cools down across the session
- **IAT patterns**: Inter-arrival time trends that foreshadow bursts or quiet periods

The S4 model, unlike the RF lag features which are a fixed window of the last K invocations, maintains this memory **across the entire session history** — with theoretically unlimited look-back at O(N) inference cost.

---

### Model Architecture (Global Model)

We implement a **single global S4 model** that handles all FQDNs together. This is the preferred approach for the following reasons:

- The **queue and GPU state are shared** — the system state at any moment is a product of all pending invocations across all FQDNs, not just one.
- A per-FQDN model would only update its state vector when a request for *that specific FQDN* arrives, meaning it would miss the queue and GPU state changes driven by other functions running between its invocations.
- FQDN identity is instead encoded **numerically** via `gpu_warm_results_sec` and `gpu_cold_results_sec`, which together represent the function's typical execution profile.

The S4 model is implemented in `s4_iluvatar.py` via the `FaaS_S4_Predictor` class, which stacks multiple S4 layers with residual connections and a linear decoder for latency regression.

---

## Next Steps

- Finalize S4 hyperparameter search: `d_model`, `d_state`, `n_layers`, learning rate, gradient accumulation steps
- Apply log-transformation to `e2etime` target in S4 training loop (matching the RF pipeline)
- Per-FQDN error breakdown to identify which functions remain hardest to predict
