# Adaptive Subgradient Methods for Online Deep Learning

Implementation of Diagonal AdaGrad with **Composite Mirror Descent (COMID)** and **Regularized Dual Averaging (RDA)** for online click-through rate prediction, based on the master thesis *"Adaptive Subgradient Methods for Online Deep Learning"* (Heinrich-Heine-University Düsseldorf, 2026).

The algorithms extend the AdaGrad method from [Duchi, Hazan, Singer (2011)](https://jmlr.org/papers/v12/duchi11a.html) to the subgradient setting, combining adaptive per-coordinate learning rates with proximal operators for regularization and projection operators for constrained feasible sets.

---

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Regularization and Constraints](#regularization-and-constraints)
- [Evaluation](#evaluation)
- [Post-Training Analysis](#post-training-analysis)
- [Examples](#examples)
- [Theory Reference](#theory-reference)

---

## Overview

This project provides a self-contained implementation of two diagonal AdaGrad variants for the **Online Convex Optimization (OCO)** framework, applied to a real-world CTR prediction task on the [Avazu dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

Both algorithms solve the unified update rule (Thesis §6.1):

```
w_{t+1} = argmin_{w ∈ K} { ⟨u_t, w⟩ + ηφ(w) + ½⟨w, H_t^x w⟩ }
```

where:
- `u_t` and `H_t^x` depend on the method (COMID or RDA)
- `φ(w)` is the regularizer (none, ℓ₁, or ℓ₂)
- `K` is the feasible set (unconstrained, box, ℓ₂-ball, or ℓ₁-ball)

The key difference between the methods:
- **COMID** performs a gradient step followed by a proximal operation (local view)
- **RDA** accumulates all past gradients and recomputes the solution from scratch (global view), leading to stronger sparsity under ℓ₁-regularization

---

## Algorithms

### Diagonal AdaGrad with COMID (Algorithm 9)

```
For t = 1, 2, ..., T:
    1. Observe loss f_t, compute subgradient g_t
    2. s̃_t = s̃_{t-1} + g_t ⊙ g_t           (accumulate squared gradients)
    3. h_t = δ + √s̃_t                         (per-coordinate scaling)
    4. u_t = ηg_t − h_t ⊙ w_t                 (direction vector, eq. 139)
    5. H_t^x = diag(h_t)                       (diagonal matrix, eq. 139)
    6. w_{t+1} = Π_K(proximal step on u_t)     (§6.2 + §6.3–6.5)
```

### Diagonal AdaGrad with RDA (Algorithm 13)

```
For t = 1, 2, ..., T:
    1. Observe loss f_t, compute subgradient g_t
    2. s̃_t = s̃_{t-1} + g_t ⊙ g_t           (accumulate squared gradients)
    3. ḡ = ḡ + g_t                             (accumulate gradient sum)
    4. u_t = (η/t) · ḡ                         (averaged direction, eq. 140)
    5. H_t^x = diag(h_t) / t                   (scaled matrix, eq. 140)
    6. w_{t+1} = Π_K(closed-form solution)      (§6.2 + §6.3–6.5)
```

---

## Project Structure

```
adaptive-subgradient-methods/
├── optimizers.py    # COMID and RDA optimizer implementations
│                    #   - Proximal operators: none, L1 (eq. 141), L2 (eq. 142)
│                    #   - Projections: box (§6.3), ℓ₂-ball (§6.4), ℓ₁-ball (§6.5)
│                    #   - Newton's method with damping for L2 proximal
│                    #   - Bisection for weighted ℓ₁-ball projection
│
├── model.py         # CTR prediction model
│                    #   - Per-field embeddings with offset-based lookup
│                    #   - Multi-layer perceptron (MLP) head
│
├── data.py          # Data loading and preprocessing
│                    #   - Deterministic feature hashing (MD5-based)
│                    #   - Vectorized hashing via categorical encoding
│                    #   - Stratified on-the-fly sampling
│                    #   - Chronological streaming from CSV
│
├── train.py         # Training pipeline and analysis
│                    #   - Prequential (test-then-train) evaluation
│                    #   - Day-by-day AUC progression
│                    #   - Weight statistics and feature importance
│                    #   - Empirical regret computation
│                    #   - Sparsity and constraint verification
│
├── requirements.txt # Python dependencies
└── .gitignore
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- pandas
- numpy
- scikit-learn

### Setup

```bash
git clone https://github.com/Eddie1337-code/adaptive-subgradient-methods.git
cd adaptive-subgradient-methods

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Download the Avazu CTR dataset from [Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/data) and place the daily CSV files in the project directory:

```
train_141021.csv
train_141022.csv
...
train_141030.csv    # last labeled day (used as test set)
train_141031.csv    # unlabeled (optional, for prediction)
```

Each CSV should contain the columns: `click`, `hour`, and categorical features (`banner_pos`, `site_domain`, `device_model`, etc.). The code auto-detects which features are available and adapts accordingly.

---

## Usage

### Basic Training

```bash
# COMID without regularization (true online, batch_size=1)
python train.py --optimizer comid --lr 0.05

# RDA with L1 regularization
python train.py --optimizer rda --reg l1 --reg_lambda 1e-4 --lr 0.05

# Faster training with mini-batches and data sampling
python train.py --optimizer comid --batch_size 128 --sample_rate 0.1

# Use Apple Silicon GPU
python train.py --optimizer comid --device mps
```

### All Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimizer` | `comid` | `comid`, `rda`, or `adagrad` (PyTorch baseline) |
| `--lr` | `0.05` | Learning rate η |
| `--delta` | `1e-10` | Numerical stability δ |
| `--reg` | `none` | Regularization: `none`, `l1`, `l2` |
| `--reg_lambda` | `1e-5` | Regularization strength λ |
| `--constraint` | `none` | Feasible set: `none`, `box`, `l2_ball`, `l1_ball` |
| `--constraint_param` | `1.0` | Constraint bound c or radius R |
| `--embed_dim` | `8` | Embedding dimension per feature field |
| `--hidden_dims` | `128 64` | MLP hidden layer dimensions |
| `--hash_size` | `100000` | Hash table size per feature field |
| `--batch_size` | `1` | Mini-batch size (1 = true online learning) |
| `--sample_rate` | `1.0` | Fraction of data to use (stratified by hour × site) |
| `--epochs` | `1` | Number of passes over training data |
| `--seed` | `42` | Random seed for reproducibility |
| `--device` | `cpu` | Device: `cpu`, `cuda`, or `mps` |
| `--log_interval` | `500` | Log every N batches |
| `--save_model` | — | Path to save model checkpoint |
| `--load_model` | — | Path to load model checkpoint |
| `--predict` | — | Path to unlabeled CSV for prediction |
| `--predict_output` | — | Output path for predictions |
| `--compare` | — | Run comparison of all optimizer variants |

---

## Regularization and Constraints

The implementation covers all combinations from Thesis Chapter 6:

### Regularizers φ (§6.2)

| Type | Formula | Effect | Code |
|------|---------|--------|------|
| None | φ ≡ 0 | Standard AdaGrad | `--reg none` |
| ℓ₁ | φ(w) = λ‖w‖₁ | Promotes sparsity (LASSO) | `--reg l1 --reg_lambda 1e-4` |
| ℓ₂ | φ(w) = λ‖w‖₂ | Controls weight norm | `--reg l2 --reg_lambda 1e-4` |

The ℓ₁ proximal uses soft-thresholding (eq. 141). The ℓ₂ proximal uses Newton's method with step-size damping (eq. 142, p. 73).

### Feasible Sets K (§6.3–6.5)

| Type | Set | Projection | Code |
|------|-----|-----------|------|
| Unconstrained | K = ℝⁿ | Identity | `--constraint none` |
| Box | K = {w : \|wᵢ\| ≤ c} | Element-wise clipping | `--constraint box --constraint_param 1.0` |
| ℓ₂-ball | K = {w : ‖w‖₂ ≤ R} | Rescaling | `--constraint l2_ball --constraint_param 5.0` |
| ℓ₁-ball | K = {w : ‖w‖₁ ≤ R} | Weighted bisection | `--constraint l1_ball --constraint_param 10.0` |

The ℓ₁-ball projection uses a **weighted** threshold ν/hᵢ (Thesis §6.5), accounting for AdaGrad's per-coordinate scaling. This differs from the standard Euclidean projection (Duchi et al. 2008) and is derived via KKT conditions with bisection.

All regularizers and constraints can be freely combined, yielding 3 × 4 = 12 configurations per optimizer.

---

## Evaluation

### Prequential (Test-Then-Train) Protocol

The training follows a prequential evaluation scheme appropriate for online learning:

```
1. Evaluate untrained model on day 1        → Baseline AUC ≈ 0.5
2. Train on day 1
3. Evaluate on day 2 (before training)      → AUC after 1 day
4. Train on day 2
5. Evaluate on day 3 (before training)      → AUC after 2 days
   ...
9. Train on day 9
10. Evaluate on day 10 (final test)         → Final AUC
```

This produces a day-by-day AUC progression table showing how the model generalizes to unseen future data.

### Metrics

- **AUC** (Area Under the ROC Curve): Probability that a random click is ranked above a random non-click. Threshold-independent, robust to class imbalance.
- **LogLoss** (Binary Cross-Entropy): Measures calibration of predicted probabilities.

---

## Post-Training Analysis

After each run, the system automatically produces:

### 1. Sparsity Analysis
Number and percentage of exact zeros per parameter layer. Under ℓ₁-regularization, RDA is expected to produce significantly more sparsity than COMID.

### 2. Constraint Verification
For each parameter, verifies that the constraint ‖w‖₁ ≤ R, ‖w‖₂ ≤ R, or |wᵢ| ≤ c is satisfied.

### 3. Effective Learning Rates
Distribution of η/hᵢ across coordinates. Large ratios indicate strong adaptivity — different coordinates learn at vastly different speeds, which is the core advantage of AdaGrad.

### 4. Empirical Regret
Computes R_T = Σ f_t(w_t) − Σ f_t(w_T), where w_T is the final model. If R_T/T → 0, the online algorithm achieves sublinear regret (Theorems 4.3, 5.4).

### 5. Feature Importance
Per-field embedding norms ‖E_i‖₂ as a proxy for feature influence. Features with larger embedding norms contribute more to predictions.

---

## Examples

### Compare COMID, RDA, and PyTorch AdaGrad

```bash
python train.py --compare --batch_size 128 --sample_rate 0.1
```

### Train with L1 regularization and save model

```bash
python train.py --optimizer rda --reg l1 --reg_lambda 1e-4 \
                --save_model model_rda_l1.pt
```

### Generate predictions on unlabeled data

```bash
python train.py --predict train_141031.csv --load_model model_rda_l1.pt
```

### Constrained optimization with ℓ₁-ball

```bash
python train.py --optimizer comid --reg l1 --reg_lambda 1e-5 \
                --constraint l1_ball --constraint_param 10.0
```

### Quick experiment (5% data, small model)

```bash
python train.py --optimizer comid --sample_rate 0.05 \
                --hash_size 5000 --embed_dim 4 --hidden_dims 32 16 \
                --batch_size 128
```

---

## Theory Reference

The implementation follows the thesis structure:

| Thesis Chapter | Content | Code |
|----------------|---------|------|
| Ch. 4, Alg. 9 | AdaGrad with COMID | `DiagonalAdaGradCOMID` |
| Ch. 5, Alg. 13 | AdaGrad with RDA | `DiagonalAdaGradRDA` |
| §6.1, eq. 139–140 | Unified update (u_t, H_t^x) | `step()` method |
| §6.2, eq. 141 | ℓ₁ proximal (soft-thresholding) | `reg_type='l1'` |
| §6.2, eq. 142 | ℓ₂ proximal (Newton + damping) | `_l2_norm_proximal()` |
| §6.3 | Box constraints | `_project_box()` |
| §6.4 | ℓ₂-ball projection | `_project_l2_ball()` |
| §6.5 | ℓ₁-ball projection (weighted bisection) | `_project_l1_ball_weighted()` |
| Thm. 4.3, 5.4 | Regret bounds | `analyze_experiment()` |

Every step in the optimizer code is annotated with the corresponding thesis equation number.

---

## License

This project was developed as part of a master thesis at the Mathematical Institute of Heinrich-Heine-University Düsseldorf.

## References

- J. Duchi, E. Hazan, Y. Singer. *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.* JMLR, 2011.
- Y. Nesterov. *Lectures on Convex Optimization.* Springer, 2018.
- L. Xiao. *Dual Averaging Methods for Regularized Stochastic Learning and Online Optimization.* JMLR, 2010.
