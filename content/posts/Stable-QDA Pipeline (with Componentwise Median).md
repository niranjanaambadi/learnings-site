Let me walk through Stable-QDA step by step with componentwise median.

---

## Stable-QDA Pipeline (with Componentwise Median)

### Input
- Training data: $X \in \mathbb{R}^{n \times d}$, labels $y \in \{0, 1, \ldots, K-1\}$
- Test point: $x^* \in \mathbb{R}^d$

---

### Step 1: Preprocessing (Optional)

```python
if auto_scale:
    X = RobustScaler().fit_transform(X)  # Median-center, IQR-scale per feature
```

**Effect:** Each feature has median ≈ 0, IQR ≈ 1. Correlations preserved.

---

### Step 2: For Each Class k, Estimate Location

```python
X_k = X[y == k]  # Samples from class k
μ_k = np.median(X_k, axis=0)  # Componentwise median
```

**What happens:** Each coordinate of $\mu_k$ is estimated independently.

**Problem:** If features are correlated, this ignores the elliptical geometry and introduces bias.

---

### Step 3: For Each Class k, Estimate Dispersion

```python
# Center the data
X_k_centered = X_k - μ_k

# MAD scale per feature
scales_k = median(|X_k_centered|, axis=0)

# Spearman correlation (rank-based)
R_k = spearman_correlation(X_k_centered)

# Reconstruct dispersion matrix
Σ_k = diag(scales_k) @ R_k @ diag(scales_k)
```

**What happens:** Dispersion matrix built from robust marginal scales + rank correlation.

---

### Step 4: Estimate Tail Index α

```python
# Project data onto first principal component
proj = X @ V[0]  # V from SVD

# McCulloch quantile estimator
α = mcculloch_estimator(proj)
```

**What happens:** Single global α estimated from pooled data (or per-class if `class_alpha=True`).

---

### Step 5: Precompute for Prediction

```python
for each class k:
    Σ_k_inv = inv(Σ_k)
    log_det_k = log(det(Σ_k))
```

---

### Step 6: Prediction (for test point x*)

```python
for each class k:
    # Mahalanobis distance squared
    d_k² = (x* - μ_k)ᵀ Σ_k⁻¹ (x* - μ_k)
    
    # Stable log-likelihood
    log_lik_k = -(d_k²)^(α/2) - 0.5 * log_det_k + log(prior_k)

# Predict class with highest log-likelihood
ŷ = argmax_k(log_lik_k)
```

**Key difference from Gaussian QDA:** The exponent is $(d^2)^{\alpha/2}$ instead of $d^2/2$.

---

## Where Componentwise Median Hurts

The problem occurs in **Step 2** and propagates to **Step 6**:

```
Step 2: μ_k is biased (ignores correlation)
           ↓
Step 6: d_k² = (x* - μ_k)ᵀ Σ_k⁻¹ (x* - μ_k) is wrong
           ↓
        Decision boundary shifts incorrectly
           ↓
        Misclassification
```

---

## The Fix

Replace Step 2 with spatial median:

```python
μ_k = spatial_median(X_k)  # Respects multivariate geometry
```

Everything else stays the same.
