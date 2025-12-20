---
title: "2025 Year in Review"
date: 2025-12-20
draft: false
tags: ["machine learning", "robustness", "research"]
---

## Overview

This year I focused on robust classification methods for heavy-tailed data, generative modeling, and probabilistic machine learning.  

The highlights include:

- Evaluating **Stable-QDA** on synthetic and real-world datasets  
- Applying tail-robust covariance estimation  
- Reviewing ICML papers on robust statistics

---

## Robust Quadratic Classification

Quadratic decision boundaries are optimal for Gaussian distributions, but fail for heavy-tailed or multimodal data.  

The class-conditional density can be modeled as an **α-stable distribution**:

$$
p(x \mid y=k) = \mathcal{S}_\alpha(\mu_k, \Sigma_k)
$$

where \(0 < \alpha < 2\) controls the tail heaviness.
This is an inline \(a^*=x-b^*\) equation.


### Python Example: Stable-QDA

```python
import torch
import torch.nn as nn

class StableQDA(nn.Module):
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha
```       
