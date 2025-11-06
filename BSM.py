import math
import numpy as np
from scipy.stats import norm

# Exact BSM call price
def bsm_call_value(s, k, r, t, sigma):
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        # degenerate / immediate payoff case
        return max(s - k * math.exp(-r * t), 0.0)

    a = sigma * math.sqrt(t)  # sigma * sqrt(T)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / a
    d2 = d1 - a

    return s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)

# Delta: dC/dS = N(d1)
def bsm_delta(s, k, r, t, sigma):
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return 1.0 if s > k else 0.0

    a = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / a
    return norm.cdf(d1)

# Gamma: d²C/dS² = φ(d1) / (S * sigma * sqrt(T))
def bsm_gamma(s, k, r, t, sigma):
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return 0.0

    a = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / a
    return norm.pdf(d1) / (s * a)
