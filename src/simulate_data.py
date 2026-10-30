"""
Project 3 - Marketing Mix Model
Step 1: Simulate 3 years of weekly marketing data with a KNOWN ground truth.

Outputs
- mmm_data.csv      -> what the model sees (date, spend per channel, controls, revenue)
- ground_truth.csv  -> true weekly contribution of every component (kept hidden from the model)
- true_params.json  -> true adstock / saturation / effect parameters + true ROAS per channel
"""
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. Calendar: 156 weeks (3 years), weeks start on Monday
# ---------------------------------------------------------------
n_weeks = 156
dates = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
t = np.arange(n_weeks)
week_of_year = dates.isocalendar().week.to_numpy().astype(int)

# Yearly seasonality wave (peaks mid-year, low in Jan)
season = np.sin(2 * np.pi * (t - 13) / 52)

# ---------------------------------------------------------------
# 2. Spend per channel (weekly, USD)
# ---------------------------------------------------------------
def noisy(base, sd):
    return base * rng.lognormal(0, sd, n_weeks)

# Google Search: always on, follows demand (seasonality) and grows over time
google = noisy(12000 * (1 + 0.25 * season) * (1 + 0.003 * t), 0.10)

# Meta: always on but bursty (campaign pushes)
meta = noisy(8000 * (1 + 0.15 * season), 0.30)

# Microsoft: small, mirrors Google (it usually copies Google campaigns)
microsoft = 0.12 * google * rng.lognormal(0, 0.12, n_weeks)

# YouTube: flighted - on for 6-8 week bursts, otherwise dark
youtube = np.zeros(n_weeks)
for start in [6, 30, 58, 84, 110, 136]:
    length = rng.integers(6, 9)
    youtube[start:start + length] = noisy(15000, 0.15)[start:start + length]

spend = pd.DataFrame(
    {"google_search": google, "meta": meta, "microsoft": microsoft, "youtube": youtube},
    index=dates,
).round(2)
channels = list(spend.columns)

# ---------------------------------------------------------------
# 3. True media response: geometric adstock -> logistic saturation
#    (same functional forms PyMC-Marketing uses, so we can test recovery)
# ---------------------------------------------------------------
def geometric_adstock(x, alpha, l_max=8):
    w = alpha ** np.arange(l_max)
    w = w / w.sum()                       # normalised weights
    return np.convolve(x, w)[: len(x)]

def logistic_saturation(x, lam):
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

true_params = {
    #                alpha = carryover, lam = saturation speed, beta = max weekly revenue
    "google_search": {"alpha": 0.20, "lam": 2.0, "beta": 70000},
    "meta":          {"alpha": 0.45, "lam": 1.5, "beta": 40000},
    "microsoft":     {"alpha": 0.20, "lam": 2.5, "beta": 12000},
    "youtube":       {"alpha": 0.70, "lam": 1.0, "beta": 30000},
}

contrib = pd.DataFrame(index=dates)
for ch in channels:
    p = true_params[ch]
    x_scaled = spend[ch].to_numpy() / spend[ch].max()   # scale 0-1, like PyMC-Marketing
    adstocked = geometric_adstock(x_scaled, p["alpha"])
    contrib[ch] = p["beta"] * logistic_saturation(adstocked, p["lam"])

# ---------------------------------------------------------------
# 4. Baseline + controls
# ---------------------------------------------------------------
intercept = 120000
trend = 250 * t                                        # organic growth
seasonality = 25000 * season

# Promo weeks: Black Friday / Cyber week and a mid-year sale
promo = np.isin(week_of_year, [47, 48, 26]).astype(int)
promo_effect = 35000 * promo

# Price increase of 10% from week 80 -> lowers demand
price_index = np.where(t >= 80, 1.10, 1.00)
price_effect = -300000 * (price_index - 1.0)               # -30k per week after the increase

noise = rng.normal(0, 8000, n_weeks)

contrib["intercept"] = intercept
contrib["trend"] = trend
contrib["seasonality"] = seasonality
contrib["promo"] = promo_effect
contrib["price"] = price_effect
contrib["noise"] = noise

revenue = contrib.sum(axis=1).round(2)

# ---------------------------------------------------------------
# 5. Save model input and hidden ground truth
# ---------------------------------------------------------------
data = spend.copy()
data["promo"] = promo
data["price_index"] = price_index
data["revenue"] = revenue
data.index.name = "date"
data.reset_index().to_csv("mmm_data.csv", index=False)

contrib.index.name = "date"
contrib.round(2).reset_index().to_csv("ground_truth.csv", index=False)

for ch in channels:
    true_params[ch]["total_spend"] = round(float(spend[ch].sum()), 2)
    true_params[ch]["total_contribution"] = round(float(contrib[ch].sum()), 2)
    true_params[ch]["true_roas"] = round(float(contrib[ch].sum() / spend[ch].sum()), 2)
with open("true_params.json", "w") as f:
    json.dump(true_params, f, indent=2)

# ---------------------------------------------------------------
# 6. Sanity summary
# ---------------------------------------------------------------
print(f"Weeks: {n_weeks}  ({dates[0].date()} to {dates[-1].date()})")
print(f"Total revenue: ${revenue.sum():,.0f}")
media_share = contrib[channels].sum().sum() / revenue.sum()
print(f"Media-driven share of revenue: {media_share:.1%}\n")
summary = pd.DataFrame(true_params).T[["total_spend", "total_contribution", "true_roas", "alpha", "lam"]]
print(summary.to_string())
