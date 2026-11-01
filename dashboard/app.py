"""
Marketing Mix Model - interactive dashboard (Streamlit).

The Bayesian model itself (PyMC-Marketing) is fitted in notebooks/03_model.ipynb; it is
too heavy to refit inside a free web app. This dashboard combines:
  - the model input data (data/mmm_data.csv), explored live,
  - the model's published results (tables below, copied from 03_model.ipynb outputs),
  - the simulation's ground truth (data/ground_truth/), which powers a budget simulator
    using the TRUE response curves, so viewers can test any budget split against reality.

Run locally:  streamlit run dashboard/app.py      (from the repo root)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
REPO_URL = "https://github.com/paolofelix/marketing-mix-model"

CHANNELS = ["google_search", "meta", "microsoft", "youtube"]
LABELS = {"google_search": "Google Search", "meta": "Meta", "microsoft": "Microsoft", "youtube": "YouTube"}
COLORS = {"Google Search": "#2a78d6", "Meta": "#eb6834", "Microsoft": "#1baf7a", "YouTube": "#eda100"}
BASE_COLORS = {"Baseline": "#d6d4ce", "Trend": "#b9b7b0", "Seasonality": "#9c9a93",
               "Promo": "#e87ba4", "Price increase": "#6250d6"}

# Current weekly spend = average of the last 52 weeks (as in 03_model.ipynb)
CURRENT = {"google_search": 16620.0, "meta": 8486.0, "microsoft": 2011.0, "youtube": 4024.0}
PLANS = {
    "Current plan": CURRENT,
    "MMM recommendation": {"google_search": 21869.0, "meta": 4243.0, "microsoft": 3016.0, "youtube": 2012.0},
    "True optimum": {"google_search": 20571.0, "meta": 5590.0, "microsoft": 2967.0, "youtube": 2012.0},
}

# Published model results (outputs of notebooks/03_model.ipynb)
ROAS_RESULTS = pd.DataFrame({
    "channel": CHANNELS,
    "true_roas": [2.45, 1.52, 3.99, 0.61],
    "mmm_mean": [2.25, 1.25, 6.68, 0.31],
    "mmm_lo": [0.77, 0.27, 0.67, 0.04],
    "mmm_hi": [3.95, 2.83, 16.65, 0.69],
    "calibrated_mean": [2.39, 1.20, 3.89, 0.31],
})
ADSTOCK = pd.DataFrame({
    "channel": CHANNELS,
    "true": [0.20, 0.45, 0.20, 0.70], "mean": [0.25, 0.43, 0.25, 0.61],
    "lo": [0.03, 0.14, 0.02, 0.26], "hi": [0.51, 0.74, 0.57, 0.90],
})
MODEL_CONTRIB = {"google_search": 5.18e6, "meta": 1.64e6, "microsoft": 1.86e6, "youtube": 0.19e6}
MARGINAL = pd.DataFrame({
    "channel": CHANNELS, "weekly_spend": [16620, 8486, 2011, 4024],
    "avg_roas": [2.22, 1.27, 6.48, 0.34], "marginal_roas": [1.70, 0.89, 3.65, 0.33],
    "marginal_lo": [0.54, 0.23, 0.33, 0.04], "marginal_hi": [2.93, 1.73, 8.71, 0.74],
    "true_marginal": [1.84, 1.43, 2.66, 0.62],
})
CALIBRATION = pd.DataFrame({
    "channel": CHANNELS, "true_roas": [2.45, 1.52, 3.99, 0.61],
    "before_mean": [2.25, 1.25, 6.68, 0.31], "before_width": [3.18, 2.56, 15.99, 0.65],
    "after_mean": [2.39, 1.20, 3.89, 0.31], "after_width": [3.14, 2.39, 2.02, 0.65],
})
MODEL_UPLIFT = {"expected": 5721, "lo": -1162, "hi": 12962, "prob_better": 0.92, "true": 3010}

st.set_page_config(page_title="Marketing Mix Model", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Data and true response functions (same forms as src/simulate_data.py)
# ---------------------------------------------------------------------------
@st.cache_data
def load():
    df = pd.read_csv(DATA_DIR / "mmm_data.csv", parse_dates=["date"])
    gt = pd.read_csv(DATA_DIR / "ground_truth" / "ground_truth.csv", parse_dates=["date"])
    params = json.loads((DATA_DIR / "ground_truth" / "true_params.json").read_text())
    return df, gt, params


def logistic(z):
    return (1 - np.exp(-z)) / (1 + np.exp(-z))


def true_response(ch, weekly_spend, params, scale):
    """Steady-state weekly revenue at a constant weekly spend (normalised adstock of a constant = itself)."""
    p = params[ch]
    return p["beta"] * logistic(p["lam"] * np.asarray(weekly_spend, dtype=float) / scale[ch])


def geometric_adstock(x, alpha, l_max=8):
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    return np.convolve(x, w)[: len(x)]


df, gt, params = load()
SCALE = {ch: float(df[ch].max()) for ch in CHANNELS}

# ---------------------------------------------------------------------------
# Header and KPIs
# ---------------------------------------------------------------------------
st.title("Marketing Mix Model: what actually drove revenue?")
st.markdown("A Bayesian MMM (PyMC-Marketing) measuring the **incremental** revenue from Google Search, "
            "Meta, Microsoft Ads and YouTube over 156 weeks. The data is simulated, so the **true** effect "
            "of every channel is known, and the model can be checked against it.")
st.sidebar.header("About")
st.sidebar.markdown("The model is fitted in `notebooks/03_model.ipynb`. Model results on this page are "
                    "copied from that notebook. The **budget simulator** and **carryover explorer** use "
                    "the simulation's true response curves, so any plan you try is scored against reality.")
st.sidebar.markdown(f"[GitHub repo]({REPO_URL}) · [Findings]({REPO_URL}/blob/main/FINDINGS.md)")

total_rev = df["revenue"].sum()
total_spend = df[CHANNELS].sum().sum()
true_media = gt[CHANNELS].sum().sum()
k = st.columns(5)
k[0].metric("Weeks", f"{len(df)}")
k[1].metric("Revenue", f"${total_rev / 1e6:,.1f}M")
k[2].metric("Media spend", f"${total_spend / 1e6:,.2f}M")
k[3].metric("Media share of revenue", "30.7%", delta=f"true {true_media / total_rev:.1%}", delta_color="off", delta_arrow="off")
k[4].metric("Holdout forecast error (MAPE)", "3.8%", help="Last 13 weeks, model refit on the first 143")

tabs = st.tabs(["Spend & revenue", "Naive regression vs MMM", "What drove revenue",
                "Budget simulator", "Carryover explorer", "Geo-lift calibration", "Model checks"])

# ---------------------------------------------------------------------------
# Tab: Spend & revenue
# ---------------------------------------------------------------------------
with tabs[0]:
    lo, hi = df["date"].min().date(), df["date"].max().date()
    rng = st.slider("Weeks shown", min_value=lo, max_value=hi, value=(lo, hi), format="MMM YYYY")
    d = df[df["date"].between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]))]

    st.subheader("Weekly revenue")
    fig = go.Figure(go.Scatter(x=d["date"], y=d["revenue"], mode="lines", line=dict(color="#0b0b0b", width=2),
                               name="Revenue", hovertemplate="$%{y:,.0f}<extra></extra>"))
    promo = d[d["promo"] == 1]
    fig.add_trace(go.Scatter(x=promo["date"], y=promo["revenue"], mode="markers", name="Promo week",
                             marker=dict(size=9, color="#e87ba4", line=dict(color="white", width=2)),
                             hovertemplate="Promo week: $%{y:,.0f}<extra></extra>"))
    price_change = df.loc[df["price_index"] > 1, "date"].min()
    if d["date"].min() <= price_change <= d["date"].max():
        fig.add_vline(x=price_change, line_dash="dash", line_color="#52514e")
        fig.add_annotation(x=price_change, y=1.04, yref="paper", text="+10% price", showarrow=False)
    fig.update_layout(height=320, hovermode="x unified", yaxis=dict(title="Revenue", tickprefix="$"),
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Weekly spend by channel")
    sl = d.melt(id_vars="date", value_vars=CHANNELS, var_name="channel", value_name="spend")
    sl["channel"] = sl["channel"].map(LABELS)
    fig = px.area(sl, x="date", y="spend", color="channel", color_discrete_map=COLORS)
    fig.update_traces(line_width=0.5, hovertemplate="%{fullData.name}: $%{y:,.0f}<extra></extra>")
    fig.update_layout(height=340, hovermode="x unified", xaxis_title=None,
                      yaxis=dict(title="Spend", tickprefix="$"), legend=dict(orientation="h", y=-0.2, title=None),
                      margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    tot = d[CHANNELS].sum()
    st.dataframe(pd.DataFrame({"Channel": [LABELS[c] for c in CHANNELS], "Spend": tot.values,
                               "Share": (tot / tot.sum()).values,
                               "Avg weekly spend": d[CHANNELS].mean().values})
                 .style.format({"Spend": "${:,.0f}", "Share": "{:.0%}", "Avg weekly spend": "${:,.0f}"}),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab: Naive regression vs MMM
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Simple regression gets the wrong answer")
    X = np.column_stack([np.ones(len(df)), df[CHANNELS]])
    naive = np.linalg.lstsq(X, df["revenue"], rcond=None)[0][1:]
    r = ROAS_RESULTS.copy()
    r["naive"] = naive
    r["Channel"] = r["channel"].map(LABELS)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=r["Channel"], y=r["naive"], name="Naive regression", marker_color="#b7cdea",
                         marker_line_color="white", marker_line_width=2,
                         hovertemplate="%{x}<br>Naive: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=r["Channel"], y=r["mmm_mean"], name="Bayesian MMM (90% interval)",
                         marker_color="#1f4f8f", marker_line_color="white", marker_line_width=2,
                         error_y=dict(type="data", symmetric=False, array=r["mmm_hi"] - r["mmm_mean"],
                                      arrayminus=r["mmm_mean"] - r["mmm_lo"], color="#52514e"),
                         hovertemplate="%{x}<br>MMM: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=r["Channel"], y=r["true_roas"], name="True ROAS", mode="markers",
                             marker=dict(symbol="diamond", size=14, color="#eb6834",
                                         line=dict(color="white", width=2)),
                             hovertemplate="%{x}<br>True: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#52514e", line_width=1)
    fig.update_layout(barmode="group", height=440, yaxis_title="ROAS (revenue per $1 of spend)",
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(r[["Channel", "true_roas", "naive", "mmm_mean", "mmm_lo", "mmm_hi"]].rename(columns={
        "true_roas": "True ROAS", "naive": "Naive regression", "mmm_mean": "MMM mean",
        "mmm_lo": "MMM 5%", "mmm_hi": "MMM 95%"}).style.format(precision=2), width="stretch", hide_index=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Why naive regression fails**")
        st.markdown("""
| Problem | In this data |
|---|---|
| **Confounding** | Google spend rises with seasonal demand, so it gets credit for demand it didn't create |
| **Collinearity** | Microsoft copies Google's campaigns, so the model can't split credit and Microsoft goes negative |
| **Carryover** | YouTube's effect lasts weeks after each burst, so its same-week correlation with revenue is near zero |
""")
    with c2:
        corr = df[CHANNELS + ["revenue"]].corr()
        corr.index = corr.columns = [LABELS.get(c, "Revenue") for c in corr.columns]
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=["#eb6834", "#f0efec", "#2a78d6"],
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_traces(hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>")
        fig.update_layout(height=340, margin=dict(t=10), coloraxis_colorbar=dict(title="r"))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab: What drove revenue
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("True weekly revenue decomposition")
    st.caption("From the simulation's ground truth (the model never sees this). Media sits on top of "
               "the baseline; the price increase pulls revenue down from week 80.")
    comp = gt.copy()
    comp["Baseline"] = comp["intercept"] + comp["noise"]
    comp = comp.rename(columns={"trend": "Trend", "seasonality": "Seasonality", "promo": "Promo",
                                "price": "Price increase", **LABELS})
    show_base = st.toggle("Show baseline, trend and seasonality", value=False)
    parts = (["Baseline", "Trend", "Seasonality"] if show_base else []) + \
            ["Promo", "Price increase"] + list(LABELS.values())
    long = comp.melt(id_vars="date", value_vars=parts, var_name="component", value_name="revenue")
    fig = px.bar(long, x="date", y="revenue", color="component",
                 color_discrete_map={**COLORS, **BASE_COLORS}, category_orders={"component": parts})
    fig.update_traces(hovertemplate="%{fullData.name}: $%{y:,.0f}<extra></extra>", marker_line_width=0)
    fig.update_layout(barmode="relative", bargap=0.05, height=420, hovermode="x unified", xaxis_title=None,
                      yaxis=dict(title="Weekly revenue", tickprefix="$"),
                      legend=dict(orientation="h", y=-0.18, title=None), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Total contribution over 156 weeks: model vs truth")
    cc = pd.DataFrame({"Channel": [LABELS[c] for c in CHANNELS],
                       "Model": [MODEL_CONTRIB[c] for c in CHANNELS],
                       "True": [params[c]["total_contribution"] for c in CHANNELS]})
    cc["Error"] = cc["Model"] / cc["True"] - 1
    fig = go.Figure([
        go.Bar(x=cc["Channel"], y=cc["True"], name="True", marker_color="#b7cdea",
               marker_line_color="white", marker_line_width=2, hovertemplate="%{x}: $%{y:,.0f}<extra>True</extra>"),
        go.Bar(x=cc["Channel"], y=cc["Model"], name="Model", marker_color="#1f4f8f",
               marker_line_color="white", marker_line_width=2,
               text=[f"{e:+.0%}" for e in cc["Error"]], textposition="outside",
               hovertemplate="%{x}: $%{y:,.0f}<extra>Model</extra>")])
    fig.update_layout(barmode="group", height=380, yaxis=dict(title="Revenue contribution", tickprefix="$"),
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.info("**Read-out:** Google is close. Microsoft is overcredited (+67%) because it is small and "
            "tracks Google closely. YouTube is undercredited by half. Every true ROAS still falls inside "
            "the model's 90% interval.")

# ---------------------------------------------------------------------------
# Tab: Budget simulator
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Budget simulator")
    st.markdown("Set a weekly spend for each channel and see the revenue it would really produce, "
                "using the simulation's **true** response curves (steady weekly spend). Start from a "
                "preset, then drag the sliders.")

    preset = st.radio("Start from", list(PLANS), horizontal=True, key="preset")
    if st.session_state.get("_last_preset") != preset:
        for ch in CHANNELS:
            st.session_state[f"spend_{ch}"] = int(PLANS[preset][ch])
        st.session_state["_last_preset"] = preset

    cols = st.columns(4)
    plan = {}
    for col, ch in zip(cols, CHANNELS):
        plan[ch] = col.slider(f"{LABELS[ch]} ($/week)", 0, int(CURRENT[ch] * 2.5), step=100,
                              key=f"spend_{ch}")

    cur_rev = sum(float(true_response(ch, CURRENT[ch], params, SCALE)) for ch in CHANNELS)
    new_rev = sum(float(true_response(ch, plan[ch], params, SCALE)) for ch in CHANNELS)
    cur_budget, new_budget = sum(CURRENT.values()), sum(plan.values())

    m = st.columns(4)
    m[0].metric("Weekly budget", f"${new_budget:,.0f}", delta=f"{new_budget - cur_budget:+,.0f} vs current",
                delta_color="off", delta_arrow="off")
    m[1].metric("True media revenue / week", f"${new_rev:,.0f}", delta=f"{new_rev - cur_rev:+,.0f}")
    m[2].metric("Media ROAS", f"{new_rev / max(new_budget, 1):.2f}", delta=f"current {cur_rev / cur_budget:.2f}",
                delta_color="off", delta_arrow="off")
    ann = (new_rev - cur_rev) * 52
    m[3].metric("Annualised change", f"{'+' if ann >= 0 else '−'}${abs(ann):,.0f}")
    if abs(new_budget - cur_budget) > 50:
        st.caption(f"⚠️ This plan changes the total budget by ${new_budget - cur_budget:+,.0f} a week. "
                   "To compare mixes fairly, keep the total at the current $31,141.")

    st.markdown("**Response curves:** weekly revenue at each level of steady weekly spend")
    fig = go.Figure()
    for ch in CHANNELS:
        name = LABELS[ch]
        xs = np.linspace(0, CURRENT[ch] * 2.5, 120)
        fig.add_trace(go.Scatter(x=xs, y=true_response(ch, xs, params, SCALE), name=name, mode="lines",
                                 line=dict(color=COLORS[name], width=2),
                                 hovertemplate=f"{name}<br>Spend $%{{x:,.0f}} → revenue $%{{y:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[CURRENT[ch]], y=[float(true_response(ch, CURRENT[ch], params, SCALE))],
                                 mode="markers", showlegend=False,
                                 marker=dict(size=10, color="white", line=dict(color=COLORS[name], width=2)),
                                 hovertemplate=f"{name} current: $%{{x:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[plan[ch]], y=[float(true_response(ch, plan[ch], params, SCALE))],
                                 mode="markers", showlegend=False,
                                 marker=dict(size=12, color=COLORS[name], line=dict(color="white", width=2)),
                                 hovertemplate=f"{name} your plan: $%{{x:,.0f}} → $%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(height=420, xaxis=dict(title="Weekly spend", tickprefix="$"),
                      yaxis=dict(title="Weekly revenue from channel", tickprefix="$"),
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Hollow circle = current spend. Filled dot = your plan. A flattening curve means each "
               "extra dollar buys less.")

    rows = []
    for ch in CHANNELS:
        r_cur = float(true_response(ch, CURRENT[ch], params, SCALE))
        r_new = float(true_response(ch, plan[ch], params, SCALE))
        mroas = float((true_response(ch, plan[ch] + 100, params, SCALE) - r_new) / 100)
        rows.append({"Channel": LABELS[ch], "Current spend": CURRENT[ch], "Your spend": plan[ch],
                     "Change": plan[ch] / CURRENT[ch] - 1, "True revenue (current)": r_cur,
                     "True revenue (your plan)": r_new, "Marginal ROAS at your spend": mroas})
    st.dataframe(pd.DataFrame(rows).style.format({
        "Current spend": "${:,.0f}", "Your spend": "${:,.0f}", "Change": "{:+.0%}",
        "True revenue (current)": "${:,.0f}", "True revenue (your plan)": "${:,.0f}",
        "Marginal ROAS at your spend": "{:.2f}"}), width="stretch", hide_index=True)

    with st.expander("How did the model's own recommendation do?"):
        mm = PLANS["MMM recommendation"]
        true_gain = sum(float(true_response(c, mm[c], params, SCALE)) for c in CHANNELS) - cur_rev
        opt = PLANS["True optimum"]
        best_gain = sum(float(true_response(c, opt[c], params, SCALE)) for c in CHANNELS) - cur_rev
        st.markdown(f"""
- The model expected **+${MODEL_UPLIFT['expected']:,}/week** (90% interval −${abs(MODEL_UPLIFT['lo']):,} to
  +${MODEL_UPLIFT['hi']:,}), with a {MODEL_UPLIFT['prob_better']:.0%} probability of beating the current plan.
- Against the true curves, its plan gains **+${true_gain:,.0f}/week**, which is
  **{true_gain / best_gain:.0%}** of the best achievable (+${best_gain:,.0f}).
- **Lesson:** trust the *direction* of an MMM recommendation more than its *size*. Keep guardrails
  (here ±50% per channel) and roll changes out in stages.
""")
        mt = MARGINAL.copy()
        mt["channel"] = mt["channel"].map(LABELS)
        st.dataframe(mt.rename(columns={"channel": "Channel", "weekly_spend": "Weekly spend",
                                        "avg_roas": "Avg ROAS (model)", "marginal_roas": "Marginal ROAS (model)",
                                        "marginal_lo": "5%", "marginal_hi": "95%",
                                        "true_marginal": "True marginal ROAS"})
                     .style.format({"Weekly spend": "${:,.0f}"}, precision=2), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab: Carryover explorer
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Carryover (adstock): why YouTube looks useless week by week")
    st.markdown("A channel's effect doesn't stop when spend stops. Pick a channel and a carryover rate "
                "(alpha) to see how one week's spend spreads into the following weeks.")
    c1, c2 = st.columns([1, 2])
    with c1:
        ch = st.selectbox("Channel", CHANNELS, index=3, format_func=LABELS.get)
        a = ADSTOCK.set_index("channel").loc[ch]
        alpha = st.slider("Carryover rate (alpha)", 0.0, 0.95, float(a["true"]), 0.05)
        st.markdown(f"""
| | alpha |
|---|---|
| True | **{a['true']:.2f}** |
| Model estimate | {a['mean']:.2f} ({a['lo']:.2f}–{a['hi']:.2f}) |
""")
        st.caption("Short for search, medium for social, long for video. Giving video a longer carryover "
                   "prior fixed a badly underestimated YouTube effect.")
    with c2:
        x = df[ch].to_numpy() / SCALE[ch]
        ad = geometric_adstock(x, alpha)
        fig = go.Figure([
            go.Bar(x=df["date"], y=x, name="Spend (scaled)", marker_color="#d6d4ce",
                   hovertemplate="Spend: %{y:.2f}<extra></extra>"),
            go.Scatter(x=df["date"], y=ad, name=f"Adstocked effect (alpha {alpha:.2f})", mode="lines",
                       line=dict(color=COLORS[LABELS[ch]], width=2),
                       hovertemplate="Effect: %{y:.2f}<extra></extra>")])
        fig.update_layout(height=320, hovermode="x unified", yaxis_title="Share of peak spend",
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=10), bargap=0.1)
        st.plotly_chart(fig, width="stretch")
        w = alpha ** np.arange(8)
        w = w / w.sum()
        fig = px.bar(x=[f"Week {i}" for i in range(8)], y=w, labels={"x": "", "y": "Share of effect"},
                     color_discrete_sequence=[COLORS[LABELS[ch]]], text_auto=".0%")
        fig.update_traces(hovertemplate="%{x}: %{y:.0%}<extra></extra>")
        fig.update_layout(height=240, yaxis_tickformat=".0%", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Bottom: how one week of spend is spread over the next 8 weeks (week 0 = the week of spend).")

# ---------------------------------------------------------------------------
# Tab: Geo-lift calibration
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("One geo-lift test fixed Microsoft's estimate")
    st.markdown("A simulated two-cell test on Microsoft over 8 weeks: **go dark** (spend off in test regions) "
                "and **double** (spend doubled). The results were added to the model as a likelihood term.")
    g = st.columns(4)
    g[0].metric("Go dark: measured lift / week", "−$7.9k", help="Model predicted −$13.0k before calibration")
    g[1].metric("Go dark: incremental ROAS", "3.93")
    g[2].metric("Double: measured lift / week", "+$3.3k", help="Model predicted +$4.6k before calibration")
    g[3].metric("Double: incremental ROAS", "1.62")

    cal = CALIBRATION.copy()
    cal["Channel"] = cal["channel"].map(LABELS)
    fig = go.Figure()
    for col, name, color, offset in [("before", "Before calibration", "#b7cdea", -0.15),
                                     ("after", "After calibration", "#1f4f8f", 0.15)]:
        fig.add_trace(go.Scatter(
            x=np.arange(4) + offset, y=cal[f"{col}_mean"], mode="markers", name=name,
            marker=dict(size=12, color=color, line=dict(color="white", width=2)),
            error_y=dict(type="data", array=cal[f"{col}_width"] / 2, color=color, thickness=2),
            customdata=cal[["Channel", f"{col}_width"]],
            hovertemplate="%{customdata[0]}<br>ROAS %{y:.2f}<br>90% width %{customdata[1]:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.arange(4), y=cal["true_roas"], mode="markers", name="True ROAS",
                             marker=dict(symbol="diamond", size=13, color="#eb6834",
                                         line=dict(color="white", width=2)),
                             hovertemplate="True ROAS %{y:.2f}<extra></extra>"))
    fig.update_layout(height=420, xaxis=dict(tickvals=list(range(4)), ticktext=cal["Channel"].tolist()),
                      yaxis_title="ROAS", legend=dict(orientation="h", y=-0.15), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Error bars show the width of the 90% interval, centred on the mean (an approximation: "
               "the real posterior intervals are skewed).")
    st.dataframe(cal[["Channel", "true_roas", "before_mean", "before_width", "after_mean", "after_width"]]
                 .rename(columns={"true_roas": "True ROAS", "before_mean": "Before", "before_width": "Before 90% width",
                                  "after_mean": "After", "after_width": "After 90% width"})
                 .style.format(precision=2), width="stretch", hide_index=True)
    st.info("**Read-out:** Microsoft's ROAS moved from 6.68 to 3.89 (true 3.99) and its interval narrowed 8x. "
            "Google improved too, because the test settled how search credit splits between the two. "
            "Meta and YouTube are now the main source of error, so the next test should cover one of them.")

# ---------------------------------------------------------------------------
# Tab: Model checks
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Model health")
    st.dataframe(pd.DataFrame([
        ("Divergences", "0"), ("Max r̂", "1.011"), ("Minimum ESS", "743"),
        ("In-sample R² / MAPE", "0.92 / 3.6%"),
        ("Holdout MAPE (last 13 weeks)", "3.8%, all 13 actuals inside the 90% interval"),
        ("Price increase effect", "about −$31k/week (true −$30k)"),
        ("Promo week effect", "about +$36k/week (true +$35k)"),
    ], columns=["Check", "Result"]), width="stretch", hide_index=True)
    for fname, cap in [("06_model_fit.png", "In-sample fit"),
                       ("10_holdout_forecast.png", "13-week out-of-sample forecast")]:
        if (FIG_DIR / fname).exists():
            st.image(str(FIG_DIR / fname), caption=cap, width="stretch")

with st.expander("About this dashboard"):
    st.markdown(f"""
- **Data:** 156 simulated weeks (2023–2025) with known adstock, saturation, seasonality, promos and a
  10% price increase (`src/simulate_data.py`).
- **Model:** Bayesian MMM in PyMC-Marketing: geometric adstock → logistic saturation, controls and yearly
  seasonality, fitted with nutpie. Refitting takes minutes, so this page shows the published results.
- **Simulator:** uses the true response curves at a steady weekly spend. YouTube actually runs in bursts,
  so its steady-state curve is an approximation.
- **Code:** [{REPO_URL}]({REPO_URL})
""")
