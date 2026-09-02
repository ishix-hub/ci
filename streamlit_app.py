"""
MandiBid Forward — Streamlit dashboard
Run with: streamlit run streamlit_app.py

Two views:
  - Farmer/Trader view: today's price, predicted next-day BAND, whether
    to sell now or wait (the actual product surface)
  - Analyst view: market comparison, volatility, model performance
    (what you'd show in your project report / demo video)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor

from data_pipeline import load_data
from analysis import engineer_features, train_test_split_time, gbm_model, price_band

st.set_page_config(page_title="MandiBid Forward", layout="wide")

st.title("🧅 MandiBid Forward")
st.caption("AI-predicted price bands + pre-commitment contracts for mandi onion trade")

df, source = load_data()
if source == "synthetic":
    st.info(
        "Running on a **synthetic dataset** that reproduces Agmarknet's exact schema and "
        "realistic price dynamics for demo purposes. Set `USE_LIVE = True` and add your free "
        "data.gov.in API key in `data_pipeline.py` to run this on real mandi data.",
        icon="ℹ️",
    )

markets = sorted(df["market"].unique())

view = st.sidebar.radio("View", ["Farmer / Trader view", "Analyst view"])
market = st.sidebar.selectbox("Market", markets, index=markets.index("Lasalgaon") if "Lasalgaon" in markets else 0)

d = engineer_features(df, market)
train, test = train_test_split_time(d, test_days=90)
pred, actual, model, features = gbm_model(train, test)
resid_std = np.std(actual - pred)

if view == "Farmer / Trader view":
    st.subheader(f"📍 {market} — Today's Recommendation")

    latest_actual = actual[-1]
    latest_pred = pred[-1]
    lo, hi = price_band(latest_pred, resid_std)

    c1, c2, c3 = st.columns(3)
    c1.metric("Today's modal price", f"₹{latest_actual:,.0f}/quintal")
    c2.metric("Predicted tomorrow", f"₹{latest_pred:,.0f}/quintal",
              delta=f"{latest_pred - latest_actual:+.0f}")
    c3.metric("Negotiation band", f"₹{lo:,.0f} – ₹{hi:,.0f}")

    if latest_pred > latest_actual * 1.03:
        st.success("📈 Prices are predicted to rise. Consider **holding** if storage allows, "
                   "or lock a forward contract at today's band before it moves.")
    elif latest_pred < latest_actual * 0.97:
        st.warning("📉 Prices are predicted to fall. Consider **selling today** rather than waiting.")
    else:
        st.info("➡️ Prices are expected to stay roughly stable. Either timing works — "
                "focus on securing a good buyer within today's band.")

    st.markdown("#### Lock a forward contract")
    qty = st.number_input("Quantity (quintals)", min_value=1, value=50)
    lock_days = st.slider("Lock price for how many days ahead?", 1, 5, 1)
    locked_price = latest_pred  # in a real product this widens with lock_days & volatility
    st.write(
        f"Locking **{qty} quintals** at **₹{locked_price:,.0f}/quintal** "
        f"(±₹{resid_std:.0f} band) for delivery in {lock_days} day(s) would guarantee "
        f"**₹{qty*locked_price:,.0f}** total, removing same-day negotiation risk for both sides."
    )
    st.caption("This is the MandiBid Forward core product: the predictive band de-risks the "
               "pre-commitment contract that farmers and traders can both agree to ahead of time.")

    st.markdown("#### Recent price trend")
    recent = d.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["arrival_date"], y=recent["modal_price"],
                              mode="lines+markers", name="Modal price"))
    fig.update_layout(yaxis_title="₹/quintal", xaxis_title="Date", height=350)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("📊 Analyst View")

    tab1, tab2, tab3 = st.tabs(["Market Comparison", "Model Performance", "Volatility (the problem, quantified)"])

    with tab1:
        st.markdown("**Same commodity, same day, different mandis — the arbitrage AI can flag**")
        latest_date = df["arrival_date"].max()
        snapshot = df[df["arrival_date"] == latest_date][["market", "modal_price"]].sort_values("modal_price")
        fig = px.bar(snapshot, x="market", y="modal_price", title=f"Modal price by market — {latest_date.date()}")
        st.plotly_chart(fig, use_container_width=True)
        spread = snapshot["modal_price"].max() - snapshot["modal_price"].min()
        st.metric("Price spread across markets today", f"₹{spread:,.0f}/quintal",
                   help="This is the information-asymmetry gap a farmer with no cross-market visibility loses.")

    with tab2:
        st.markdown("**Forecast accuracy: next-day modal price**")
        results_df = pd.DataFrame({
            "date": test["arrival_date"].values,
            "actual": actual,
            "predicted": pred,
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df["date"], y=results_df["actual"], name="Actual"))
        fig.add_trace(go.Scatter(x=results_df["date"], y=results_df["predicted"], name="Predicted (GBM)"))
        fig.update_layout(yaxis_title="₹/quintal", height=350)
        st.plotly_chart(fig, use_container_width=True)

        mae = np.mean(np.abs(actual - pred))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        c1, c2 = st.columns(2)
        c1.metric("MAE (test set)", f"₹{mae:,.0f}")
        c2.metric("MAPE (test set)", f"{mape:.1f}%")
        st.caption(
            "Published literature on onion price forecasting reports LSTM/GRU MAPE around "
            "10-15% and ARIMA MAPE well above that on volatile years (Nature Sci Reports, 2025). "
            "A gradient-boosted baseline in this range is a reasonable, lightweight stand-in — "
            "full deep-learning models are the natural next iteration, noted as a limitation below."
        )

    with tab3:
        st.markdown("**Why this problem exists: price volatility farmers face without forecasting**")
        d["pct_change"] = d["modal_price"].pct_change() * 100
        fig = px.histogram(d, x="pct_change", nbins=60,
                            title="Distribution of day-over-day price changes (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Std. dev of daily price change", f"{d['pct_change'].std():.1f}%")
