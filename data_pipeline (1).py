"""
Data pipeline for the "Dynamic Pricing for the Mandi" project.

Source dataset: "Current Daily Price of Various Commodities from Various
Markets (Mandi)" — Directorate of Marketing & Inspection (DMI), Ministry of
Agriculture & Farmers Welfare, published on data.gov.in via the Agmarknet
portal. Resource ID: 9ef84268-d588-465a-a308-a864a43d0070

Schema (confirmed from data.gov.in API metadata):
  state, district, market, commodity, variety, grade,
  arrival_date (dd/mm/yyyy), min_price, max_price, modal_price
  (prices in Rs. per quintal)

HOW TO GET A REAL KEY (2 minutes, free):
  1. Go to https://data.gov.in and register (Google/gov email works).
  2. Go to "My Account" -> "Generate API Key".
  3. Paste it into API_KEY below and set USE_LIVE = True.

Until then, this script falls back to a SYNTHETIC generator that reproduces
the same schema and realistic price dynamics (grounded in published
literature on onion price volatility in Nashik-belt Maharashtra mandis —
see references in the analysis notebook / report), so every downstream
script (EDA, modeling, Streamlit dashboard) runs unmodified once you swap
in real data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

# ---- fill this in with YOUR OWN key from data.gov.in for live data ----
API_KEY = "PASTE_YOUR_KEY_HERE"
USE_LIVE = False
# -------------------------------------------------------------------


def pull_live(commodity="Onion", state="Maharashtra", max_records=5000, page_size=1000):
    """Paginate through the real Agmarknet API. Requires your own API_KEY."""
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        raise ValueError("Set your own API_KEY (free, from data.gov.in) before calling pull_live().")

    records = []
    offset = 0
    while offset < max_records:
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": page_size,
            "offset": offset,
            "filters[commodity]": commodity,
            "filters[state]": state,
        }
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        batch = data.get("records", [])
        if not batch:
            break
        records.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break

    df = pd.DataFrame(records)
    return _clean(df)


def _clean(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["modal_price", "arrival_date"])
    return df.sort_values("arrival_date").reset_index(drop=True)


def generate_synthetic(start="2022-01-01", end="2024-12-31", seed=42):
    """
    Realistic synthetic stand-in for the Agmarknet onion dataset, covering
    4 markets in Maharashtra's onion belt. Grounded in real dynamics:
      - strong seasonality (kharif/rabi harvest troughs, monsoon-onset spikes)
      - the well-documented Aug-Nov onion price spike pattern
      - long right tail (occasional shortage spikes to 3-4x baseline)
      - noisy day-to-day arrivals affecting min/max spread
    Matches Agmarknet's exact column schema so it's a drop-in replacement.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    markets = [
        ("Maharashtra", "Nashik", "Lasalgaon"),
        ("Maharashtra", "Nashik", "Pimpalgaon"),
        ("Maharashtra", "Nashik", "Nashik"),
        ("Maharashtra", "Solapur", "Solapur"),
    ]

    rows = []
    for state, district, market in markets:
        base = 1400 + rng.normal(0, 100)  # base modal price, Rs/quintal
        market_offset = rng.normal(0, 150)
        for d in dates:
            doy = d.dayofyear
            # seasonal component: trough around harvest (Mar-May & Oct-Dec), spike Aug-Oct pre-kharif
            seasonal = 500 * np.sin(2 * np.pi * (doy - 200) / 365) + 300 * np.sin(4 * np.pi * doy / 365)
            trend = 0.15 * (d - dates[0]).days  # mild multi-year inflation drift
            noise = rng.normal(0, 120)
            # occasional shortage shock (autocorrelated via day-of-year clustering around Sep-Oct)
            shock = 0
            if 240 <= doy <= 300 and rng.random() < 0.03:
                shock = rng.uniform(800, 2500)
            modal = max(300, base + market_offset + seasonal + trend + noise + shock)
            spread = modal * rng.uniform(0.08, 0.22)
            rows.append({
                "state": state,
                "district": district,
                "market": market,
                "commodity": "Onion",
                "variety": "Other" if rng.random() > 0.5 else "Local",
                "grade": rng.choice(["FAQ", "Medium", "Local"], p=[0.5, 0.3, 0.2]),
                "arrival_date": d,
                "min_price": round(modal - spread, 1),
                "max_price": round(modal + spread, 1),
                "modal_price": round(modal, 1),
            })

    df = pd.DataFrame(rows)
    return df.sort_values("arrival_date").reset_index(drop=True)


def load_data():
    if USE_LIVE:
        try:
            return pull_live(), "live"
        except Exception as e:
            print(f"[warn] live pull failed ({e}); falling back to synthetic data.")
    return generate_synthetic(), "synthetic"


if __name__ == "__main__":
    df, source = load_data()
    print(f"Loaded {len(df)} rows from {source} source.")
    print(df.head())
    df.to_csv("mandi_onion_data.csv", index=False)
    print("Saved to mandi_onion_data.csv")
