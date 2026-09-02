"""
Analysis for "Dynamic Pricing for the Mandi" / MandiBid Forward.

1. EDA: price levels, volatility, seasonality, market-to-market spread
   (this quantifies the information-asymmetry problem itself)
2. Feature engineering: lag prices, rolling stats, day-of-week, day-of-year
3. Models compared:
     - Naive baseline (today's price = tomorrow's forecast)
     - ARIMA (classical time-series baseline used in literature)
     - Gradient Boosting Regressor (ML baseline; stand-in for the
       LSTM/GRU results reported in the literature, which needs a
       deep-learning stack — see report for why GBM is a reasonable
       proxy at this project's scale)
4. Output: a predicted next-day price BAND (not point estimate) per
   market — this is the actual deliverable "MandiBid Forward" ships to
   farmers/traders.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import load_data


def engineer_features(df, market_name):
    d = df[df["market"] == market_name].copy()
    d = d.groupby("arrival_date", as_index=False)["modal_price"].mean()
    d = d.sort_values("arrival_date").reset_index(drop=True)

    for lag in [1, 2, 3, 7, 14]:
        d[f"lag_{lag}"] = d["modal_price"].shift(lag)
    d["roll_mean_7"] = d["modal_price"].shift(1).rolling(7).mean()
    d["roll_std_7"] = d["modal_price"].shift(1).rolling(7).std()
    d["roll_mean_30"] = d["modal_price"].shift(1).rolling(30).mean()
    d["dow"] = d["arrival_date"].dt.dayofweek
    d["doy"] = d["arrival_date"].dt.dayofyear
    d["month"] = d["arrival_date"].dt.month

    d = d.dropna().reset_index(drop=True)
    return d


def train_test_split_time(d, test_days=90):
    train = d.iloc[:-test_days]
    test = d.iloc[-test_days:]
    return train, test


def naive_baseline(test):
    pred = test["lag_1"].values
    actual = test["modal_price"].values
    return pred, actual


def arima_baseline(train, test):
    from statsmodels.tsa.arima.model import ARIMA
    history = list(train["modal_price"])
    preds = []
    for actual in test["modal_price"]:
        model = ARIMA(history, order=(2, 1, 2))
        fit = model.fit()
        pred = fit.forecast(1)[0]
        preds.append(pred)
        history.append(actual)  # walk-forward validation
    return np.array(preds), test["modal_price"].values


def gbm_model(train, test):
    features = ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
                "roll_mean_7", "roll_std_7", "roll_mean_30", "dow", "doy", "month"]
    X_train, y_train = train[features], train["modal_price"]
    X_test, y_test = test[features], test["modal_price"]

    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test.values, model, features


def evaluate(name, pred, actual):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    print(f"{name:20s}  MAE={mae:8.1f}  RMSE={rmse:8.1f}  MAPE={mape:6.2f}%")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape}


def price_band(pred_point, residual_std, z=1.0):
    """Turn a point forecast into a negotiation-ready band (what MandiBid
    Forward actually shows a farmer/trader)."""
    return pred_point - z * residual_std, pred_point + z * residual_std


if __name__ == "__main__":
    df, source = load_data()
    print(f"Data source: {source}\n")

    market = "Lasalgaon"
    d = engineer_features(df, market)
    train, test = train_test_split_time(d, test_days=90)

    print(f"=== Forecasting: {market} onion modal price (next-day) ===")
    print(f"Train: {len(train)} days | Test: {len(test)} days\n")

    results = []

    pred_naive, actual = naive_baseline(test)
    results.append(evaluate("Naive (t-1)", pred_naive, actual))

    pred_arima, actual = arima_baseline(train, test)
    results.append(evaluate("ARIMA(2,1,2)", pred_arima, actual))

    pred_gbm, actual, model, features = gbm_model(train, test)
    results.append(evaluate("Gradient Boosting", pred_gbm, actual))

    print("\n=== Feature importance (Gradient Boosting) ===")
    importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    for f, imp in importances:
        print(f"  {f:15s} {imp:.3f}")

    # Build the actual product output: price band for the last test day
    resid_std = np.std(actual - pred_gbm)
    last_pred = pred_gbm[-1]
    lo, hi = price_band(last_pred, resid_std)
    print(f"\n=== MandiBid Forward output (example, {market}) ===")
    print(f"Predicted next-day modal price: Rs. {last_pred:.0f}/quintal")
    print(f"Recommended negotiation band (±1 std): Rs. {lo:.0f} - Rs. {hi:.0f}/quintal")
    print(f"Actual price that materialized: Rs. {actual[-1]:.0f}/quintal")

    pd.DataFrame(results).to_csv("model_comparison.csv", index=False)
    d.to_csv("engineered_features.csv", index=False)
    print("\nSaved model_comparison.csv and engineered_features.csv")
