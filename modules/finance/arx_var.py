"""
ARX/VAR-specific visualization and utility functions for financial modeling notebooks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# --- Helper: Convert ARX selection DataFrame to best_lags dict ---
def best_lags_from_df(sel_df_arx):
    """
    Given a DataFrame with columns ['asset', 'p', ...], returns a dict {asset: p} for best lag per asset.
    Expects sel_df_arx to be indexed by asset or have an 'asset' column.
    """
    if 'asset' in sel_df_arx.columns:
        return {row['asset']: int(row['p']) for _, row in sel_df_arx.iterrows()}
    else:
        # Indexed by asset
        return {idx: int(row['p']) for idx, row in sel_df_arx.iterrows()}

# --- Shared ARX expanding window forecast helper ---
def _run_arx_expanding_forecast(data, asset, p, control_cols, min_train=60):
    cols_lag = [f"{asset}_l{lag}" for lag in range(1, p + 1)]
    cols_exog = cols_lag + control_cols
    df_fit = data.dropna(subset=[asset] + cols_exog)
    y_all = df_fit[asset]
    X_all = sm.add_constant(df_fit[cols_exog], has_constant="add")
    preds, actuals, dates = [], [], []
    for i in range(min_train, len(df_fit)):
        y_train = y_all.iloc[:i]
        X_train = X_all.iloc[:i]
        m = sm.OLS(y_train, X_train).fit()
        x_next = sm.add_constant(df_fit[cols_exog].iloc[i:i+1], has_constant="add")
        pred = m.predict(x_next).iloc[0]
        preds.append(pred)
        actuals.append(y_all.iloc[i])
        dates.append(df_fit.index[i])
    return np.array(preds), np.array(actuals), np.array(dates)

# --- ARX: Predictions vs Actuals Plot ---
def plot_arx_predictions(data, assets, best_lags, control_cols, min_train=60, figsize=(15,4)):
    """
    Plot predictions vs actuals for ARX models for each asset.
    Returns (fig, summary_df) with RMSE, MAE, dir_acc per asset.
    """
    fig, axes = plt.subplots(1, len(assets), figsize=figsize)
    summary = []
    for idx, asset in enumerate(assets):
        p = best_lags[asset]
        preds, actuals, dates = _run_arx_expanding_forecast(data, asset, p, control_cols, min_train)
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mae = np.mean(np.abs(preds - actuals))
        dir_acc = np.mean(np.sign(preds) == np.sign(actuals)) * 100
        summary.append({"asset": asset, "rmse": rmse, "mae": mae, "dir_acc": dir_acc})
        ax = axes[idx] if len(assets) > 1 else axes
        ax.plot(dates, actuals, label="Actual", alpha=0.7)
        ax.plot(dates, preds, label="Forecast", alpha=0.7)
        ax.set_title(f"{asset} (p={p})\nRMSE={rmse:.4f}, DirAcc={dir_acc:.1f}%")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, pd.DataFrame(summary)

# --- ARX: Cumulative Returns Plot ---
def plot_arx_cumulative_returns(data, assets, best_lags, control_cols, min_train=60, figsize=(15,4)):
    """
    Plot cumulative returns for ARX strategy (long if pred>0, short if pred<0) vs buy-and-hold.
    Returns (fig, summary_df) with final returns per asset.
    """
    fig, axes = plt.subplots(1, len(assets), figsize=figsize)
    summary = []
    for idx, asset in enumerate(assets):
        p = best_lags[asset]
        preds, actuals, dates = _run_arx_expanding_forecast(data, asset, p, control_cols, min_train)
        # Strategy: long if pred>0, short if pred<0
        strat_returns = np.sign(preds) * actuals
        bh_returns = actuals
        strat_cum = np.cumsum(strat_returns)
        bh_cum = np.cumsum(bh_returns)
        ax = axes[idx] if len(assets) > 1 else axes
        ax.plot(dates, strat_cum, label="Strategy", color="blue")
        ax.plot(dates, bh_cum, label="Buy & Hold", color="gray", linestyle="--")
        ax.set_title(f"{asset} (p={p})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        summary.append({"asset": asset, "strategy_final": strat_cum[-1], "buyhold_final": bh_cum[-1]})
    plt.tight_layout()
    return fig, pd.DataFrame(summary)

# --- ARX: Rolling RMSE Plot (Overfitting Diagnostic) ---
def plot_arx_rolling_rmse(data, assets, best_lags, control_cols, min_train=60, window=30, figsize=(15,4)):
    """
    Plot rolling RMSE for ARX models to visualize convergence/overfitting.
    Returns (fig, summary_df) with mean/median rolling RMSE per asset.
    """
    fig, axes = plt.subplots(1, len(assets), figsize=figsize)
    summary = []
    for idx, asset in enumerate(assets):
        p = best_lags[asset]
        preds, actuals, dates = _run_arx_expanding_forecast(data, asset, p, control_cols, min_train)
        rolling_rmse = pd.Series((preds - actuals) ** 2, index=dates).rolling(window).apply(lambda x: np.sqrt(np.mean(x)))
        ax = axes[idx] if len(assets) > 1 else axes
        ax.plot(rolling_rmse.index, rolling_rmse.values, label=f"Rolling RMSE ({window}d)", color="purple")
        ax.set_title(f"{asset} (p={p})")
        ax.set_xlabel("Date")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        summary.append({"asset": asset, "mean_rmse": np.nanmean(rolling_rmse), "median_rmse": np.nanmedian(rolling_rmse)})
    plt.tight_layout()
    return fig, pd.DataFrame(summary)

# --- VARX expanding window forecast helper ---
def _run_var_expanding_forecast(Y, X_exog, p, min_train=60):
    # Y: DataFrame of targets (n_obs, n_assets)
    # X_exog: DataFrame of exogenous controls (n_obs, n_controls)
    Y = Y.dropna()
    X_exog = X_exog.loc[Y.index]
    preds, actuals, dates = [], [], []
    for i in range(min_train, len(Y)):
        y_train = Y.iloc[:i]
        x_train = X_exog.iloc[:i]
        model = VAR(y_train, exog=x_train).fit(p, trend="c")
        y_last = y_train.values[-model.k_ar:]
        x_future = X_exog.iloc[i:i+1].values
        pred = model.forecast(y_last, steps=1, exog_future=x_future)[0]
        preds.append(pred)
        actuals.append(Y.iloc[i].values)
        dates.append(Y.index[i])
    preds = np.array(preds)  # shape (n_forecasts, n_assets)
    actuals = np.array(actuals)
    return preds, actuals, np.array(dates)

# --- VARX: Predictions vs Actuals Plot ---
def plot_var_predictions(Y, X_exog, best_p, min_train=60, figsize=(15,4)):
    """
    Plot predictions vs actuals for VARX model for each asset.
    Returns (fig, summary_df) with RMSE, MAE, dir_acc per asset.
    """
    assets = list(Y.columns)
    preds, actuals, dates = _run_var_expanding_forecast(Y, X_exog, best_p, min_train)
    fig, axes = plt.subplots(1, len(assets), figsize=figsize)
    summary = []
    for idx, asset in enumerate(assets):
        pred = preds[:, idx]
        act = actuals[:, idx]
        rmse = np.sqrt(np.mean((pred - act) ** 2))
        mae = np.mean(np.abs(pred - act))
        dir_acc = np.mean(np.sign(pred) == np.sign(act)) * 100
        summary.append({"asset": asset, "rmse": rmse, "mae": mae, "dir_acc": dir_acc})
        ax = axes[idx] if len(assets) > 1 else axes
        ax.plot(dates, act, label="Actual", alpha=0.7)
        ax.plot(dates, pred, label="Forecast", alpha=0.7)
        ax.set_title(f"{asset} (p={best_p})\nRMSE={rmse:.4f}, DirAcc={dir_acc:.1f}%")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, pd.DataFrame(summary)

# --- VARX: Cumulative Returns Plot ---
def plot_var_cumulative_returns(Y, X_exog, best_p, min_train=60, figsize=(15,4)):
    """
    Plot cumulative returns for VARX strategy (long if pred>0, short if pred<0) vs buy-and-hold.
    Returns (fig, summary_df) with final returns per asset.
    """
    assets = list(Y.columns)
    preds, actuals, dates = _run_var_expanding_forecast(Y, X_exog, best_p, min_train)
    fig, axes = plt.subplots(1, len(assets), figsize=figsize)
    summary = []
    for idx, asset in enumerate(assets):
        pred = preds[:, idx]
        act = actuals[:, idx]
        strat_returns = np.sign(pred) * act
        bh_returns = act
        strat_cum = np.cumsum(strat_returns)
        bh_cum = np.cumsum(bh_returns)
        ax = axes[idx] if len(assets) > 1 else axes
        ax.plot(dates, strat_cum, label="Strategy", color="blue")
        ax.plot(dates, bh_cum, label="Buy & Hold", color="gray", linestyle="--")
        ax.set_title(f"{asset} (p={best_p})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        summary.append({"asset": asset, "strategy_final": strat_cum[-1], "buyhold_final": bh_cum[-1]})
    plt.tight_layout()
    return fig, pd.DataFrame(summary)

# --- VARX: Rolling RMSE Plot (Overfitting Diagnostic) ---
def plot_var_rolling_rmse(Y, X_exog, best_p, min_train=60, window=30, figsize=(15,4)):
    """
    Plot rolling RMSE for VARX model to visualize convergence/overfitting.
    Returns (fig, summary_df) with mean/median rolling RMSE per asset.
    """
    assets = list(Y.columns)
    preds, actuals, dates = _run_var_expanding_forecast(Y, X_exog, best_p, min_train)
    fig, axes = plt.subplots(1, len(assets), figsize=figsize)
    summary = []
    for idx, asset in enumerate(assets):
        pred = preds[:, idx]
        act = actuals[:, idx]
        rolling_rmse = pd.Series((pred - act) ** 2, index=dates).rolling(window).apply(lambda x: np.sqrt(np.mean(x)))
        ax = axes[idx] if len(assets) > 1 else axes
        ax.plot(rolling_rmse.index, rolling_rmse.values, label=f"Rolling RMSE ({window}d)", color="purple")
        ax.set_title(f"{asset} (p={best_p})")
        ax.set_xlabel("Date")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        summary.append({"asset": asset, "mean_rmse": np.nanmean(rolling_rmse), "median_rmse": np.nanmedian(rolling_rmse)})
    plt.tight_layout()
    return fig, pd.DataFrame(summary)

# (VAR equivalents would be implemented similarly)
