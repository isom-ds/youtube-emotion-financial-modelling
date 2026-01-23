import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Literal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error


# ----------------------------
# 1) Trading-day alignment for weekend-rich socials
# ----------------------------
Agg = Literal["mean", "sum", "last"]

def map_to_next_trading_day(
    df: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    agg: Agg = "mean",
) -> pd.DataFrame:
    """
    Map each row in df (which may include weekends/holidays) to the NEXT trading day
    in trading_index (or same day if it's already a trading day). Then aggregate.

    This keeps the native trading-day calendar (no synthetic weekend rows),
    while rolling weekend/holiday social signal into the next trading session.
    """
    if df is None or df.empty:
        return df

    df = df.sort_index()
    ti = pd.DatetimeIndex(trading_index).sort_values()

    # Assign each timestamp to the next trading day
    pos = ti.searchsorted(df.index, side="left")
    valid = pos < len(ti)

    mapped = df.loc[valid].copy()
    mapped["_trading_day_"] = ti[pos[valid]].values

    # Aggregate multiple source days (e.g., Sat+Sun+Mon) into one trading day bucket
    if agg == "mean":
        out = mapped.groupby("_trading_day_").mean(numeric_only=True)
    elif agg == "sum":
        out = mapped.groupby("_trading_day_").sum(numeric_only=True)
    elif agg == "last":
        out = mapped.groupby("_trading_day_").last()
    else:
        raise ValueError("agg must be one of: 'mean', 'sum', 'last'")

    out.index = pd.DatetimeIndex(out.index)
    out = out.reindex(ti)  # keep exactly the trading calendar
    return out


def forward_fill_with_flag(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forward-fill within existing rows only, plus a boolean flag per column indicating
    where values were filled (originally missing).
    """
    if df is None or df.empty:
        return df, df

    missing = df.isna()
    filled = df.ffill()
    flags = missing.astype(int).add_suffix("_was_filled")
    return filled, flags


# ----------------------------
# 2) Rolling / walk-forward CV
# ----------------------------
def rolling_cv_splits(n_samples: int, initial_train_size: int, step_size: int, test_size: int):
    splits = []
    train_end = initial_train_size
    while train_end + test_size <= n_samples:
        tr = np.arange(0, train_end)
        te = np.arange(train_end, train_end + test_size)
        splits.append((tr, te))
        train_end += step_size
    return splits


# ----------------------------
# 3) Group-aware ARX builder with group-specific lag0 control
# ----------------------------
@dataclass
class LagSpec:
    ar_p: int
    q_social: int
    q_controls: int
    q_cross: int


def _add_lags(
    out: pd.DataFrame,
    df: pd.DataFrame,
    cols: List[str],
    max_lag: int,
    include_lag0: bool,
    prefix: str,
):
    if df is None or df.empty or not cols or max_lag < 0:
        return
    for c in cols:
        if include_lag0:
            out[f"{prefix}{c}_lag0"] = df[c]
        for lag in range(1, max_lag + 1):
            out[f"{prefix}{c}_lag{lag}"] = df[c].shift(lag)


def make_group_arx_supervised(
    y: pd.Series,
    social: Optional[pd.DataFrame],
    controls: Optional[pd.DataFrame],
    cross: Optional[pd.DataFrame],
    lags: LagSpec,
    horizon: int = 1,
    include_lag0_social: bool = True,
    include_lag0_controls: bool = False,
    include_lag0_cross: bool = False,
):
    """
    Target is y(t+horizon). With horizon=1, that's next trading day's return.

    Key design choice (recommended):
      - socials: allow lag0 (signal dated t predicts t+1)
      - controls: lagged-only (avoid subtle simultaneity / reporting timing issues)
      - cross-assets: lagged-only (avoid mechanical contemporaneous correlations)
    """
    y = y.astype(float)

    feat = pd.DataFrame(index=y.index)

    # AR terms: past returns only
    for lag in range(1, lags.ar_p + 1):
        feat[f"y_lag{lag}"] = y.shift(lag)

    # X groups
    if social is not None:
        _add_lags(feat, social, list(social.columns), lags.q_social, include_lag0_social, "soc_")
    if controls is not None:
        _add_lags(feat, controls, list(controls.columns), lags.q_controls, include_lag0_controls, "ctl_")
    if cross is not None:
        _add_lags(feat, cross, list(cross.columns), lags.q_cross, include_lag0_cross, "xas_")

    target = y.shift(-horizon).rename("target")
    data = pd.concat([feat, target], axis=1).dropna()

    X_sup = data.drop(columns=["target"])
    y_sup = data["target"]
    return X_sup, y_sup


# ----------------------------
# 4) AutoML (Optuna) over lags + model hyperparams
# ----------------------------
def automl_group_arx(
    y: pd.Series,
    social: Optional[pd.DataFrame],
    controls: Optional[pd.DataFrame],
    cross: Optional[pd.DataFrame],
    horizon: int = 1,
    initial_train_size: int = 60,
    step_size: int = 5,
    test_size: int = 5,
    n_trials: int = 60,
    random_seed: int = 42,
    p_range=(1, 5),
    q_social_range=(0, 3),
    q_controls_range=(0, 5),
    q_cross_range=(0, 1),
    model_family="elasticnet",  # "elasticnet" or "ridge"
    include_lag0_social: bool = True,
    include_lag0_controls: bool = False,
    include_lag0_cross: bool = False,
):
    import optuna

    def build_model(params: Dict):
        if model_family == "ridge":
            return Ridge(alpha=params["alpha"], random_state=random_seed)
        return ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            random_state=random_seed,
            max_iter=20000,
        )

    def cv_score(lags: LagSpec, params: Dict) -> float:
        X_sup, y_sup = make_group_arx_supervised(
            y=y, social=social, controls=controls, cross=cross,
            lags=lags, horizon=horizon,
            include_lag0_social=include_lag0_social,
            include_lag0_controls=include_lag0_controls,
            include_lag0_cross=include_lag0_cross,
        )
        splits = rolling_cv_splits(len(X_sup), initial_train_size, step_size, test_size)
        if not splits:
            raise ValueError("No rolling CV splits possible (check window sizes vs sample length).")

        pipe = Pipeline([("scaler", StandardScaler()), ("model", build_model(params))])

        Xn, yn = X_sup.to_numpy(), y_sup.to_numpy()
        scores = []
        for tr, te in splits:
            pipe.fit(Xn[tr], yn[tr])
            pred = pipe.predict(Xn[te])
            scores.append(mean_absolute_error(yn[te], pred))
        return float(np.mean(scores))

    def objective(trial: optuna.Trial) -> float:
        lags = LagSpec(
            ar_p=trial.suggest_int("p", *p_range),
            q_social=trial.suggest_int("q_social", *q_social_range),
            q_controls=trial.suggest_int("q_controls", *q_controls_range),
            q_cross=trial.suggest_int("q_cross", *q_cross_range),
        )

        if model_family == "ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}
        else:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 1e2, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            }

        return cv_score(lags, params)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params.copy()
    best_lags = LagSpec(
        ar_p=best.pop("p"),
        q_social=best.pop("q_social"),
        q_controls=best.pop("q_controls"),
        q_cross=best.pop("q_cross"),
    )
    best_params = best

    X_sup, y_sup = make_group_arx_supervised(
        y=y, social=social, controls=controls, cross=cross,
        lags=best_lags, horizon=horizon,
        include_lag0_social=include_lag0_social,
        include_lag0_controls=include_lag0_controls,
        include_lag0_cross=include_lag0_cross,
    )

    final_pipe = Pipeline([("scaler", StandardScaler()), ("model", build_model(best_params))])
    final_pipe.fit(X_sup.to_numpy(), y_sup.to_numpy())

    coef = pd.Series(final_pipe.named_steps["model"].coef_, index=X_sup.columns).sort_values(
        key=np.abs, ascending=False
    )

    return {
        "best_score_mae": float(study.best_value),
        "best_lags": best_lags,
        "best_model_family": model_family,
        "best_model_params": best_params,
        "coef_sorted": coef,
        "fitted_pipeline": final_pipe,
        "study": study,
    }


# ----------------------------
# 5) Wrapper: run S&P / BTC / Gold (returns already logged)
# ----------------------------
def run_three_assets_on_returns(
    returns: pd.DataFrame,        # trading-day index; columns: ["sp500","bitcoin","gold"] (rename as needed)
    social_daily: pd.DataFrame,   # includes weekends
    controls_td: pd.DataFrame,    # trading-day index (or daily; we'll map if needed)
    horizon: int = 1,
    social_weekend_agg: Agg = "mean",
    controls_if_daily_agg: Agg = "last",
):
    """
    You said:
      - keep native trading-day calendar (no synthetic weekend/holiday rows)
      - prices forward-filled within existing rows only, flag rows that needed filling

    This function:
      1) uses returns' index as the trading calendar
      2) maps weekend/holiday social rows into the NEXT trading day
      3) forward-fills missing returns/controls only within trading days and adds fill flags as extra controls
      4) runs AutoML ARX separately for each asset
    """
    # 1) Trading calendar from returns
    trading_idx = pd.DatetimeIndex(returns.index).sort_values()

    # 2) Map socials (weekends) -> next trading day, aggregate
    social_td = map_to_next_trading_day(social_daily, trading_idx, agg=social_weekend_agg)

    # 3) Controls: if already trading-day, just reindex; if daily/weekend, map similarly
    if controls_td is not None and not controls_td.empty:
        # Heuristic: if controls index equals trading index -> treat as TD; else map to next TD
        if pd.DatetimeIndex(controls_td.index).equals(trading_idx):
            controls_mapped = controls_td.reindex(trading_idx)
        else:
            controls_mapped = map_to_next_trading_day(controls_td, trading_idx, agg=controls_if_daily_agg)
    else:
        controls_mapped = controls_td

    # 4) Forward-fill within existing rows only + flags (for returns and controls)
    returns_filled, returns_fill_flags = forward_fill_with_flag(returns.reindex(trading_idx))
    controls_filled, controls_fill_flags = forward_fill_with_flag(controls_mapped.reindex(trading_idx))

    # Add fill flags to controls so the model can “know” when imputation happened
    if controls_filled is None or controls_filled.empty:
        controls_final = pd.DataFrame(index=trading_idx)
    else:
        controls_final = controls_filled.copy()

    if isinstance(controls_fill_flags, pd.DataFrame) and not controls_fill_flags.empty:
        controls_final = pd.concat([controls_final, controls_fill_flags], axis=1)

    # Optionally also include return fill flags as controls (useful if any asset had gaps)
    if isinstance(returns_fill_flags, pd.DataFrame) and not returns_fill_flags.empty:
        controls_final = pd.concat([controls_final, returns_fill_flags], axis=1)

    # 5) Cross-asset predictors = other two return series (use lagged-only in model)
    r = returns_filled
    cross_for = {
        "sp500":   r[["bitcoin", "gold"]],
        "bitcoin": r[["sp500", "gold"]],
        "gold":    r[["sp500", "bitcoin"]],
    }

    # Asset-specific bounds (sane for Jan–May trading days)
    bounds = {
        "sp500":  dict(p_range=(1, 5), q_social_range=(0, 2), q_controls_range=(0, 3), q_cross_range=(0, 1)),
        "bitcoin":dict(p_range=(1, 7), q_social_range=(0, 3), q_controls_range=(0, 2), q_cross_range=(0, 1)),
        "gold":   dict(p_range=(1, 5), q_social_range=(0, 2), q_controls_range=(0, 5), q_cross_range=(0, 1)),
    }

    results = {}
    for asset in ["sp500", "bitcoin", "gold"]:
        results[asset] = automl_group_arx(
            y=r[asset],
            social=social_td,
            controls=controls_final,
            cross=cross_for[asset],
            horizon=horizon,                 # horizon=1 => predict next trading day's return
            initial_train_size=60,
            step_size=5,
            test_size=5,
            n_trials=60,
            model_family="elasticnet",
            # key paper-safe timing choices:
            include_lag0_social=True,        # socials dated t predict t+1
            include_lag0_controls=False,     # conservative: controls lagged-only
            include_lag0_cross=False,        # avoid mechanical contemporaneous spillovers
            **bounds[asset],
        )

    return {
        "results": results,
        "aligned": {
            "returns": r,
            "social_trading_day": social_td,
            "controls_final": controls_final,
        }
    }
