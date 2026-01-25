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

# Index family column prefixes for heterogeneous lag support
INDEX_FAMILIES = {
    "ei": ["ei_creator", "ei_creator_engweighted", "ei_creator_topicweighted", 
           "ei_creator_engweighted_topicweighted", "ei_community", "ei_community_topicweighted"],
    "epi": ["epi", "epi_engweighted", "epi_topicweighted", "epi_engweighted_topicweighted"],
    "epi_signed": ["epi_signed", "epi_signed_engweighted", "epi_signed_topicweighted", 
                   "epi_signed_engweighted_topicweighted"],
    "ccd": ["ccd", "ccd_engweighted", "ccd_topicweighted", "ccd_engweighted_topicweighted"],
    "intensity": ["intensity", "intensity_engweighted", "intensity_topicweighted", 
                  "intensity_engweighted_topicweighted"],
    "surprise": ["surprise", "surprise_engweighted", "surprise_topicweighted", 
                 "surprise_engweighted_topicweighted"],
    "split": ["split"],
}


def _classify_social_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Classify social columns into index families."""
    classified = {family: [] for family in INDEX_FAMILIES}
    unclassified = []
    
    for col in columns:
        col_lower = col.lower()
        matched = False
        
        # Check each family (order matters: check more specific first)
        # epi_signed before epi, ei_creator/ei_community before general
        for family, prefixes in INDEX_FAMILIES.items():
            for prefix in prefixes:
                if col_lower == prefix or col_lower.startswith(prefix + "_"):
                    # Exact match or prefix match
                    if col not in classified[family]:
                        classified[family].append(col)
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            unclassified.append(col)
    
    return classified, unclassified


@dataclass
class LagSpec:
    """
    Lag specification for ARX models with heterogeneous emotion index lags.
    
    Supports per-family lag orders for fine-grained control over how different
    emotion indices enter the model with potentially different lag structures.
    """
    ar_p: int
    q_controls: int
    q_cross: int
    # Per-family emotion index lags (heterogeneous)
    q_ei: int = 0
    q_epi: int = 0
    q_epi_signed: int = 0
    q_ccd: int = 0
    q_intensity: int = 0
    q_surprise: int = 0
    q_split: int = 0
    # Backward compatibility: if set, overrides all family lags
    q_social: Optional[int] = None
    
    def __post_init__(self):
        """Handle backward compatibility with q_social."""
        if self.q_social is not None:
            import warnings
            warnings.warn(
                "LagSpec.q_social is deprecated. Use per-family lags "
                "(q_ei, q_epi, q_epi_signed, q_ccd, q_intensity, q_surprise, q_split) instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Set all family lags to q_social value
            self.q_ei = self.q_social
            self.q_epi = self.q_social
            self.q_epi_signed = self.q_social
            self.q_ccd = self.q_social
            self.q_intensity = self.q_social
            self.q_surprise = self.q_social
            self.q_split = self.q_social
    
    def get_family_lag(self, family: str) -> int:
        """Get lag order for a specific index family."""
        lag_map = {
            "ei": self.q_ei,
            "epi": self.q_epi,
            "epi_signed": self.q_epi_signed,
            "ccd": self.q_ccd,
            "intensity": self.q_intensity,
            "surprise": self.q_surprise,
            "split": self.q_split,
        }
        return lag_map.get(family, 0)
    
    def has_any_social_lags(self) -> bool:
        """Check if any social/emotion index lags are enabled."""
        return any([
            self.q_ei > 0, self.q_epi > 0, self.q_epi_signed > 0,
            self.q_ccd > 0, self.q_intensity > 0, self.q_surprise > 0, self.q_split > 0
        ])


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
    
    Supports heterogeneous lag orders per emotion index family:
      - ei, epi, epi_signed, ccd, intensity, surprise, split
    """
    y = y.astype(float)

    feat = pd.DataFrame(index=y.index)

    # AR terms: past returns only
    for lag in range(1, lags.ar_p + 1):
        feat[f"y_lag{lag}"] = y.shift(lag)

    # Social/emotion indices with heterogeneous lags per family
    if social is not None and not social.empty:
        classified, unclassified = _classify_social_columns(list(social.columns))
        
        # Add lags for each family with its specific lag order
        for family, cols in classified.items():
            if cols:
                family_lag = lags.get_family_lag(family)
                if family_lag > 0 or include_lag0_social:
                    _add_lags(feat, social, cols, family_lag, include_lag0_social, "soc_")
        
        # Handle unclassified columns with max of all family lags (fallback)
        if unclassified:
            max_lag = max(
                lags.q_ei, lags.q_epi, lags.q_epi_signed, 
                lags.q_ccd, lags.q_intensity, lags.q_surprise, lags.q_split
            )
            if max_lag > 0 or include_lag0_social:
                _add_lags(feat, social, unclassified, max_lag, include_lag0_social, "soc_")
    
    # Controls
    if controls is not None:
        _add_lags(feat, controls, list(controls.columns), lags.q_controls, include_lag0_controls, "ctl_")
    
    # Cross-asset predictors
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
    step_size: int = 1,
    test_size: int = 1,
    n_trials: int = 300,
    random_seed: int = 42,
    p_range: Tuple[int, int] = (1, 7),
    q_controls_range: Tuple[int, int] = (0, 3),
    q_cross_range: Tuple[int, int] = (0, 1),
    # Per-family emotion index lag ranges (heterogeneous)
    q_ei_range: Tuple[int, int] = (0, 5),
    q_epi_range: Tuple[int, int] = (0, 5),
    q_epi_signed_range: Tuple[int, int] = (0, 5),
    q_ccd_range: Tuple[int, int] = (0, 5),
    q_intensity_range: Tuple[int, int] = (0, 5),
    q_surprise_range: Tuple[int, int] = (0, 5),
    q_split_range: Tuple[int, int] = (0, 5),
    model_family: str = "elasticnet",  # "elasticnet" or "ridge"
    include_lag0_social: bool = True,
    include_lag0_controls: bool = False,
    include_lag0_cross: bool = False,
    use_pruning: bool = True,
    cv_metric: str = "both",  # "mae", "rmse", or "both" (returns both, optimizes on mae)
    # Deprecated parameter for backward compatibility
    q_social_range: Optional[Tuple[int, int]] = None,
):
    """
    Optuna-based hyperparameter search for ARX models with heterogeneous emotion lags.
    
    Supports per-family lag orders for 7 emotion index families:
    - ei: Emotion Index (creator/community sentiment)
    - epi: Emotion Polarity Index (intensity × divergence)
    - epi_signed: Signed EPI (preserves valence direction)
    - ccd: Creator-Community Divergence
    - intensity: Emotional Intensity
    - surprise: Informational novelty
    - split: Within-community polarization
    
    Args:
        y: Target return series
        social: Social/emotion index DataFrame
        controls: Macro-financial control variables
        cross: Cross-asset predictors (not used in univariate ARX)
        horizon: Forecast horizon (1 = next trading day)
        initial_train_size: Initial training window size
        step_size: Step size for rolling window
        test_size: Test set size per fold
        n_trials: Number of Optuna trials (default 300 with pruning)
        random_seed: Random seed for reproducibility
        p_range: AR lag order range
        q_controls_range: Control variable lag range
        q_cross_range: Cross-asset lag range (set to (0,0) for univariate)
        q_*_range: Per-family emotion index lag ranges
        model_family: "elasticnet" or "ridge"
        include_lag0_*: Whether to include contemporaneous (lag0) features
        use_pruning: Whether to use Optuna MedianPruner
        cv_metric: Metric(s) to compute ("mae", "rmse", or "both")
        q_social_range: DEPRECATED - use per-family ranges instead
    
    Returns:
        Dict with best_score_mae, best_score_rmse, best_lags, best_model_params,
        coef_sorted, fitted_pipeline, study, feature_names
    """
    import optuna
    from sklearn.metrics import mean_squared_error
    
    # Handle deprecated q_social_range
    if q_social_range is not None:
        import warnings
        warnings.warn(
            "q_social_range is deprecated. Use per-family ranges "
            "(q_ei_range, q_epi_range, etc.) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Apply to all families
        q_ei_range = q_social_range
        q_epi_range = q_social_range
        q_epi_signed_range = q_social_range
        q_ccd_range = q_social_range
        q_intensity_range = q_social_range
        q_surprise_range = q_social_range
        q_split_range = q_social_range

    def build_model(params: Dict):
        if model_family == "ridge":
            return Ridge(alpha=params["alpha"], random_state=random_seed)
        return ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            random_state=random_seed,
            max_iter=20000,
        )

    def cv_score(lags: LagSpec, params: Dict) -> Tuple[float, float]:
        """Returns (mae, rmse) tuple."""
        X_sup, y_sup = make_group_arx_supervised(
            y=y, social=social, controls=controls, cross=cross,
            lags=lags, horizon=horizon,
            include_lag0_social=include_lag0_social,
            include_lag0_controls=include_lag0_controls,
            include_lag0_cross=include_lag0_cross,
        )
        splits = rolling_cv_splits(len(X_sup), initial_train_size, step_size, test_size)
        if not splits:
            return np.inf, np.inf

        pipe = Pipeline([("scaler", StandardScaler()), ("model", build_model(params))])

        Xn, yn = X_sup.to_numpy(), y_sup.to_numpy()
        mae_scores = []
        rmse_scores = []
        for tr, te in splits:
            pipe.fit(Xn[tr], yn[tr])
            pred = pipe.predict(Xn[te])
            mae_scores.append(mean_absolute_error(yn[te], pred))
            rmse_scores.append(np.sqrt(mean_squared_error(yn[te], pred)))
        
        return float(np.mean(mae_scores)), float(np.mean(rmse_scores))

    def objective(trial: optuna.Trial) -> float:
        # Build LagSpec with heterogeneous emotion index lags
        lags = LagSpec(
            ar_p=trial.suggest_int("p", *p_range),
            q_controls=trial.suggest_int("q_controls", *q_controls_range),
            q_cross=trial.suggest_int("q_cross", *q_cross_range),
            q_ei=trial.suggest_int("q_ei", *q_ei_range),
            q_epi=trial.suggest_int("q_epi", *q_epi_range),
            q_epi_signed=trial.suggest_int("q_epi_signed", *q_epi_signed_range),
            q_ccd=trial.suggest_int("q_ccd", *q_ccd_range),
            q_intensity=trial.suggest_int("q_intensity", *q_intensity_range),
            q_surprise=trial.suggest_int("q_surprise", *q_surprise_range),
            q_split=trial.suggest_int("q_split", *q_split_range),
        )

        if model_family == "ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}
        else:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 1e2, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            }

        mae, rmse = cv_score(lags, params)
        
        # Store both metrics as user attributes
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("rmse", rmse)
        
        # Optimize on MAE (primary metric)
        return mae

    # Configure Optuna with optional pruning
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    if use_pruning:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract best parameters
    best = study.best_params.copy()
    best_lags = LagSpec(
        ar_p=best.pop("p"),
        q_controls=best.pop("q_controls"),
        q_cross=best.pop("q_cross"),
        q_ei=best.pop("q_ei"),
        q_epi=best.pop("q_epi"),
        q_epi_signed=best.pop("q_epi_signed"),
        q_ccd=best.pop("q_ccd"),
        q_intensity=best.pop("q_intensity"),
        q_surprise=best.pop("q_surprise"),
        q_split=best.pop("q_split"),
    )
    best_params = best

    # Fit final model on full data
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

    # Get best trial metrics
    best_mae = study.best_trial.user_attrs.get("mae", study.best_value)
    best_rmse = study.best_trial.user_attrs.get("rmse", np.nan)

    return {
        "best_score_mae": float(best_mae),
        "best_score_rmse": float(best_rmse),
        "best_lags": best_lags,
        "best_model_family": model_family,
        "best_model_params": best_params,
        "coef_sorted": coef,
        "fitted_pipeline": final_pipe,
        "feature_names": list(X_sup.columns),
        "X_supervised": X_sup,
        "y_supervised": y_sup,
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

    # 5) For univariate ARX, no cross-asset predictors (use VARX for cross-asset)
    r = returns_filled

    # Asset-specific bounds (sane for Jan–May trading days)
    # Using new heterogeneous lag interface with all emotion index families
    bounds = {
        "sp500":  dict(
            p_range=(1, 5), q_controls_range=(0, 3), q_cross_range=(0, 0),
            q_ei_range=(0, 3), q_epi_range=(0, 3), q_epi_signed_range=(0, 3),
            q_ccd_range=(0, 3), q_intensity_range=(0, 3), q_surprise_range=(0, 3), q_split_range=(0, 3),
        ),
        "bitcoin": dict(
            p_range=(1, 7), q_controls_range=(0, 2), q_cross_range=(0, 0),
            q_ei_range=(0, 5), q_epi_range=(0, 5), q_epi_signed_range=(0, 5),
            q_ccd_range=(0, 5), q_intensity_range=(0, 5), q_surprise_range=(0, 5), q_split_range=(0, 5),
        ),
        "gold":   dict(
            p_range=(1, 5), q_controls_range=(0, 5), q_cross_range=(0, 0),
            q_ei_range=(0, 3), q_epi_range=(0, 3), q_epi_signed_range=(0, 3),
            q_ccd_range=(0, 3), q_intensity_range=(0, 3), q_surprise_range=(0, 3), q_split_range=(0, 3),
        ),
    }

    results = {}
    for asset in ["sp500", "bitcoin", "gold"]:
        results[asset] = automl_group_arx(
            y=r[asset],
            social=social_td,
            controls=controls_final,
            cross=None,  # Univariate: no cross-asset predictors
            horizon=horizon,                 # horizon=1 => predict next trading day's return
            initial_train_size=60,
            step_size=1,
            test_size=1,
            n_trials=300,
            model_family="elasticnet",
            use_pruning=True,
            # key paper-safe timing choices:
            include_lag0_social=True,        # socials dated t predict t+1
            include_lag0_controls=False,     # conservative: controls lagged-only
            include_lag0_cross=False,        # not used in univariate
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
