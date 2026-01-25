import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Literal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Index family column prefixes for heterogeneous lag support (shared with ARX)
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


def _classify_exog_columns(columns: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Classify exogenous columns into emotion index families and controls."""
    classified = {family: [] for family in INDEX_FAMILIES}
    controls = []
    
    for col in columns:
        col_lower = col.lower()
        matched = False
        
        for family, prefixes in INDEX_FAMILIES.items():
            for prefix in prefixes:
                if col_lower == prefix or col_lower.startswith(prefix + "_"):
                    if col not in classified[family]:
                        classified[family].append(col)
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            controls.append(col)
    
    return classified, controls


@dataclass
class VARXLagSpec:
    """
    Lag specification for VARX models with heterogeneous exogenous lags.
    
    Supports per-family lag orders for emotion indices while using a common
    lag order for endogenous variables and control variables.
    """
    p: int  # Endogenous (AR) lag order
    q_controls: int  # Control variable lag order
    # Per-family emotion index lags (heterogeneous)
    q_ei: int = 0
    q_epi: int = 0
    q_epi_signed: int = 0
    q_ccd: int = 0
    q_intensity: int = 0
    q_surprise: int = 0
    q_split: int = 0
    # Backward compatibility
    q: Optional[int] = None
    
    def __post_init__(self):
        """Handle backward compatibility with single q."""
        if self.q is not None:
            import warnings
            warnings.warn(
                "VARXLagSpec.q is deprecated. Use per-family lags "
                "(q_ei, q_epi, etc.) and q_controls instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.q_controls = self.q
            self.q_ei = self.q
            self.q_epi = self.q
            self.q_epi_signed = self.q
            self.q_ccd = self.q
            self.q_intensity = self.q
            self.q_surprise = self.q
            self.q_split = self.q
    
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
    
    def max_exog_lag(self) -> int:
        """Get maximum exogenous lag order."""
        return max(
            self.q_controls, self.q_ei, self.q_epi, self.q_epi_signed,
            self.q_ccd, self.q_intensity, self.q_surprise, self.q_split
        )


def make_varx_design(Y: pd.DataFrame, X: pd.DataFrame, p: int, q: int):
    """
    Legacy function for backward compatibility.
    
    Y: (T, k) endogenous returns (SPX, BTC, Gold) on trading-day index
    X: (T, m) exogenous features aligned to Y index (socials+macro+fill flags)
    Build Z = [Y_{t-1..t-p}, X_{t..t-q}] and target Y_t
    """
    Z_parts = []

    # lagged Y
    for i in range(1, p+1):
        Yi = Y.shift(i)
        Yi.columns = [f"{c}_lag{i}" for c in Y.columns]
        Z_parts.append(Yi)

    # current + lagged X
    for j in range(0, q+1):
        Xj = X.shift(j)
        Xj.columns = [f"{c}_xlag{j}" for c in X.columns]
        Z_parts.append(Xj)

    Z = pd.concat(Z_parts, axis=1)
    data = pd.concat([Z, Y], axis=1).dropna()
    Z = data[Z.columns]
    Yt = data[Y.columns]
    return Z, Yt


def make_varx_design_heterogeneous(
    Y: pd.DataFrame, 
    X: pd.DataFrame, 
    lags: VARXLagSpec,
    include_lag0_exog: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build VARX design matrix with heterogeneous exogenous lags.
    
    Y: (T, k) endogenous returns on trading-day index
    X: (T, m) exogenous features (emotion indices + controls)
    lags: VARXLagSpec with per-family lag orders
    include_lag0_exog: Whether to include contemporaneous exogenous (t) in prediction
    
    Returns:
        Z: Design matrix with lagged endogenous and exogenous features
        Yt: Target matrix (returns at time t)
    """
    Z_parts = []

    # Lagged endogenous (Y_{t-1}, ..., Y_{t-p})
    for i in range(1, lags.p + 1):
        Yi = Y.shift(i)
        Yi.columns = [f"{c}_lag{i}" for c in Y.columns]
        Z_parts.append(Yi)

    # Classify exogenous columns
    classified, controls = _classify_exog_columns(list(X.columns))
    
    # Add control variables with their lag order
    if controls and lags.q_controls >= 0:
        start_lag = 0 if include_lag0_exog else 1
        for j in range(start_lag, lags.q_controls + 1):
            Xj = X[controls].shift(j)
            Xj.columns = [f"{c}_xlag{j}" for c in controls]
            Z_parts.append(Xj)
    
    # Add emotion index families with heterogeneous lags
    for family, cols in classified.items():
        if cols:
            family_lag = lags.get_family_lag(family)
            if family_lag > 0 or include_lag0_exog:
                start_lag = 0 if include_lag0_exog else 1
                for j in range(start_lag, family_lag + 1):
                    Xj = X[cols].shift(j)
                    Xj.columns = [f"{c}_xlag{j}" for c in cols]
                    Z_parts.append(Xj)

    Z = pd.concat(Z_parts, axis=1)
    data = pd.concat([Z, Y], axis=1).dropna()
    Z = data[Z.columns]
    Yt = data[Y.columns]
    return Z, Yt


def rolling_splits(T, initial_train, step, val_size):
    """Generate rolling window CV splits."""
    splits = []
    train_end = initial_train
    while train_end + val_size <= T:
        tr = np.arange(0, train_end)
        va = np.arange(train_end, train_end + val_size)
        splits.append((tr, va))
        train_end += step
    return splits


def cv_score_varx(Y, X, p, q, model_kind, params,
                 initial_train=40, step=5, val_size=5):
    """Legacy CV scoring with single q for backward compatibility."""
    Z, Yt = make_varx_design(Y, X, p, q)
    idx_map = {d:i for i, d in enumerate(Z.index)}

    splits_raw = rolling_splits(len(Y.index), initial_train, step, val_size)
    splits = []
    for tr, va in splits_raw:
        tr_dates = Y.index[tr]
        va_dates = Y.index[va]
        tr2 = [idx_map[d] for d in tr_dates if d in idx_map]
        va2 = [idx_map[d] for d in va_dates if d in idx_map]
        if len(tr2) > 20 and len(va2) > 0:
            splits.append((np.array(tr2), np.array(va2)))

    if model_kind == "ridge":
        reg = Ridge(**params)
    elif model_kind == "mt_enet":
        reg = MultiTaskElasticNet(**params, max_iter=20000)
    else:
        raise ValueError("model_kind must be 'ridge' or 'mt_enet'")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", reg)
    ])

    rmses = []
    for tr2, va2 in splits:
        model.fit(Z.iloc[tr2], Yt.iloc[tr2])
        pred = model.predict(Z.iloc[va2])
        rmse = np.sqrt(mean_squared_error(Yt.iloc[va2], pred))
        rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else np.inf


def cv_score_varx_heterogeneous(
    Y: pd.DataFrame, 
    X: pd.DataFrame, 
    lags: VARXLagSpec, 
    model_kind: str, 
    params: Dict,
    initial_train: int = 60, 
    step: int = 1, 
    val_size: int = 1,
    include_lag0_exog: bool = True,
) -> Tuple[float, float]:
    """
    Cross-validation scoring for VARX with heterogeneous exogenous lags.
    
    Returns:
        Tuple of (mae, rmse) averaged across CV folds
    """
    Z, Yt = make_varx_design_heterogeneous(Y, X, lags, include_lag0_exog)
    idx_map = {d: i for i, d in enumerate(Z.index)}

    splits_raw = rolling_splits(len(Y.index), initial_train, step, val_size)
    splits = []
    for tr, va in splits_raw:
        tr_dates = Y.index[tr]
        va_dates = Y.index[va]
        tr2 = [idx_map[d] for d in tr_dates if d in idx_map]
        va2 = [idx_map[d] for d in va_dates if d in idx_map]
        if len(tr2) > 20 and len(va2) > 0:
            splits.append((np.array(tr2), np.array(va2)))

    if not splits:
        return np.inf, np.inf

    if model_kind == "ridge":
        reg = Ridge(**params)
    elif model_kind == "mt_enet":
        reg = MultiTaskElasticNet(**params, max_iter=20000)
    else:
        raise ValueError("model_kind must be 'ridge' or 'mt_enet'")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", reg)
    ])

    mae_scores = []
    rmse_scores = []
    for tr2, va2 in splits:
        model.fit(Z.iloc[tr2], Yt.iloc[tr2])
        pred = model.predict(Z.iloc[va2])
        mae_scores.append(mean_absolute_error(Yt.iloc[va2], pred))
        rmse_scores.append(np.sqrt(mean_squared_error(Yt.iloc[va2], pred)))

    return float(np.mean(mae_scores)), float(np.mean(rmse_scores))


def automl_regularized_varx(Y, X,
                            p_grid=(1,2,3,5,7,10,15),
                            q_grid=(0,1,2,3,5,7,10),
                            ridge_alphas=np.logspace(-4, 4, 17),
                            enet_alphas=np.logspace(-4, 2, 13),
                            enet_l1s=(0.1,0.3,0.5,0.7,0.9)):
    """Legacy grid search for backward compatibility."""
    best = {"score": np.inf}
    for p in p_grid:
        for q in q_grid:
            # Ridge
            for a in ridge_alphas:
                s = cv_score_varx(Y, X, p, q, "ridge", {"alpha": a})
                if s < best["score"]:
                    best = {"score": s, "p": p, "q": q, "model": "ridge", "params": {"alpha": float(a)}}

            # MultiTask ElasticNet (joint sparsity across outputs)
            for a in enet_alphas:
                for l1 in enet_l1s:
                    s = cv_score_varx(Y, X, p, q, "mt_enet", {"alpha": a, "l1_ratio": l1})
                    if s < best["score"]:
                        best = {"score": s, "p": p, "q": q, "model": "mt_enet",
                                "params": {"alpha": float(a), "l1_ratio": float(l1)}}

    return best


def automl_varx_optuna(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    initial_train: int = 60,
    step: int = 1,
    val_size: int = 1,
    n_trials: int = 300,
    random_seed: int = 42,
    p_range: Tuple[int, int] = (1, 7),
    q_controls_range: Tuple[int, int] = (0, 3),
    # Per-family emotion index lag ranges (heterogeneous)
    q_ei_range: Tuple[int, int] = (0, 5),
    q_epi_range: Tuple[int, int] = (0, 5),
    q_epi_signed_range: Tuple[int, int] = (0, 5),
    q_ccd_range: Tuple[int, int] = (0, 5),
    q_intensity_range: Tuple[int, int] = (0, 5),
    q_surprise_range: Tuple[int, int] = (0, 5),
    q_split_range: Tuple[int, int] = (0, 5),
    model_family: str = "mt_enet",  # "ridge" or "mt_enet"
    include_lag0_exog: bool = True,
    use_pruning: bool = True,
) -> Dict:
    """
    Optuna-based hyperparameter search for VARX models with heterogeneous exogenous lags.
    
    Supports per-family lag orders for 7 emotion index families while jointly
    modeling all assets (cross-asset spillovers via endogenous lag matrices).
    
    Args:
        Y: Endogenous return DataFrame (T, k) with assets as columns
        X: Exogenous features DataFrame (emotion indices + controls)
        initial_train: Initial training window size
        step: Step size for rolling window
        val_size: Validation set size per fold
        n_trials: Number of Optuna trials (default 300)
        random_seed: Random seed for reproducibility
        p_range: Endogenous AR lag order range
        q_controls_range: Control variable lag range
        q_*_range: Per-family emotion index lag ranges
        model_family: "ridge" or "mt_enet" (MultiTaskElasticNet)
        include_lag0_exog: Include contemporaneous exogenous features
        use_pruning: Use Optuna MedianPruner
    
    Returns:
        Dict with best_score_mae, best_score_rmse, best_lags, best_model_params,
        coef_matrix, fitted_pipeline, study, feature_names
    """
    import optuna
    
    def build_model(params: Dict):
        if model_family == "ridge":
            return Ridge(**params)
        return MultiTaskElasticNet(**params, max_iter=20000)

    def objective(trial: optuna.Trial) -> float:
        lags = VARXLagSpec(
            p=trial.suggest_int("p", *p_range),
            q_controls=trial.suggest_int("q_controls", *q_controls_range),
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

        mae, rmse = cv_score_varx_heterogeneous(
            Y, X, lags, model_family, params,
            initial_train, step, val_size, include_lag0_exog
        )
        
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("rmse", rmse)
        
        # Optimize on MAE (consistent with ARX)
        return mae

    # Configure Optuna
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    if use_pruning:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract best parameters
    best = study.best_params.copy()
    best_lags = VARXLagSpec(
        p=best.pop("p"),
        q_controls=best.pop("q_controls"),
        q_ei=best.pop("q_ei"),
        q_epi=best.pop("q_epi"),
        q_epi_signed=best.pop("q_epi_signed"),
        q_ccd=best.pop("q_ccd"),
        q_intensity=best.pop("q_intensity"),
        q_surprise=best.pop("q_surprise"),
        q_split=best.pop("q_split"),
    )
    best_params = best

    # Fit final model
    Z, Yt = make_varx_design_heterogeneous(Y, X, best_lags, include_lag0_exog)
    
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", build_model(best_params))
    ])
    final_pipe.fit(Z.values, Yt.values)

    # Extract coefficient matrix (features × assets)
    coef_matrix = pd.DataFrame(
        final_pipe.named_steps["reg"].coef_.T,
        index=Z.columns,
        columns=Yt.columns
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
        "coef_matrix": coef_matrix,
        "fitted_pipeline": final_pipe,
        "feature_names": list(Z.columns),
        "Z_design": Z,
        "Y_target": Yt,
        "study": study,
    }
