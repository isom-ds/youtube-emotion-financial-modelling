"""
Evaluation metrics and statistical tests for financial model comparison.

Includes:
- Standard metrics (MAE, RMSE, MAPE, R², directional accuracy)
- Forecast comparison tests (Diebold-Mariano, Clark-West)
- Bootstrap confidence intervals for coefficients
- Permutation importance
- Multiple testing correction (FDR)
- Volatility clustering diagnostics (ACF, Ljung-Box)
- Overfitting diagnostics (Information Coefficient)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox


# =============================================================================
# Core Metrics
# =============================================================================

def compute_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    prefix: str = "",
    include_clustering: bool = False,
    max_lags: int = 20,
) -> Dict[str, float]:
    """
    Compute comprehensive prediction metrics for volatility models.
    
    Args:
        y_true: Actual volatility values (absolute returns)
        y_pred: Predicted volatility values
        prefix: Optional prefix for metric keys (e.g., "train_", "test_")
        include_clustering: If True, compute volatility clustering diagnostics on residuals
        max_lags: Maximum lags for ACF analysis (default: 20)
    
    Returns:
        Dict with keys: mae, rmse, mape, r2
        If include_clustering=True, also includes: acf_squared, ljung_box_stat, ljung_box_pvalue, persistence, half_life
        Note: Directional accuracy removed as it's not applicable for volatility prediction
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove any NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {f"{prefix}mae": np.nan, f"{prefix}rmse": np.nan, 
                f"{prefix}mape": np.nan, f"{prefix}r2": np.nan}
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (exclude zeros to avoid division by zero)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    # R² score
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    
    metrics = {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": float(rmse),
        f"{prefix}mape": float(mape),
        f"{prefix}r2": float(r2),
    }
    
    # Volatility clustering diagnostics on prediction errors
    if include_clustering and len(y_true) > max_lags:
        residuals = y_true - y_pred
        squared_residuals = residuals ** 2
        
        # ACF of squared residuals
        acf_values = acf(squared_residuals, nlags=max_lags, fft=False)
        
        # Ljung-Box test
        lb_result = acorr_ljungbox(squared_residuals, lags=max_lags, return_df=False)
        lb_stat = lb_result[0][-1]
        lb_pvalue = lb_result[1][-1]
        
        # Persistence measure
        n = len(residuals)
        threshold = 1.96 / np.sqrt(n)
        significant_acf = acf_values[1:][np.abs(acf_values[1:]) > threshold]
        persistence = np.sum(significant_acf) if len(significant_acf) > 0 else 0.0
        
        # Half-life
        half_life_idx = np.where(acf_values[1:] < 0.5)[0]
        half_life = half_life_idx[0] + 1 if len(half_life_idx) > 0 else max_lags
        
        metrics.update({
            f"{prefix}ljung_box_stat": float(lb_stat),
            f"{prefix}ljung_box_pvalue": float(lb_pvalue),
            f"{prefix}acf_persistence": float(persistence),
            f"{prefix}acf_half_life": int(half_life),
        })
    
    return metrics


def get_significance_stars(
    pvalue: float,
    levels: Dict[float, str] = None,
) -> str:
    """
    Convert p-value to significance stars.
    
    Args:
        pvalue: P-value to convert
        levels: Dict mapping significance levels to star markers
                Default: {0.01: '***', 0.05: '**', 0.10: '*'}
    
    Returns:
        Star string or empty string if not significant
    """
    if levels is None:
        levels = {0.01: '***', 0.05: '**', 0.10: '*'}
    
    if pd.isna(pvalue):
        return ''
    
    for level, stars in sorted(levels.items()):
        if pvalue <= level:
            return stars
    return ''


# =============================================================================
# Diebold-Mariano Test
# =============================================================================

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss_function: str = "SE",
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t) is the loss differential.
    
    Args:
        e1: Forecast errors from model 1
        e2: Forecast errors from model 2
        h: Forecast horizon (for HAC variance adjustment)
        loss_function: "SE" (squared error) or "AE" (absolute error)
        alternative: "two-sided", "less" (e1 < e2), or "greater" (e1 > e2)
    
    Returns:
        Dict with 'statistic', 'pvalue', 'mean_diff' (positive = model 1 worse)
    """
    e1 = np.asarray(e1).flatten()
    e2 = np.asarray(e2).flatten()
    
    assert len(e1) == len(e2), "Error arrays must have same length"
    
    # Compute loss differential
    if loss_function == "SE":
        d = e1**2 - e2**2
    elif loss_function == "AE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss_function must be 'SE' or 'AE'")
    
    n = len(d)
    d_bar = np.mean(d)
    
    # Newey-West HAC variance estimator
    # Autocovariance at lag k
    def autocovariance(d, k):
        return np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)) if k > 0 else np.var(d)
    
    # HAC bandwidth = h - 1 (for h-step ahead forecasts)
    bandwidth = max(h - 1, 0)
    
    # Long-run variance
    gamma_0 = autocovariance(d, 0)
    if bandwidth > 0:
        lr_var = gamma_0 + 2 * sum(
            (1 - k / (bandwidth + 1)) * autocovariance(d, k) 
            for k in range(1, bandwidth + 1)
        )
    else:
        lr_var = gamma_0
    
    # Ensure positive variance
    lr_var = max(lr_var, 1e-10)
    
    # DM statistic (asymptotically N(0,1))
    dm_stat = d_bar / np.sqrt(lr_var / n)
    
    # P-value
    if alternative == "two-sided":
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    elif alternative == "less":
        pvalue = stats.norm.cdf(dm_stat)
    elif alternative == "greater":
        pvalue = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
    return {
        "statistic": float(dm_stat),
        "pvalue": float(pvalue),
        "mean_diff": float(d_bar),
        "stars": get_significance_stars(pvalue),
    }


# =============================================================================
# Clark-West Test
# =============================================================================

def clark_west_test(
    y_true: np.ndarray,
    f_restricted: np.ndarray,
    f_unrestricted: np.ndarray,
    alternative: str = "greater",
) -> Dict[str, float]:
    """
    Clark-West test for nested model comparison.
    
    Tests whether unrestricted model (with additional predictors) provides
    better forecasts than restricted model, adjusting for parameter estimation noise.
    
    H0: Restricted model forecasts as well as unrestricted
    H1: Unrestricted model forecasts better (alternative="greater")
    
    Args:
        y_true: Actual values
        f_restricted: Forecasts from restricted (baseline) model
        f_unrestricted: Forecasts from unrestricted (enhanced) model
        alternative: "greater" (unrestricted better) or "two-sided"
    
    Returns:
        Dict with 'statistic', 'pvalue'
    """
    y = np.asarray(y_true).flatten()
    f1 = np.asarray(f_restricted).flatten()
    f2 = np.asarray(f_unrestricted).flatten()
    
    assert len(y) == len(f1) == len(f2), "Arrays must have same length"
    
    e1 = y - f1  # Restricted errors
    e2 = y - f2  # Unrestricted errors
    
    # Clark-West adjustment: f_hat = (e1^2 - e2^2) + (f1 - f2)^2
    # This corrects for the noise in parameter estimation
    adj = (f1 - f2) ** 2
    f_hat = e1**2 - e2**2 + adj
    
    n = len(f_hat)
    mean_f = np.mean(f_hat)
    std_f = np.std(f_hat, ddof=1)
    
    # Test statistic
    cw_stat = mean_f / (std_f / np.sqrt(n))
    
    # P-value (one-sided: unrestricted should be better)
    if alternative == "greater":
        pvalue = 1 - stats.norm.cdf(cw_stat)
    elif alternative == "two-sided":
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(cw_stat)))
    else:
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    
    return {
        "statistic": float(cw_stat),
        "pvalue": float(pvalue),
        "mean_msfe_diff": float(np.mean(e1**2) - np.mean(e2**2)),
        "stars": get_significance_stars(pvalue),
    }


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_coefficient_ci(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    feature_names: List[str],
    n_iter: int = 1000,
    alpha_levels: List[float] = [0.01, 0.05, 0.10],
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for model coefficients.
    
    Uses case resampling bootstrap to estimate coefficient distributions
    and compute confidence intervals at multiple significance levels.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        pipeline: Fitted sklearn Pipeline with 'model' step containing coef_
        feature_names: List of feature names
        n_iter: Number of bootstrap iterations (default 1000)
        alpha_levels: Significance levels for CIs (default [0.01, 0.05, 0.10])
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: coef, se, ci_lower_*, ci_upper_*, pvalue, sig_*
    """
    np.random.seed(random_seed)
    
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n_samples, n_features = X.shape
    
    # Get original coefficients
    original_coef = pipeline.named_steps["model"].coef_
    if original_coef.ndim > 1:
        original_coef = original_coef.flatten()
    
    # Bootstrap
    boot_coefs = np.zeros((n_iter, n_features))
    
    for i in range(n_iter):
        # Resample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        # Clone and fit pipeline
        from sklearn.base import clone
        pipe_boot = clone(pipeline)
        try:
            pipe_boot.fit(X_boot, y_boot)
            coef = pipe_boot.named_steps["model"].coef_
            if coef.ndim > 1:
                coef = coef.flatten()
            boot_coefs[i] = coef
        except Exception:
            boot_coefs[i] = np.nan
    
    # Remove failed iterations
    boot_coefs = boot_coefs[~np.isnan(boot_coefs).any(axis=1)]
    
    # Compute statistics
    results = []
    for j, name in enumerate(feature_names):
        row = {"feature": name, "coef": original_coef[j]}
        
        if len(boot_coefs) > 0:
            coef_dist = boot_coefs[:, j]
            row["se"] = np.std(coef_dist)
            row["boot_mean"] = np.mean(coef_dist)
            
            # Confidence intervals at each level
            for alpha in alpha_levels:
                lower_pct = alpha / 2 * 100
                upper_pct = (1 - alpha / 2) * 100
                row[f"ci_lower_{int((1-alpha)*100)}"] = np.percentile(coef_dist, lower_pct)
                row[f"ci_upper_{int((1-alpha)*100)}"] = np.percentile(coef_dist, upper_pct)
            
            # Bootstrap p-value (two-sided test for coef != 0)
            # Proportion of bootstrap samples on opposite side of zero
            if original_coef[j] >= 0:
                pvalue = 2 * np.mean(coef_dist <= 0)
            else:
                pvalue = 2 * np.mean(coef_dist >= 0)
            pvalue = min(pvalue, 1.0)
            row["pvalue"] = pvalue
            row["stars"] = get_significance_stars(pvalue)
            
            # Significance flags at each level
            for alpha in alpha_levels:
                ci_lower = row[f"ci_lower_{int((1-alpha)*100)}"]
                ci_upper = row[f"ci_upper_{int((1-alpha)*100)}"]
                # Significant if CI excludes zero
                row[f"sig_{int((1-alpha)*100)}"] = not (ci_lower <= 0 <= ci_upper)
        else:
            row["se"] = np.nan
            row["pvalue"] = np.nan
            row["stars"] = ""
        
        results.append(row)
    
    return pd.DataFrame(results).set_index("feature")


# =============================================================================
# Permutation Importance
# =============================================================================

def permutation_importance_cv(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    scoring: str = "mae",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance with bootstrap confidence intervals.
    
    Measures feature importance by permuting each feature and measuring
    the decrease in model performance.
    
    Args:
        model: Fitted model/pipeline
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
        scoring: "mae" or "rmse"
        random_seed: Random seed
    
    Returns:
        DataFrame with importance_mean, importance_std, and significance
    """
    from sklearn.inspection import permutation_importance as sklearn_perm_imp
    
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    
    # Define scoring function
    if scoring == "mae":
        scorer = "neg_mean_absolute_error"
    elif scoring == "rmse":
        scorer = "neg_root_mean_squared_error"
    else:
        raise ValueError("scoring must be 'mae' or 'rmse'")
    
    # Compute permutation importance
    result = sklearn_perm_imp(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=random_seed,
        scoring=scorer,
    )
    
    # Build results DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": -result.importances_mean,  # Negate because sklearn uses negative
        "importance_std": result.importances_std,
    })
    
    # Compute significance (importance > 0)
    # Using t-test: importance_mean / (importance_std / sqrt(n_repeats))
    df["t_stat"] = df["importance_mean"] / (df["importance_std"] / np.sqrt(n_repeats) + 1e-10)
    df["pvalue"] = 1 - stats.t.cdf(df["t_stat"], df=n_repeats - 1)
    df["stars"] = df["pvalue"].apply(get_significance_stars)
    
    return df.set_index("feature").sort_values("importance_mean", ascending=False)


# =============================================================================
# Multiple Testing Correction
# =============================================================================

def apply_fdr_correction(
    pvalues: Union[np.ndarray, pd.Series],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply multiple testing correction using FDR or other methods.
    
    Args:
        pvalues: Array of p-values
        method: Correction method:
            - 'bonferroni': Bonferroni (conservative)
            - 'fdr_bh': Benjamini-Hochberg FDR (recommended)
            - 'fdr_by': Benjamini-Yekutieli FDR
            - 'holm': Holm-Bonferroni
        alpha: Family-wise error rate
    
    Returns:
        Tuple of (reject, corrected_pvalues, alpha_corrected)
    """
    from statsmodels.stats.multitest import multipletests
    
    pvalues = np.asarray(pvalues)
    
    # Handle NaN values
    valid_mask = ~np.isnan(pvalues)
    valid_pvalues = pvalues[valid_mask]
    
    if len(valid_pvalues) == 0:
        return np.array([False] * len(pvalues)), pvalues, alpha
    
    reject_valid, corrected_valid, _, _ = multipletests(
        valid_pvalues, alpha=alpha, method=method
    )
    
    # Map back to original array
    reject = np.array([False] * len(pvalues))
    corrected = np.array([np.nan] * len(pvalues))
    
    reject[valid_mask] = reject_valid
    corrected[valid_mask] = corrected_valid
    
    return reject, corrected, alpha


# =============================================================================
# CV Method Comparison
# =============================================================================

def compare_cv_methods(
    rolling_scores: Dict[str, List[float]],
    expanding_scores: Dict[str, List[float]],
    metric: str = "mae",
) -> Dict[str, any]:
    """
    Compare rolling vs expanding window CV methods.
    
    Args:
        rolling_scores: Dict with metric lists from rolling window CV
        expanding_scores: Dict with metric lists from expanding window CV
        metric: Which metric to compare ("mae" or "rmse")
    
    Returns:
        Dict with 'better_method', 'rolling_mean', 'expanding_mean', 
        'difference', 'pvalue' (paired t-test)
    """
    rolling = np.array(rolling_scores.get(metric, rolling_scores.get("mae", [])))
    expanding = np.array(expanding_scores.get(metric, expanding_scores.get("mae", [])))
    
    # Ensure same length (use min)
    n = min(len(rolling), len(expanding))
    if n == 0:
        return {"better_method": "unknown", "rolling_mean": np.nan, "expanding_mean": np.nan}
    
    rolling = rolling[:n]
    expanding = expanding[:n]
    
    rolling_mean = np.mean(rolling)
    expanding_mean = np.mean(expanding)
    
    # Paired t-test
    if n > 1:
        t_stat, pvalue = stats.ttest_rel(rolling, expanding)
    else:
        t_stat, pvalue = np.nan, np.nan
    
    better = "rolling" if rolling_mean < expanding_mean else "expanding"
    
    return {
        "better_method": better,
        "rolling_mean": float(rolling_mean),
        "expanding_mean": float(expanding_mean),
        "difference": float(expanding_mean - rolling_mean),
        "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
        "pvalue": float(pvalue) if not np.isnan(pvalue) else None,
        "stars": get_significance_stars(pvalue) if not np.isnan(pvalue) else "",
    }


# =============================================================================
# Rolling Window Evaluation
# =============================================================================

def rolling_window_forecast(
    model_class,
    model_params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    initial_train: int = 60,
    step: int = 1,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Perform rolling window out-of-sample forecasting.
    
    Args:
        model_class: Model class to instantiate
        model_params: Parameters for model
        X: Feature matrix
        y: Target vector
        initial_train: Initial training window size
        step: Step size for rolling window
        horizon: Forecast horizon
    
    Returns:
        Tuple of (predictions, actuals, fold_metrics)
    """
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler
    
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n = len(y)
    
    predictions = []
    actuals = []
    fold_metrics = []
    
    train_end = initial_train
    while train_end + horizon <= n:
        # Train on [0, train_end)
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        # Test on [train_end, train_end + horizon)
        test_idx = train_end
        X_test = X[test_idx:test_idx + 1]
        y_test = y[test_idx]
        
        # Fit and predict
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)[0]
        
        predictions.append(pred)
        actuals.append(y_test)
        
        # Compute fold metrics
        fold_metrics.append({
            "train_size": train_end,
            "prediction": pred,
            "actual": y_test,
            "error": y_test - pred,
            "squared_error": (y_test - pred) ** 2,
            "abs_error": abs(y_test - pred),
        })
        
        train_end += step
    
    return np.array(predictions), np.array(actuals), fold_metrics


# =============================================================================
# Overfitting Diagnostics
# =============================================================================

def compute_prediction_quality(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    rolling_window: int = 60,
) -> Dict[str, float]:
    """
    Compute Information Coefficient (IC) and stability metrics.
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        dates: Dates for rolling analysis (optional)
        rolling_window: Window size for rolling IC (default: 60)
    
    Returns:
        Dict with IC, ic_mean, ic_std, ic_stability, rolling_ic series
    """
    # Overall Information Coefficient (Spearman correlation)
    ic, _ = stats.spearmanr(predictions, actuals)
    
    # Rolling IC
    if dates is not None and len(dates) == len(predictions):
        df = pd.DataFrame({'pred': predictions, 'actual': actuals}, index=dates)
        rolling_ic = df.rolling(rolling_window).apply(
            lambda x: stats.spearmanr(x['pred'], x['actual'])[0] if len(x) >= 10 else np.nan,
            raw=False
        )['pred']
        
        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()
        ic_stability = ic_mean / ic_std if ic_std > 0 else 0.0
    else:
        rolling_ic = None
        ic_mean = ic
        ic_std = 0.0
        ic_stability = 0.0
    
    return {
        'information_coefficient': float(ic),
        'ic_mean': float(ic_mean),
        'ic_std': float(ic_std),
        'ic_stability': float(ic_stability),
        'rolling_ic': rolling_ic,
    }


# =============================================================================
# Lead Feature Validation
# =============================================================================

def validate_lead_granger(
    bootstrap_df: pd.DataFrame,
    data_df: pd.DataFrame,
    y_col: str,
    maxlead: int = 3,
    alpha: float = 0.05,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Validate lead features using Granger causality tests.
    
    Extracts significant lead features from bootstrap results and tests whether
    future index values statistically predict current returns using Granger tests.
    
    Args:
        bootstrap_df: DataFrame with bootstrap results (must have 'sig_95' column or 'pvalue' < alpha)
        data_df: DataFrame with time series data (index = dates, columns = features + target)
        y_col: Name of target column in data_df
        maxlead: Maximum lead order to test (default: 3)
        alpha: Significance level for Granger tests (default: 0.05)
        verbose: Print detailed Granger test output
    
    Returns:
        DataFrame with columns: [index_name, lead_order, test_statistic, pvalue, significant]
        where 'significant' indicates if lead Granger-causes target at specified alpha level
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    import warnings
    import re
    
    # Extract lead features from bootstrap results
    # Look for features with '_lead' in name and significance markers
    if 'sig_95' in bootstrap_df.columns:
        lead_features = bootstrap_df[
            (bootstrap_df.index.str.contains('_lead', case=False)) &
            (bootstrap_df['sig_95'] == True)
        ]
    elif 'pvalue' in bootstrap_df.columns:
        lead_features = bootstrap_df[
            (bootstrap_df.index.str.contains('_lead', case=False)) &
            (bootstrap_df['pvalue'] < alpha)
        ]
    else:
        # Fallback: just look for lead features
        lead_features = bootstrap_df[bootstrap_df.index.str.contains('_lead', case=False)]
    
    if len(lead_features) == 0:
        return pd.DataFrame(columns=['index_name', 'lead_order', 'test_statistic', 'pvalue', 'significant'])
    
    results = []
    
    for feat_name in lead_features.index:
        # Parse feature name to extract index name and lead order
        # Format: soc_{index}_lead{k} or {index}_xlead{k}
        match = re.search(r'(.+?)_(?:x)?lead(\d+)', feat_name, re.IGNORECASE)
        if not match:
            continue
        
        base_name = match.group(1).replace('soc_', '').replace('ctl_', '')
        lead_order = int(match.group(2))
        
        # Check if this index column exists in data
        if base_name not in data_df.columns:
            continue
        
        # Test Granger causality: X(t) -> Y(t+lead)
        # Shift Y backward to align Y(t+lead) with X(t)
        df_test = data_df[[y_col, base_name]].copy()
        df_test[y_col] = df_test[y_col].shift(-lead_order)
        df_test = df_test.dropna()
        
        if len(df_test) < 3 * lead_order + 10:
            # Insufficient data
            results.append({
                'index_name': base_name,
                'lead_order': lead_order,
                'test_statistic': np.nan,
                'pvalue': np.nan,
                'significant': False
            })
            continue
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Run Granger test with lag=1
                result = grangercausalitytests(df_test, maxlag=1, verbose=verbose)
            
            # Extract F-test results
            test_stat = result[1][0]['ssr_ftest'][0]
            pvalue = result[1][0]['ssr_ftest'][1]
            
            results.append({
                'index_name': base_name,
                'lead_order': lead_order,
                'test_statistic': float(test_stat),
                'pvalue': float(pvalue),
                'significant': pvalue < alpha
            })
        except Exception as e:
            results.append({
                'index_name': base_name,
                'lead_order': lead_order,
                'test_statistic': np.nan,
                'pvalue': np.nan,
                'significant': False
            })
    
    return pd.DataFrame(results)


def generate_lag_lead_interpretation_table(
    bootstrap_df: pd.DataFrame,
    granger_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Generate interpretation table summarizing lag and lead effects per index.
    
    Identifies which indices have significant lag features, lead features, or both,
    and provides interpretation guidance for each pattern.
    
    Args:
        bootstrap_df: DataFrame with bootstrap coefficient results (index = feature names)
                     Must have 'sig_95' column or 'pvalue' column for significance
        granger_df: Optional DataFrame from validate_lead_granger() with Granger test results
        alpha: Significance level (default: 0.05)
    
    Returns:
        DataFrame with columns: [index, max_sig_lag, max_sig_lead, pattern, interpretation]
        where pattern is one of: "Lag-only", "Lead-only", "Both", or "Neither"
    """
    import re
    
    # Extract significance info
    if 'sig_95' in bootstrap_df.columns:
        sig_features = bootstrap_df[bootstrap_df['sig_95'] == True]
    elif 'pvalue' in bootstrap_df.columns:
        sig_features = bootstrap_df[bootstrap_df['pvalue'] < alpha]
    else:
        # Fallback: use all features
        sig_features = bootstrap_df
    
    # Parse feature names to extract index names and lag/lead orders
    index_effects = {}
    
    for feat_name in sig_features.index:
        # Match lag features: soc_{index}_lag{k} or {index}_xlag{k}
        lag_match = re.search(r'(.+?)_(?:x)?lag(\d+)', feat_name, re.IGNORECASE)
        if lag_match:
            base_name = lag_match.group(1).replace('soc_', '').replace('ctl_', '')
            lag_order = int(lag_match.group(2))
            
            if base_name not in index_effects:
                index_effects[base_name] = {'lags': [], 'leads': []}
            index_effects[base_name]['lags'].append(lag_order)
            continue
        
        # Match lead features: soc_{index}_lead{k} or {index}_xlead{k}
        lead_match = re.search(r'(.+?)_(?:x)?lead(\d+)', feat_name, re.IGNORECASE)
        if lead_match:
            base_name = lead_match.group(1).replace('soc_', '').replace('ctl_', '')
            lead_order = int(lead_match.group(2))
            
            if base_name not in index_effects:
                index_effects[base_name] = {'lags': [], 'leads': []}
            index_effects[base_name]['leads'].append(lead_order)
    
    # Build interpretation table
    results = []
    
    for index_name, effects in index_effects.items():
        max_sig_lag = max(effects['lags']) if effects['lags'] else 0
        max_sig_lead = max(effects['leads']) if effects['leads'] else 0
        
        # Determine pattern
        if max_sig_lag > 0 and max_sig_lead > 0:
            pattern = "Both"
            interpretation = (
                f"Both lagged (up to t-{max_sig_lag}) and lead (up to t+{max_sig_lead}) effects detected. "
                "Suggests complex temporal relationship with both historical influence and anticipatory signals. "
                "May indicate forward-looking discourse or common latent factors."
            )
        elif max_sig_lag > 0:
            pattern = "Lag-only"
            interpretation = (
                f"Only lagged effects (up to t-{max_sig_lag}) detected. "
                "Standard causal interpretation: past index values predict future returns. "
                "Consistent with social signals influencing market behavior."
            )
        elif max_sig_lead > 0:
            pattern = "Lead-only"
            # Check if Granger validated
            granger_validated = False
            if granger_df is not None:
                granger_validated = granger_df[
                    (granger_df['index_name'] == index_name) &
                    (granger_df['significant'] == True)
                ].shape[0] > 0
            
            if granger_validated:
                interpretation = (
                    f"Only lead effects (up to t+{max_sig_lead}) detected and Granger-validated. "
                    "Suggests anticipatory social signals or reverse causality (returns precede social response). "
                    "May also indicate measurement timing issues or common drivers."
                )
            else:
                interpretation = (
                    f"Only lead effects (up to t+{max_sig_lead}) detected. "
                    "May indicate anticipatory signals, reverse causality, or potential data leakage. "
                    "Requires careful validation in out-of-sample predictions."
                )
        else:
            pattern = "Neither"
            interpretation = "No significant lag or lead effects detected in the model."
        
        results.append({
            'index': index_name,
            'max_sig_lag': max_sig_lag,
            'max_sig_lead': max_sig_lead,
            'pattern': pattern,
            'interpretation': interpretation
        })
    
    return pd.DataFrame(results)
