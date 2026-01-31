"""
Statistical tests for time series analysis.

Provides functions for:
- Stationarity testing (ADF, KPSS)
- Granger causality with AIC/BIC lag selection
- Cointegration testing (Johansen)
- Multiple testing correction (FDR, Bonferroni)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import warnings
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


def adf_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    
    Args:
        series: Time series to test
        maxlag: Maximum lag order. If None, uses AIC
    
    Returns:
        dict with 'statistic', 'pvalue', 'lags', 'nobs', 'critical_values', 'is_stationary'
    """
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return {
            'statistic': np.nan,
            'pvalue': np.nan,
            'lags': np.nan,
            'nobs': len(series_clean),
            'critical_values': {},
            'is_stationary': False,
            'error': 'Insufficient data'
        }
    
    try:
        result = adfuller(series_clean, maxlag=maxlag, autolag='AIC')
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'lags': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'pvalue': np.nan,
            'lags': np.nan,
            'nobs': len(series_clean),
            'critical_values': {},
            'is_stationary': False,
            'error': str(e)
        }


def kpss_test(series: pd.Series, regression: str = 'c', nlags: str = 'auto') -> Dict:
    """
    KPSS test for stationarity.
    
    H0: Series is stationary
    H1: Series has a unit root (non-stationary)
    
    Note: Opposite hypothesis to ADF test!
    
    Args:
        series: Time series to test
        regression: 'c' for constant, 'ct' for constant+trend
        nlags: Number of lags. 'auto' uses Schwert criterion
    
    Returns:
        dict with 'statistic', 'pvalue', 'lags', 'critical_values', 'is_stationary'
    """
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return {
            'statistic': np.nan,
            'pvalue': np.nan,
            'lags': np.nan,
            'critical_values': {},
            'is_stationary': False,
            'error': 'Insufficient data'
        }
    
    try:
        result = kpss(series_clean, regression=regression, nlags=nlags)
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'lags': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05  # KPSS: null = stationary
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'pvalue': np.nan,
            'lags': np.nan,
            'critical_values': {},
            'is_stationary': False,
            'error': str(e)
        }


def _test_granger_leads(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maxlead: int,
    ic: str,
    verbose: bool
) -> Dict:
    """
    Helper function to test if X(t) predicts Y(t+k) for k=1 to maxlead.
    
    Shifts Y backward by k periods to align Y(t+k) with X(t), then runs
    standard Granger test with lag=1.
    
    Args:
        df: DataFrame with time series
        x_col: Predictor variable
        y_col: Target variable
        maxlead: Maximum lead to test
        ic: 'aic' or 'bic'
        verbose: Print detailed test output
    
    Returns:
        dict with lead test results
    """
    pvalues = {}
    ic_values = {}
    
    for lead in range(1, maxlead + 1):
        # Shift Y backward to align Y(t+lead) with X(t)
        df_shifted = df[[y_col, x_col]].copy()
        df_shifted[y_col] = df_shifted[y_col].shift(-lead)
        data = df_shifted.dropna()
        
        if len(data) < 3 * lead:
            pvalues[lead] = np.nan
            ic_values[lead] = np.nan
            continue
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                result = grangercausalitytests(data, maxlag=1, verbose=verbose)
            
            pvalues[lead] = result[1][0]['ssr_ftest'][1]
            
            if ic == 'aic':
                ic_values[lead] = result[1][1][1].aic
            else:
                ic_values[lead] = result[1][1][1].bic
        except:
            pvalues[lead] = np.nan
            ic_values[lead] = np.nan
    
    # Find optimal lead
    valid_leads = {k: v for k, v in ic_values.items() if not np.isnan(v)}
    optimal_lead = min(valid_leads, key=valid_leads.get) if valid_leads else None
    
    valid_pvalues = [p for p in pvalues.values() if not np.isnan(p)]
    min_pvalue = min(valid_pvalues) if valid_pvalues else np.nan
    
    return {
        'optimal_lead': optimal_lead,
        'optimal_lead_pvalue': pvalues.get(optimal_lead, np.nan) if optimal_lead else np.nan,
        'pvalues': pvalues,
        'ic_values': ic_values,
        'min_pvalue': min_pvalue,
        'granger_causes': any(p < 0.05 for p in valid_pvalues),
        'granger_causes_at_optimal_lead': pvalues.get(optimal_lead, np.nan) < 0.05 if optimal_lead else False
    }


def granger_causality_test(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maxlag: int = 10,
    maxlead: int = 10,
    ic: str = 'aic',
    alpha: float = 0.05,
    verbose: bool = False
) -> Dict:
    """
    Granger causality test: Does x Granger-cause y?
    
    Tests whether past values of x help predict y beyond y's own past values (lags),
    and optionally whether current x predicts future y (leads).
    
    **Lag tests** (X(t-k) → Y(t)): Does past X predict current Y? Traditional Granger causality.
    **Lead tests** (X(t) → Y(t+k)): Does current X predict future Y?
    
    IMPORTANT: Significant leads may indicate:
    - Y precedes X (reverse causality)
    - Anticipatory behavior (X anticipates future Y)
    - Common underlying driver affecting both
    - NOT evidence of forward causality from X to Y
    
    Args:
        df: DataFrame with time series
        x_col: Potential cause variable (predictor)
        y_col: Effect variable (target)
        maxlag: Maximum lag to test (default 10)
        maxlead: Maximum lead to test (default 10). Set to 0 to skip lead tests.
        ic: Information criterion ('aic' or 'bic') for optimal lag selection
        verbose: Print detailed test output
    
    Returns:
        dict with:
            If maxlead=0 (backward compatible):
                - optimal_lag: Best lag based on IC
                - optimal_lag_pvalue: P-value at optimal lag
                - pvalues: Dict of p-values for each lag
                - ic_values: Dict of IC values for each lag
                - min_pvalue: Minimum p-value across all lags
                - granger_causes: Boolean (True if any lag significant at 5%)
                - granger_causes_at_optimal_lag: Boolean at optimal lag
            
            If maxlead>0:
                - lag_results: Dict with lag test results (structure above)
                - lead_results: Dict with lead test results (similar structure)
                - best_direction: 'lag' or 'lead' based on IC comparison
                - best_order: Optimal lag or lead value
    """
    data = df[[y_col, x_col]].dropna()
    
    if len(data) < maxlag * 3:
        error_dict = {
            'optimal_lag': None,
            'optimal_lag_pvalue': np.nan,
            'pvalues': {},
            'ic_values': {},
            'min_pvalue': np.nan,
            'granger_causes': False,
            'granger_causes_at_optimal_lag': False,
            'error': f'Insufficient data: {len(data)} obs, need {maxlag * 3}'
        }
        if maxlead > 0:
            return {
                'lag_results': error_dict,
                'lead_results': error_dict.copy(),
                'best_direction': None,
                'best_order': None
            }
        return error_dict
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)
        
        # Extract p-values and IC values
        pvalues = {}
        ic_values = {}
        
        for lag in range(1, maxlag + 1):
            # F-test p-value: result[lag][0]['ssr_ftest'] = (F-stat, p-value, df_denom, df_num)
            pvalues[lag] = result[lag][0]['ssr_ftest'][1]
            
            # Extract IC from restricted model (without x)
            # result[lag][1] = [restricted_model, unrestricted_model]
            try:
                if ic == 'aic':
                    ic_values[lag] = result[lag][1][1].aic  # unrestricted model AIC
                else:
                    ic_values[lag] = result[lag][1][1].bic  # unrestricted model BIC
            except:
                ic_values[lag] = np.nan
        
        # Select optimal lag based on IC (lower is better)
        valid_lags = {k: v for k, v in ic_values.items() if not np.isnan(v)}
        optimal_lag = min(valid_lags, key=valid_lags.get) if valid_lags else 1
        min_pvalue = min(pvalues.values())
        
        lag_results = {
            'optimal_lag': optimal_lag,
            'optimal_lag_pvalue': pvalues[optimal_lag],
            'pvalues': pvalues,
            'ic_values': ic_values,
            'min_pvalue': min_pvalue,
            'granger_causes': min_pvalue < alpha,
            'granger_causes_at_optimal_lag': pvalues[optimal_lag] < alpha
        }
    
    except Exception as e:
        lag_results = {
            'optimal_lag': None,
            'optimal_lag_pvalue': np.nan,
            'pvalues': {},
            'ic_values': {},
            'min_pvalue': np.nan,
            'granger_causes': False,
            'granger_causes_at_optimal_lag': False,
            'error': str(e)
        }
    
    # Return early if no lead tests requested (backward compatible)
    if maxlead == 0:
        return lag_results
    
    # Test leads (new functionality)
    lead_results = _test_granger_leads(df, x_col, y_col, maxlead, ic, verbose)
    
    # Determine best direction based on IC
    best_lag_ic = lag_results.get('ic_values', {}).get(lag_results.get('optimal_lag'), np.inf)
    best_lead_ic = lead_results.get('ic_values', {}).get(lead_results.get('optimal_lead'), np.inf)
    
    if np.isnan(best_lag_ic):
        best_lag_ic = np.inf
    if np.isnan(best_lead_ic):
        best_lead_ic = np.inf
    
    if best_lag_ic <= best_lead_ic:
        best_direction = 'lag'
        best_order = lag_results.get('optimal_lag')
    else:
        best_direction = 'lead'
        best_order = lead_results.get('optimal_lead')
    
    return {
        'lag_results': lag_results,
        'lead_results': lead_results,
        'best_direction': best_direction,
        'best_order': best_order
    }


def bidirectional_granger_test(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    maxlag: int = 10,
    maxlead: int = 10,
    ic: str = 'aic',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test Granger causality in both directions with both lags and leads.
    
    Args:
        df: DataFrame with time series
        col1: First variable
        col2: Second variable
        maxlag: Maximum lag to test
        maxlead: Maximum lead to test
        ic: 'aic' or 'bic' for lag selection
        alpha: Significance level for hypothesis testing
    Returns:
        DataFrame with results for both directions and both types (lags + leads)
        If maxlead=0, returns 2 rows (backward compatible)
        If maxlead>0, returns 4 rows (2 directions × 2 types)
    """
    result1 = granger_causality_test(df, col1, col2, maxlag=maxlag, maxlead=maxlead, ic=ic, alpha=alpha)
    result2 = granger_causality_test(df, col2, col1, maxlag=maxlag, maxlead=maxlead, ic=ic, alpha=alpha)
    
    # Backward compatible: if maxlead=0, return old format
    if maxlead == 0:
        summary = pd.DataFrame({
            'direction': [f'{col1} → {col2}', f'{col2} → {col1}'],
            'optimal_lag': [result1.get('optimal_lag'), result2.get('optimal_lag')],
            'pvalue_at_optimal': [result1.get('optimal_lag_pvalue'), result2.get('optimal_lag_pvalue')],
            'min_pvalue': [result1.get('min_pvalue'), result2.get('min_pvalue')],
            'granger_causes': [result1.get('granger_causes'), result2.get('granger_causes')],
            'granger_causes_at_optimal': [result1.get('granger_causes_at_optimal_lag'), 
                                           result2.get('granger_causes_at_optimal_lag')]
        })
        return summary
    
    # New format with leads
    rows = []
    
    # Direction 1: col1 → col2
    lag1 = result1['lag_results']
    lead1 = result1['lead_results']
    rows.append({
        'direction': f'{col1} → {col2}',
        'type': 'lag',
        'optimal_order': lag1.get('optimal_lag'),
        'pvalue_at_optimal': lag1.get('optimal_lag_pvalue'),
        'min_pvalue': lag1.get('min_pvalue'),
        'granger_causes': lag1.get('granger_causes'),
        'granger_causes_at_optimal': lag1.get('granger_causes_at_optimal_lag')
    })
    rows.append({
        'direction': f'{col1} → {col2}',
        'type': 'lead',
        'optimal_order': lead1.get('optimal_lead'),
        'pvalue_at_optimal': lead1.get('optimal_lead_pvalue'),
        'min_pvalue': lead1.get('min_pvalue'),
        'granger_causes': lead1.get('granger_causes'),
        'granger_causes_at_optimal': lead1.get('granger_causes_at_optimal_lead')
    })
    
    # Direction 2: col2 → col1
    lag2 = result2['lag_results']
    lead2 = result2['lead_results']
    rows.append({
        'direction': f'{col2} → {col1}',
        'type': 'lag',
        'optimal_order': lag2.get('optimal_lag'),
        'pvalue_at_optimal': lag2.get('optimal_lag_pvalue'),
        'min_pvalue': lag2.get('min_pvalue'),
        'granger_causes': lag2.get('granger_causes'),
        'granger_causes_at_optimal': lag2.get('granger_causes_at_optimal_lag')
    })
    rows.append({
        'direction': f'{col2} → {col1}',
        'type': 'lead',
        'optimal_order': lead2.get('optimal_lead'),
        'pvalue_at_optimal': lead2.get('optimal_lead_pvalue'),
        'min_pvalue': lead2.get('min_pvalue'),
        'granger_causes': lead2.get('granger_causes'),
        'granger_causes_at_optimal': lead2.get('granger_causes_at_optimal_lead')
    })
    
    summary = pd.DataFrame(rows)
    return summary


def johansen_cointegration_test(
    df: pd.DataFrame,
    cols: List[str],
    det_order: int = 0,
    k_ar_diff: int = 1
) -> Dict:
    """
    Johansen cointegration test for multiple time series.
    
    Tests for long-run equilibrium relationships between variables.
    
    Args:
        df: DataFrame
        cols: List of column names to test
        det_order: Deterministic term order
            -1: no deterministic terms
            0: constant term
            1: constant + linear trend
        k_ar_diff: Number of lags in differenced VAR
    
    Returns:
        dict with trace/eigenvalue statistics and critical values
    """
    data = df[cols].dropna()
    
    if len(data) < 30:
        return {
            'error': f'Insufficient data: {len(data)} obs, need ≥30',
            'n_cointegrating_trace': 0,
            'n_cointegrating_eigen': 0,
            'is_cointegrated': False
        }
    
    try:
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Check trace statistic at 5% level
        # cvt[:, 1] is 5% critical value column
        n_coint_trace = int(np.sum(result.lr1 > result.cvt[:, 1]))
        
        # Check max eigenvalue statistic at 5% level  
        # cvm[:, 1] is 5% critical value column
        n_coint_eigen = int(np.sum(result.lr2 > result.cvm[:, 1]))
        
        return {
            'trace_stat': result.lr1.tolist(),
            'trace_crit_5pct': result.cvt[:, 1].tolist(),
            'max_eigen_stat': result.lr2.tolist(),
            'max_eigen_crit_5pct': result.cvm[:, 1].tolist(),
            'n_cointegrating_trace': n_coint_trace,
            'n_cointegrating_eigen': n_coint_eigen,
            'is_cointegrated': n_coint_trace > 0 or n_coint_eigen > 0
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'n_cointegrating_trace': 0,
            'n_cointegrating_eigen': 0,
            'is_cointegrated': False
        }


def granger_causality_matrix(
    df: pd.DataFrame,
    index_cols: List[str],
    asset_cols: List[str],
    maxlag: int = 10,
    maxlead: int = 10,
    ic: str = 'aic',
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Test Granger causality from each index to each asset with lags and leads.
    Apply multiple testing correction.
    
    Args:
        df: DataFrame
        index_cols: Emotion/sentiment indices (predictors)
        asset_cols: Asset returns (targets)
        maxlag: Max lag for Granger test (default 10)
        maxlead: Max lead for Granger test (default 10). Set to 0 for backward compatibility.
        ic: 'aic' or 'bic' for lag selection
        alpha: Significance level (default 0.05)
        correction_method: Method for multiple testing correction
            'bonferroni': Conservative correction
            'fdr_bh': Benjamini-Hochberg FDR control (recommended)
            'fdr_by': Benjamini-Yekutieli FDR control
            'holm': Holm-Bonferroni
            'sidak': Sidak correction
    
    Returns:
        If maxlead=0 (backward compatible):
            Tuple of (pvalue_matrix, pvalue_corrected_matrix, significance_matrix)
        
        If maxlead>0:
            Tuple of 6 DataFrames:
                - lag_pvalue_matrix: Raw p-values at optimal lag
                - lag_pvalue_corrected_matrix: Corrected p-values for lags
                - lag_significance_matrix: Binary matrix for lags
                - lead_pvalue_matrix: Raw p-values at optimal lead
                - lead_pvalue_corrected_matrix: Corrected p-values for leads
                - lead_significance_matrix: Binary matrix for leads
    """
    n_tests = len(index_cols) * len(asset_cols)
    print(f"Running {n_tests} Granger causality tests...")
    print(f"  Index → Asset causality")
    print(f"  Max lag: {maxlag}, Max lead: {maxlead}, IC: {ic.upper()}")
    print(f"  Multiple testing correction: {correction_method}")
    
    results = []
    
    for idx_col in index_cols:
        for asset_col in asset_cols:
            result = granger_causality_test(df, idx_col, asset_col, maxlag=maxlag, maxlead=maxlead, ic=ic)
            
            # Handle backward compatible format (maxlead=0)
            if maxlead == 0:
                results.append({
                    'index': idx_col,
                    'asset': asset_col,
                    'optimal_lag': result.get('optimal_lag'),
                    'pvalue': result.get('optimal_lag_pvalue', np.nan),
                    'min_pvalue': result.get('min_pvalue', np.nan)
                })
            else:
                # New format with lags and leads
                lag_res = result.get('lag_results', {})
                lead_res = result.get('lead_results', {})
                results.append({
                    'index': idx_col,
                    'asset': asset_col,
                    'optimal_lag': lag_res.get('optimal_lag'),
                    'pvalue_lag': lag_res.get('optimal_lag_pvalue', np.nan),
                    'min_pvalue_lag': lag_res.get('min_pvalue', np.nan),
                    'optimal_lead': lead_res.get('optimal_lead'),
                    'pvalue_lead': lead_res.get('optimal_lead_pvalue', np.nan),
                    'min_pvalue_lead': lead_res.get('min_pvalue', np.nan)
                })
    
    results_df = pd.DataFrame(results)
    
    # Backward compatible return (maxlead=0)
    if maxlead == 0:
        pvalue_matrix = results_df.pivot(index='index', columns='asset', values='pvalue')
        
        # Apply multiple testing correction
        pvalues_flat = pvalue_matrix.values.flatten()
        valid_mask = ~np.isnan(pvalues_flat)
        corrected_flat = np.full_like(pvalues_flat, np.nan)
        
        if valid_mask.sum() > 0:
            reject, pvals_corrected, _, _ = multipletests(
                pvalues_flat[valid_mask], alpha=alpha, method=correction_method
            )
            corrected_flat[valid_mask] = pvals_corrected
        
        pvalue_corrected_matrix = pd.DataFrame(
            corrected_flat.reshape(pvalue_matrix.shape),
            index=pvalue_matrix.index, columns=pvalue_matrix.columns
        )
        significance_matrix = (pvalue_corrected_matrix < alpha).astype(int)
        
        print(f"\n✓ Tests complete")
        print(f"  Significant relationships (corrected α={alpha}): {significance_matrix.sum().sum()} of {n_tests}")
        print(f"  Uncorrected significant (α={alpha}): {(pvalue_matrix < alpha).sum().sum()}")
        
        return pvalue_matrix, pvalue_corrected_matrix, significance_matrix
    
    # New format: separate matrices for lags and leads
    lag_pvalue_matrix = results_df.pivot(index='index', columns='asset', values='pvalue_lag')
    lead_pvalue_matrix = results_df.pivot(index='index', columns='asset', values='pvalue_lead')
    
    # Apply multiple testing correction to lags
    lag_pvalues_flat = lag_pvalue_matrix.values.flatten()
    lag_valid_mask = ~np.isnan(lag_pvalues_flat)
    lag_corrected_flat = np.full_like(lag_pvalues_flat, np.nan)
    
    if lag_valid_mask.sum() > 0:
        reject, pvals_corrected, _, _ = multipletests(
            lag_pvalues_flat[lag_valid_mask], alpha=alpha, method=correction_method
        )
        lag_corrected_flat[lag_valid_mask] = pvals_corrected
    
    lag_pvalue_corrected_matrix = pd.DataFrame(
        lag_corrected_flat.reshape(lag_pvalue_matrix.shape),
        index=lag_pvalue_matrix.index, columns=lag_pvalue_matrix.columns
    )
    lag_significance_matrix = (lag_pvalue_corrected_matrix < alpha).astype(int)
    
    # Apply multiple testing correction to leads
    lead_pvalues_flat = lead_pvalue_matrix.values.flatten()
    lead_valid_mask = ~np.isnan(lead_pvalues_flat)
    lead_corrected_flat = np.full_like(lead_pvalues_flat, np.nan)
    
    if lead_valid_mask.sum() > 0:
        reject, pvals_corrected, _, _ = multipletests(
            lead_pvalues_flat[lead_valid_mask], alpha=alpha, method=correction_method
        )
        lead_corrected_flat[lead_valid_mask] = pvals_corrected
    
    lead_pvalue_corrected_matrix = pd.DataFrame(
        lead_corrected_flat.reshape(lead_pvalue_matrix.shape),
        index=lead_pvalue_matrix.index, columns=lead_pvalue_matrix.columns
    )
    lead_significance_matrix = (lead_pvalue_corrected_matrix < alpha).astype(int)
    
    print(f"\n✓ Tests complete")
    print(f"  LAG tests - Significant relationships (corrected α={alpha}): {lag_significance_matrix.sum().sum()} of {n_tests}")
    print(f"  LAG tests - Uncorrected significant (α={alpha}): {(lag_pvalue_matrix < alpha).sum().sum()}")
    print(f"  LEAD tests - Significant relationships (corrected α={alpha}): {lead_significance_matrix.sum().sum()} of {n_tests}")
    print(f"  LEAD tests - Uncorrected significant (α={alpha}): {(lead_pvalue_matrix < alpha).sum().sum()}")
    
    return (lag_pvalue_matrix, lag_pvalue_corrected_matrix, lag_significance_matrix,
            lead_pvalue_matrix, lead_pvalue_corrected_matrix, lead_significance_matrix)


def plot_granger_matrix(
    pvalue_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    title: str = "Granger Causality",
    figsize: Tuple[int, int] = (14, 6),
    lead_pvalue_matrix: Optional[pd.DataFrame] = None,
    lead_significance_matrix: Optional[pd.DataFrame] = None
):
    """
    Visualize Granger causality results.
    
    Args:
        pvalue_matrix: Raw or corrected p-values for lags
        significance_matrix: Binary significance matrix for lags
        title: Plot title
        figsize: Figure size (will be adjusted if leads are included)
        lead_pvalue_matrix: Optional p-values for leads
        lead_significance_matrix: Optional binary significance matrix for leads
    """
    # If leads provided, create 2x2 subplot grid
    if lead_pvalue_matrix is not None and lead_significance_matrix is not None:
        figsize = (14, 12)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = [axes[0], axes[1]]
    
    # Plot lag results
    ax1, ax2 = axes[0], axes[1]
    
    # P-values (log scale for visibility)
    # -log10(p) so that small p-values (strong evidence) appear large
    log_pvals = -np.log10(pvalue_matrix.replace(0, 1e-10) + 1e-10)
    
    im1 = ax1.imshow(log_pvals, aspect='auto', cmap='RdYlGn', vmin=0, vmax=3)
    ax1.set_xticks(range(len(pvalue_matrix.columns)))
    ax1.set_xticklabels(pvalue_matrix.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(pvalue_matrix.index)))
    ax1.set_yticklabels(pvalue_matrix.index)
    ax1.set_title(f'{title} — LAG: X(t-k) → Y(t)\n-log₁₀(p-value)')
    ax1.set_xlabel('Asset (Target)')
    ax1.set_ylabel('Index (Predictor)')
    
    # Add horizontal lines
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    for i in range(1, len(pvalue_matrix.index)):
        ax1.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('-log₁₀(p)', rotation=270, labelpad=15)
    
    # Significance after correction (lags)
    im2 = ax2.imshow(significance_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_xticks(range(len(significance_matrix.columns)))
    ax2.set_xticklabels(significance_matrix.columns, rotation=45, ha='right')
    ax2.set_yticks(range(len(significance_matrix.index)))
    ax2.set_yticklabels(significance_matrix.index)
    ax2.set_title(f'{title} — LAG: X(t-k) → Y(t)\nSignificant (corrected)')
    ax2.set_xlabel('Asset (Target)')
    ax2.set_ylabel('Index (Predictor)')
    
    # Add grid
    for i in range(1, len(significance_matrix.index)):
        ax2.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    for j in range(1, len(significance_matrix.columns)):
        ax2.axvline(x=j-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar2.set_ticklabels(['No', 'Yes'])
    
    # Plot lead results if provided
    if lead_pvalue_matrix is not None and lead_significance_matrix is not None:
        ax3, ax4 = axes[2], axes[3]
        
        # Lead p-values
        log_pvals_lead = -np.log10(lead_pvalue_matrix.replace(0, 1e-10) + 1e-10)
        im3 = ax3.imshow(log_pvals_lead, aspect='auto', cmap='RdYlGn', vmin=0, vmax=3)
        ax3.set_xticks(range(len(lead_pvalue_matrix.columns)))
        ax3.set_xticklabels(lead_pvalue_matrix.columns, rotation=45, ha='right')
        ax3.set_yticks(range(len(lead_pvalue_matrix.index)))
        ax3.set_yticklabels(lead_pvalue_matrix.index)
        ax3.set_title(f'{title} — LEAD: X(t) → Y(t+k)\n-log₁₀(p-value)')
        ax3.set_xlabel('Asset (Target)')
        ax3.set_ylabel('Index (Predictor)')
        
        for i in range(1, len(lead_pvalue_matrix.index)):
            ax3.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('-log₁₀(p)', rotation=270, labelpad=15)
        
        # Lead significance
        im4 = ax4.imshow(lead_significance_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax4.set_xticks(range(len(lead_significance_matrix.columns)))
        ax4.set_xticklabels(lead_significance_matrix.columns, rotation=45, ha='right')
        ax4.set_yticks(range(len(lead_significance_matrix.index)))
        ax4.set_yticklabels(lead_significance_matrix.index)
        ax4.set_title(f'{title} — LEAD: X(t) → Y(t+k)\nSignificant (corrected)')
        ax4.set_xlabel('Asset (Target)')
        ax4.set_ylabel('Index (Predictor)')
        
        for i in range(1, len(lead_significance_matrix.index)):
            ax4.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        for j in range(1, len(lead_significance_matrix.columns)):
            ax4.axvline(x=j-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        cbar4 = plt.colorbar(im4, ax=ax4, ticks=[0, 1])
        cbar4.set_ticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    plt.show()
