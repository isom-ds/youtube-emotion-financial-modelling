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


def granger_causality_test(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maxlag: int = 10,
    ic: str = 'aic',
    verbose: bool = False
) -> Dict:
    """
    Granger causality test: Does x Granger-cause y?
    
    Tests whether past values of x help predict y beyond y's own past values.
    
    Args:
        df: DataFrame with time series
        x_col: Potential cause variable (predictor)
        y_col: Effect variable (target)
        maxlag: Maximum lag to test
        ic: Information criterion ('aic' or 'bic') for optimal lag selection
        verbose: Print detailed test output
    
    Returns:
        dict with:
            - optimal_lag: Best lag based on IC
            - optimal_lag_pvalue: P-value at optimal lag
            - pvalues: Dict of p-values for each lag
            - ic_values: Dict of IC values for each lag
            - min_pvalue: Minimum p-value across all lags
            - granger_causes: Boolean (True if any lag significant at 5%)
            - granger_causes_at_optimal_lag: Boolean at optimal lag
    """
    data = df[[y_col, x_col]].dropna()
    
    if len(data) < maxlag * 3:
        return {
            'optimal_lag': None,
            'optimal_lag_pvalue': np.nan,
            'pvalues': {},
            'ic_values': {},
            'min_pvalue': np.nan,
            'granger_causes': False,
            'granger_causes_at_optimal_lag': False,
            'error': f'Insufficient data: {len(data)} obs, need {maxlag * 3}'
        }
    
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
        
        return {
            'optimal_lag': optimal_lag,
            'optimal_lag_pvalue': pvalues[optimal_lag],
            'pvalues': pvalues,
            'ic_values': ic_values,
            'min_pvalue': min_pvalue,
            'granger_causes': min_pvalue < 0.05,
            'granger_causes_at_optimal_lag': pvalues[optimal_lag] < 0.05
        }
    
    except Exception as e:
        return {
            'optimal_lag': None,
            'optimal_lag_pvalue': np.nan,
            'pvalues': {},
            'ic_values': {},
            'min_pvalue': np.nan,
            'granger_causes': False,
            'granger_causes_at_optimal_lag': False,
            'error': str(e)
        }


def bidirectional_granger_test(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    maxlag: int = 10,
    ic: str = 'aic'
) -> pd.DataFrame:
    """
    Test Granger causality in both directions.
    
    Args:
        df: DataFrame with time series
        col1: First variable
        col2: Second variable
        maxlag: Maximum lag to test
        ic: 'aic' or 'bic' for lag selection
    
    Returns:
        DataFrame with results for both directions
    """
    result1 = granger_causality_test(df, col1, col2, maxlag=maxlag, ic=ic)
    result2 = granger_causality_test(df, col2, col1, maxlag=maxlag, ic=ic)
    
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
    ic: str = 'aic',
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Test Granger causality from each index to each asset.
    Apply multiple testing correction.
    
    Args:
        df: DataFrame
        index_cols: Emotion/sentiment indices (predictors)
        asset_cols: Asset returns (targets)
        maxlag: Max lag for Granger test
        ic: 'aic' or 'bic' for lag selection
        alpha: Significance level (default 0.05)
        correction_method: Method for multiple testing correction
            'bonferroni': Conservative correction
            'fdr_bh': Benjamini-Hochberg FDR control (recommended)
            'fdr_by': Benjamini-Yekutieli FDR control
            'holm': Holm-Bonferroni
            'sidak': Sidak correction
    
    Returns:
        Tuple of:
            - pvalue_matrix: Raw p-values at optimal lag
            - pvalue_corrected_matrix: Corrected p-values
            - significance_matrix: Binary matrix (1=significant after correction)
    """
    n_tests = len(index_cols) * len(asset_cols)
    print(f"Running {n_tests} Granger causality tests...")
    print(f"  Index → Asset causality")
    print(f"  Max lag: {maxlag}, IC: {ic.upper()}")
    print(f"  Multiple testing correction: {correction_method}")
    
    results = []
    
    for idx_col in index_cols:
        for asset_col in asset_cols:
            result = granger_causality_test(df, idx_col, asset_col, maxlag=maxlag, ic=ic)
            results.append({
                'index': idx_col,
                'asset': asset_col,
                'optimal_lag': result.get('optimal_lag'),
                'pvalue': result.get('optimal_lag_pvalue', np.nan),
                'min_pvalue': result.get('min_pvalue', np.nan)
            })
    
    results_df = pd.DataFrame(results)
    
    # Pivot to matrix form
    pvalue_matrix = results_df.pivot(index='index', columns='asset', values='pvalue')
    min_pvalue_matrix = results_df.pivot(index='index', columns='asset', values='min_pvalue')
    
    # Apply multiple testing correction
    pvalues_flat = pvalue_matrix.values.flatten()
    valid_mask = ~np.isnan(pvalues_flat)
    
    corrected_flat = np.full_like(pvalues_flat, np.nan)
    
    if valid_mask.sum() > 0:
        reject, pvals_corrected, _, _ = multipletests(
            pvalues_flat[valid_mask],
            alpha=alpha,
            method=correction_method
        )
        corrected_flat[valid_mask] = pvals_corrected
    
    pvalue_corrected_matrix = pd.DataFrame(
        corrected_flat.reshape(pvalue_matrix.shape),
        index=pvalue_matrix.index,
        columns=pvalue_matrix.columns
    )
    
    # Significance matrix
    significance_matrix = (pvalue_corrected_matrix < alpha).astype(int)
    
    print(f"\n✓ Tests complete")
    print(f"  Significant relationships (corrected α={alpha}): {significance_matrix.sum().sum()} of {n_tests}")
    print(f"  Uncorrected significant (α={alpha}): {(pvalue_matrix < alpha).sum().sum()}")
    
    return pvalue_matrix, pvalue_corrected_matrix, significance_matrix


def plot_granger_matrix(
    pvalue_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    title: str = "Granger Causality",
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Visualize Granger causality results.
    
    Args:
        pvalue_matrix: Raw or corrected p-values
        significance_matrix: Binary significance matrix
        title: Plot title
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # P-values (log scale for visibility)
    # -log10(p) so that small p-values (strong evidence) appear large
    log_pvals = -np.log10(pvalue_matrix.replace(0, 1e-10) + 1e-10)
    
    im1 = ax1.imshow(log_pvals, aspect='auto', cmap='RdYlGn', vmin=0, vmax=3)
    ax1.set_xticks(range(len(pvalue_matrix.columns)))
    ax1.set_xticklabels(pvalue_matrix.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(pvalue_matrix.index)))
    ax1.set_yticklabels(pvalue_matrix.index)
    ax1.set_title(f'{title}\n-log₁₀(p-value)')
    ax1.set_xlabel('Asset (Target)')
    ax1.set_ylabel('Index (Predictor)')
    
    # Add horizontal lines at significance thresholds
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    for i in range(1, len(pvalue_matrix.index)):
        ax1.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('-log₁₀(p)', rotation=270, labelpad=15)
    
    # Significance after correction
    im2 = ax2.imshow(significance_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_xticks(range(len(significance_matrix.columns)))
    ax2.set_xticklabels(significance_matrix.columns, rotation=45, ha='right')
    ax2.set_yticks(range(len(significance_matrix.index)))
    ax2.set_yticklabels(significance_matrix.index)
    ax2.set_title(f'{title}\nSignificant (corrected)')
    ax2.set_xlabel('Asset (Target)')
    ax2.set_ylabel('Index (Predictor)')
    
    # Add grid
    for i in range(1, len(significance_matrix.index)):
        ax2.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    for j in range(1, len(significance_matrix.columns)):
        ax2.axvline(x=j-0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar2.set_ticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    plt.show()
