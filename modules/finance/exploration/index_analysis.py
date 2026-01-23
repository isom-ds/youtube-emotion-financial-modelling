"""
Index analysis utilities for comparing emotion index variants.

Provides standardized analysis functions for:
- Comparing different weighting schemes
- Event window analysis
- Granger causality testing
- Asset correlation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from .exploratory_stats import describe_block, rolling_mean_std_plot, corr_heatmap
from .statistical_tests import (
    granger_causality_matrix, 
    plot_granger_matrix,
    bidirectional_granger_test,
    johansen_cointegration_test
)


def find_index_variants(columns: pd.Index, base_name: str) -> List[str]:
    """
    Find all variants of an index (base, engweighted, topicweighted, engweighted_topicweighted).
    
    Args:
        columns: DataFrame columns
        base_name: Base index name (e.g., 'epi', 'ccd', 'surprise')
    
    Returns:
        List of column names matching the base with all weighting variants
    
    Example:
        >>> find_index_variants(df.columns, 'epi')
        ['epi', 'epi_engweighted', 'epi_topicweighted', 'epi_engweighted_topicweighted']
    """
    variants = []
    
    # Check for base
    if base_name in columns:
        variants.append(base_name)
    
    # Check for weighted variants
    for suffix in ['_engweighted', '_topicweighted', '_engweighted_topicweighted']:
        variant = f"{base_name}{suffix}"
        if variant in columns:
            variants.append(variant)
    
    return variants


def find_all_index_families(columns: pd.Index) -> Dict[str, List[str]]:
    """
    Automatically detect all index families and their variants.
    
    Args:
        columns: DataFrame columns
    
    Returns:
        Dict mapping family name to list of variants
    
    Example:
        >>> families = find_all_index_families(df.columns)
        >>> families['epi']
        ['epi', 'epi_engweighted', 'epi_topicweighted', 'epi_engweighted_topicweighted']
    """
    families = {}
    
    # Base indices to look for
    base_indices = [
        'ei_creator', 'ei_community', 'epi', 'epi_signed',
        'ccd', 'intensity', 'surprise', 'split'
    ]
    
    for base in base_indices:
        variants = find_index_variants(columns, base)
        if variants:
            families[base] = variants
    
    return families


def compare_index_variants_descriptive(
    df: pd.DataFrame,
    variants: List[str],
    title: str = "Index Variants Comparison"
) -> pd.DataFrame:
    """
    Generate descriptive statistics comparing index variants.
    
    Args:
        df: DataFrame with indices
        variants: List of index column names to compare
        title: Title for output
    
    Returns:
        DataFrame with descriptive statistics
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    available_variants = [v for v in variants if v in df.columns]
    
    if not available_variants:
        print(f"Warning: No variants found in DataFrame")
        return pd.DataFrame()
    
    stats = describe_block(df, available_variants)
    display(stats)
    
    # Correlation between variants
    if len(available_variants) > 1:
        corr = df[available_variants].corr()
        print(f"\nCorrelation between variants:")
        display(corr)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.xticks(range(len(available_variants)), available_variants, rotation=45, ha='right')
        plt.yticks(range(len(available_variants)), available_variants)
        plt.title(f"{title}\nCorrelation Matrix")
        
        # Add correlation values as text
        for i in range(len(available_variants)):
            for j in range(len(available_variants)):
                text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(label='Correlation')
        plt.tight_layout()
        plt.show()
    
    return stats


def compare_index_variants_timeseries(
    df: pd.DataFrame,
    variants: List[str],
    title: str = "Index Variants Over Time",
    rolling_window: int = 7
):
    """
    Plot all variants on same chart for visual comparison.
    
    Args:
        df: DataFrame with indices
        variants: List of index column names to plot
        title: Plot title
        rolling_window: Window for rolling mean smoothing
    """
    available_variants = [v for v in variants if v in df.columns]
    
    if not available_variants:
        print(f"Warning: No variants found in DataFrame")
        return
    
    plt.figure(figsize=(18, 6))
    for var in available_variants:
        plt.plot(df.index, df[var].rolling(rolling_window).mean(), label=var, alpha=0.8, linewidth=2)
    
    plt.axhline(0, linestyle='--', linewidth=1, color='gray', alpha=0.5)
    plt.title(f"{title} (rolling mean, w={rolling_window})")
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_variants_vs_assets(
    df: pd.DataFrame,
    variants: List[str],
    asset_cols: List[str],
    title_prefix: str = "Index Variants",
    rolling_window: int = 14
):
    """
    Compare how different variants correlate with assets.
    
    Args:
        df: DataFrame with indices and assets
        variants: List of index column names
        asset_cols: List of asset return column names
        title_prefix: Prefix for plot titles
        rolling_window: Window for rolling correlation
    """
    available_variants = [v for v in variants if v in df.columns]
    available_assets = [a for a in asset_cols if a in df.columns]
    
    if not available_variants or not available_assets:
        print(f"Warning: Insufficient data for comparison")
        return
    
    for asset in available_assets:
        plt.figure(figsize=(18, 5))
        for var in available_variants:
            corr = df[var].rolling(rolling_window).corr(df[asset])
            plt.plot(df.index, corr, label=var, alpha=0.8, linewidth=2)
        
        plt.axhline(0, linestyle='--', linewidth=1, color='gray', alpha=0.5)
        plt.title(f"{title_prefix} vs {asset} — Rolling Correlation (w={rolling_window})")
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def event_window_comparison(
    df: pd.DataFrame,
    variants: List[str],
    event_windows: List[Tuple[str, str, str]],
    pre_days: int = 7,
    post_days: int = 7
) -> pd.DataFrame:
    """
    Compare index variants during event windows.
    
    Args:
        df: DataFrame with indices
        variants: List of index columns to compare
        event_windows: List of (name, start_date, end_date) tuples
        pre_days: Days before event
        post_days: Days after event
    
    Returns:
        DataFrame with mean values for each variant in each phase
    """
    available_variants = [v for v in variants if v in df.columns]
    
    if not available_variants:
        print(f"Warning: No variants found in DataFrame")
        return pd.DataFrame()
    
    rows = []
    
    for name, start_s, end_s in event_windows:
        start, end = pd.to_datetime(start_s), pd.to_datetime(end_s)
        
        pre = df.loc[(df.index >= start - pd.Timedelta(days=pre_days)) & (df.index < start), available_variants]
        dur = df.loc[(df.index >= start) & (df.index <= end), available_variants]
        post = df.loc[(df.index > end) & (df.index <= end + pd.Timedelta(days=post_days)), available_variants]
        
        for phase, block in [("pre", pre), ("during", dur), ("post", post)]:
            row = {"event": name, "phase": phase, "n": len(block)}
            for v in available_variants:
                if v in block.columns:
                    row[f"{v}_mean"] = block[v].mean()
                    row[f"{v}_std"] = block[v].std()
            rows.append(row)
        
        # Deltas
        pre_m, dur_m, post_m = pre.mean(), dur.mean(), post.mean()
        
        row = {"event": name, "phase": "Δ during-pre", "n": np.nan}
        for v in available_variants:
            if v in dur_m.index:
                row[f"{v}_mean"] = dur_m[v] - pre_m[v]
        rows.append(row)
        
        row = {"event": name, "phase": "Δ post-during", "n": np.nan}
        for v in available_variants:
            if v in post_m.index:
                row[f"{v}_mean"] = post_m[v] - dur_m[v]
        rows.append(row)
    
    return pd.DataFrame(rows)


def granger_comparison_variants(
    df: pd.DataFrame,
    variants: List[str],
    asset_cols: List[str],
    maxlag: int = 10,
    ic: str = 'aic',
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Granger causality tests for all variants and compare results.
    
    Args:
        df: DataFrame with indices and assets
        variants: List of index variants to test
        asset_cols: List of asset return columns
        maxlag: Maximum lag for Granger test
        ic: 'aic' or 'bic' for lag selection
        alpha: Significance level
        correction_method: Multiple testing correction method
    
    Returns:
        Tuple of (pvalue_matrix, pvalue_corrected_matrix, significance_matrix)
    """
    available_variants = [v for v in variants if v in df.columns]
    available_assets = [a for a in asset_cols if a in df.columns]
    
    if not available_variants or not available_assets:
        print(f"Warning: Insufficient data for Granger tests")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"\n{'='*70}")
    print(f"GRANGER CAUSALITY TESTS: {available_variants[0].split('_')[0].upper()} Variants")
    print(f"{'='*70}")
    
    pval_mat, pval_corrected, sig_mat = granger_causality_matrix(
        df, available_variants, available_assets, maxlag=maxlag, ic=ic,
        alpha=alpha, correction_method=correction_method
    )
    
    print(f"\nP-values (at optimal lag):")
    display(pval_mat)
    
    print(f"\nCorrected p-values ({correction_method}):")
    display(pval_corrected)
    
    print(f"\nSignificant relationships (α={alpha}, corrected):")
    display(sig_mat)
    
    # Visualization
    plot_granger_matrix(pval_corrected, sig_mat, 
                       title=f"Granger: {available_variants[0].split('_')[0].upper()} → Assets")
    
    return pval_mat, pval_corrected, sig_mat


def summarize_variant_performance(
    pvalue_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    variants: List[str]
) -> pd.DataFrame:
    """
    Summarize which variants perform best in Granger tests.
    
    Args:
        pvalue_matrix: P-values from Granger tests
        significance_matrix: Significance indicators
        variants: List of variant names
    
    Returns:
        DataFrame ranking variants by performance
    """
    summary = []
    
    for var in variants:
        if var in significance_matrix.index:
            n_significant = significance_matrix.loc[var].sum()
            mean_pvalue = pvalue_matrix.loc[var].mean()
            min_pvalue = pvalue_matrix.loc[var].min()
            
            summary.append({
                'variant': var,
                'n_significant': n_significant,
                'mean_pvalue': mean_pvalue,
                'min_pvalue': min_pvalue
            })
    
    summary_df = pd.DataFrame(summary).sort_values('n_significant', ascending=False)
    
    print(f"\n{'='*70}")
    print("VARIANT PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    display(summary_df)
    
    return summary_df
