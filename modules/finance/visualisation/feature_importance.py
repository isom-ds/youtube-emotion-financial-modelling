"""
Feature importance visualization module.

Provides matplotlib-based visualizations for:
- Coefficient importance with significance markers
- SHAP summary plots
- Permutation importance
- Model comparison charts
- Significance tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union
import warnings


# Default significance markers
DEFAULT_SIG_MARKERS = {0.01: '***', 0.05: '**', 0.10: '*'}

# Index family groupings for aggregation
INDEX_FAMILY_MAP = {
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


def _get_feature_family(feature_name: str) -> str:
    """Map a feature name to its index family."""
    name_lower = feature_name.lower()
    
    # Remove lag suffix for matching
    base_name = name_lower.split("_lag")[0].replace("soc_", "")
    
    for family, prefixes in INDEX_FAMILY_MAP.items():
        for prefix in prefixes:
            if base_name == prefix or base_name.startswith(prefix + "_"):
                return family
    
    # Check for control/other categories
    if "ctl_" in name_lower or any(x in name_lower for x in ["dln_", "dy10", "eurusd", "usdcny", "oil"]):
        return "controls"
    if "y_lag" in name_lower:
        return "ar_lags"
    if "xas_" in name_lower:
        return "cross_asset"
    
    return "other"


def _get_significance_stars(pvalue: float, markers: Dict[float, str] = None) -> str:
    """Convert p-value to significance stars."""
    if markers is None:
        markers = DEFAULT_SIG_MARKERS
    if pd.isna(pvalue):
        return ''
    for level, stars in sorted(markers.items()):
        if pvalue <= level:
            return stars
    return ''


# =============================================================================
# Coefficient Importance Plot
# =============================================================================

def plot_coefficient_importance(
    coefs: pd.Series,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    pvalues: Optional[pd.Series] = None,
    sig_markers: Dict[float, str] = None,
    top_n: int = 20,
    title: str = "Feature Coefficients",
    figsize: Tuple[int, int] = (10, 8),
    color_positive: str = "#2ecc71",
    color_negative: str = "#e74c3c",
    show_ci: bool = True,
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot coefficient importance with optional confidence intervals and significance markers.
    
    Args:
        coefs: Series of coefficients indexed by feature name
        ci_lower: Lower confidence interval bounds
        ci_upper: Upper confidence interval bounds  
        pvalues: P-values for significance testing
        sig_markers: Dict mapping significance levels to star markers
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size (width, height)
        color_positive: Color for positive coefficients
        color_negative: Color for negative coefficients
        show_ci: Whether to show confidence interval error bars
    
    Returns:
        Tuple of (matplotlib Figure, summary DataFrame)
    """
    if sig_markers is None:
        sig_markers = DEFAULT_SIG_MARKERS
    
    # Sort by absolute value and take top N
    coefs_sorted = coefs.abs().sort_values(ascending=False).head(top_n)
    features = coefs_sorted.index.tolist()
    values = coefs.loc[features].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors based on sign
    colors = [color_positive if v >= 0 else color_negative for v in values]
    
    y_pos = np.arange(len(features))
    
    # Plot bars
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add confidence intervals if provided
    if show_ci and ci_lower is not None and ci_upper is not None:
        ci_l = ci_lower.loc[features].values
        ci_u = ci_upper.loc[features].values
        xerr = np.array([values - ci_l, ci_u - values])
        ax.errorbar(values, y_pos, xerr=xerr, fmt='none', color='black', capsize=3, alpha=0.7)
    
    # Add significance stars
    if pvalues is not None:
        for i, feat in enumerate(features):
            pval = pvalues.get(feat, np.nan)
            stars = _get_significance_stars(pval, sig_markers)
            if stars:
                # Position stars at end of bar
                x_pos = values[i]
                offset = 0.002 * np.sign(x_pos) if x_pos != 0 else 0.002
                ax.annotate(stars, (x_pos + offset, i), va='center', 
                           ha='left' if x_pos >= 0 else 'right', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top to bottom
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Coefficient')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend for significance
    legend_text = "  ".join([f"{stars}: p<{level}" for level, stars in sorted(sig_markers.items())])
    ax.annotate(legend_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                fontsize=9, va='top', ha='left', style='italic')
    
    plt.tight_layout()
    
    # Build summary DataFrame
    summary = pd.DataFrame({
        'coefficient': coefs.loc[features],
        'abs_coefficient': coefs.loc[features].abs(),
    })
    if pvalues is not None:
        summary['pvalue'] = pvalues.loc[features] if hasattr(pvalues, 'loc') else [pvalues.get(f, np.nan) for f in features]
        summary['significance'] = summary['pvalue'].apply(lambda p: _get_significance_stars(p, sig_markers))
    if ci_lower is not None:
        summary['ci_lower'] = ci_lower.loc[features]
    if ci_upper is not None:
        summary['ci_upper'] = ci_upper.loc[features]
    
    return fig, summary


# =============================================================================
# SHAP Summary Plot
# =============================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    plot_type: str = "bar",
    top_n: int = 20,
    title: str = "SHAP Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot SHAP summary (bar or beeswarm style).
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature DataFrame for beeswarm coloring
        feature_names: Feature names (uses X.columns if None)
        plot_type: "bar" or "beeswarm"
        top_n: Number of features to show
        title: Plot title
        figsize: Figure size
    
    Returns:
        Tuple of (matplotlib Figure, importance DataFrame)
    """
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP not installed. Install with: pip install shap")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "SHAP not available", ha='center', va='center', transform=ax.transAxes)
        return fig, pd.DataFrame()
    
    if feature_names is None:
        feature_names = list(X.columns) if hasattr(X, 'columns') else [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Get top features
    top_features = importance_df.head(top_n)['feature'].tolist()
    top_idx = [feature_names.index(f) for f in top_features]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == "bar":
        # Bar plot of mean absolute SHAP values
        y_pos = np.arange(len(top_features))
        values = importance_df.head(top_n)['mean_abs_shap'].values
        
        ax.barh(y_pos, values, color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        
    elif plot_type == "beeswarm":
        # Beeswarm plot
        shap_subset = shap_values[:, top_idx]
        X_subset = X.iloc[:, top_idx] if hasattr(X, 'iloc') else X[:, top_idx]
        
        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=shap_subset,
            base_values=np.zeros(len(shap_subset)),
            data=X_subset.values if hasattr(X_subset, 'values') else X_subset,
            feature_names=top_features
        )
        
        plt.close(fig)  # Close our fig, shap will create its own
        shap.plots.beeswarm(explanation, show=False, max_display=top_n)
        fig = plt.gcf()
        fig.suptitle(title)
    
    plt.tight_layout()
    
    return fig, importance_df


# =============================================================================
# Permutation Importance Plot
# =============================================================================

def plot_permutation_importance(
    importances: pd.DataFrame,
    top_n: int = 20,
    title: str = "Permutation Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    color: str = "#9b59b6",
    show_error: bool = True,
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot permutation importance with error bars.
    
    Args:
        importances: DataFrame with 'importance_mean', 'importance_std', 'pvalue' columns
        top_n: Number of features to show
        title: Plot title
        figsize: Figure size
        color: Bar color
        show_error: Whether to show error bars
    
    Returns:
        Tuple of (matplotlib Figure, filtered DataFrame)
    """
    # Sort and filter
    df = importances.sort_values('importance_mean', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(df))
    values = df['importance_mean'].values
    
    # Plot bars
    ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add error bars
    if show_error and 'importance_std' in df.columns:
        ax.errorbar(values, y_pos, xerr=df['importance_std'].values, 
                    fmt='none', color='black', capsize=3, alpha=0.7)
    
    # Add significance stars
    if 'pvalue' in df.columns or 'stars' in df.columns:
        for i, (idx, row) in enumerate(df.iterrows()):
            stars = row.get('stars', _get_significance_stars(row.get('pvalue', np.nan)))
            if stars:
                ax.annotate(stars, (values[i] + 0.001, i), va='center', ha='left', 
                           fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Importance (decrease in performance when permuted)')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, df


# =============================================================================
# Model Comparison Charts
# =============================================================================

def plot_model_comparison_bars(
    baseline_metrics: Dict[str, Dict[str, float]],
    enhanced_metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = ["mae", "rmse"],
    assets: Optional[List[str]] = None,
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = (12, 5),
    baseline_label: str = "Baseline",
    enhanced_label: str = "Enhanced",
    dm_pvalues: Optional[Dict[str, float]] = None,
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot grouped bar chart comparing baseline vs enhanced model performance.
    
    Args:
        baseline_metrics: Dict[asset] -> Dict[metric_name] -> value
        enhanced_metrics: Dict[asset] -> Dict[metric_name] -> value
        metric_names: List of metrics to plot
        assets: Asset names (defaults to keys from baseline_metrics)
        title: Plot title
        figsize: Figure size
        baseline_label: Label for baseline bars
        enhanced_label: Label for enhanced bars
        dm_pvalues: Optional Diebold-Mariano p-values for significance markers
    
    Returns:
        Tuple of (matplotlib Figure, comparison DataFrame)
    """
    if assets is None:
        assets = list(baseline_metrics.keys())
    
    n_assets = len(assets)
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    comparison_data = []
    
    for ax, metric in zip(axes, metric_names):
        x = np.arange(n_assets)
        width = 0.35
        
        baseline_vals = [baseline_metrics.get(asset, {}).get(metric, np.nan) for asset in assets]
        enhanced_vals = [enhanced_metrics.get(asset, {}).get(metric, np.nan) for asset in assets]
        
        # Plot bars
        bars1 = ax.bar(x - width/2, baseline_vals, width, label=baseline_label, 
                       color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, enhanced_vals, width, label=enhanced_label,
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add significance markers if provided
        if dm_pvalues is not None:
            for i, asset in enumerate(assets):
                pval = dm_pvalues.get(asset, np.nan)
                stars = _get_significance_stars(pval)
                if stars:
                    max_val = max(baseline_vals[i], enhanced_vals[i])
                    ax.annotate(stars, (i, max_val * 1.02), ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(assets)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Collect comparison data
        for asset, bval, eval_ in zip(assets, baseline_vals, enhanced_vals):
            comparison_data.append({
                'asset': asset,
                'metric': metric,
                'baseline': bval,
                'enhanced': eval_,
                'improvement': bval - eval_,
                'improvement_pct': (bval - eval_) / bval * 100 if bval != 0 else np.nan,
            })
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, pd.DataFrame(comparison_data)


# =============================================================================
# Significance Table / Heatmap
# =============================================================================

def plot_significance_table(
    features: List[str],
    pvalues: Union[np.ndarray, pd.Series],
    corrected_pvalues: Optional[Union[np.ndarray, pd.Series]] = None,
    alpha_levels: List[float] = [0.01, 0.05, 0.10],
    title: str = "Feature Significance",
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot significance table as heatmap with raw and corrected p-values.
    
    Args:
        features: Feature names
        pvalues: Raw p-values
        corrected_pvalues: FDR-corrected p-values (optional)
        alpha_levels: Significance levels for coloring
        title: Plot title
        figsize: Figure size
    
    Returns:
        Tuple of (matplotlib Figure, significance DataFrame)
    """
    pvalues = np.asarray(pvalues)
    
    # Build DataFrame
    df = pd.DataFrame({
        'feature': features,
        'pvalue': pvalues,
    })
    
    if corrected_pvalues is not None:
        df['pvalue_corrected'] = np.asarray(corrected_pvalues)
    
    # Add significance columns
    for alpha in alpha_levels:
        df[f'sig_{int((1-alpha)*100)}'] = df['pvalue'] <= alpha
    
    df['stars'] = df['pvalue'].apply(_get_significance_stars)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create significance matrix for heatmap
    sig_cols = [f'sig_{int((1-a)*100)}' for a in alpha_levels]
    sig_matrix = df[sig_cols].values.astype(float)
    
    # Plot heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(sig_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(sig_cols)))
    ax.set_xticklabels([f'α={a}' for a in alpha_levels])
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(len(sig_cols)):
            text = '✓' if sig_matrix[i, j] else ''
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=12)
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Significant')
    
    plt.tight_layout()
    
    return fig, df.set_index('feature')


# =============================================================================
# Family Importance Summary (Grouped)
# =============================================================================

def plot_family_importance_summary(
    coefs: pd.Series,
    pvalues: Optional[pd.Series] = None,
    assets: Optional[List[str]] = None,
    title: str = "Index Family Importance by Asset",
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot aggregated importance by index family, grouped visualization.
    
    Aggregates coefficients within each index family (EI, EPI, etc.)
    and displays as grouped bar chart.
    
    Args:
        coefs: Series of coefficients (can be multi-index with asset)
        pvalues: Optional p-values for significance
        assets: Asset names if not in index
        title: Plot title
        figsize: Figure size
    
    Returns:
        Tuple of (matplotlib Figure, family importance DataFrame)
    """
    # Map features to families
    family_importance = {}
    
    for feature, coef in coefs.items():
        family = _get_feature_family(feature)
        if family not in family_importance:
            family_importance[family] = []
        family_importance[family].append(abs(coef))
    
    # Aggregate (mean absolute coefficient per family)
    family_summary = pd.DataFrame({
        'family': list(family_importance.keys()),
        'mean_abs_coef': [np.mean(v) for v in family_importance.values()],
        'sum_abs_coef': [np.sum(v) for v in family_importance.values()],
        'n_features': [len(v) for v in family_importance.values()],
    }).sort_values('mean_abs_coef', ascending=False)
    
    # Define family colors
    family_colors = {
        'ei': '#e74c3c',
        'epi': '#3498db',
        'epi_signed': '#9b59b6',
        'ccd': '#2ecc71',
        'intensity': '#f39c12',
        'surprise': '#1abc9c',
        'split': '#e91e63',
        'controls': '#95a5a6',
        'ar_lags': '#34495e',
        'cross_asset': '#7f8c8d',
        'other': '#bdc3c7',
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    families = family_summary['family'].tolist()
    values = family_summary['mean_abs_coef'].values
    colors = [family_colors.get(f, '#bdc3c7') for f in families]
    
    y_pos = np.arange(len(families))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.upper().replace('_', ' ') for f in families])
    ax.invert_yaxis()
    ax.set_xlabel('Mean |Coefficient|')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    # Add count annotations
    for i, (_, row) in enumerate(family_summary.iterrows()):
        ax.annotate(f"n={row['n_features']}", (values[i] + 0.001, i), 
                   va='center', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    return fig, family_summary


# =============================================================================
# Cumulative Returns Comparison
# =============================================================================

def plot_cumulative_returns_comparison(
    predictions_baseline: Dict[str, np.ndarray],
    predictions_enhanced: Dict[str, np.ndarray],
    actuals: Dict[str, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    assets: Optional[List[str]] = None,
    title: str = "Cumulative Returns: Baseline vs Enhanced",
    figsize: Tuple[int, int] = (15, 4),
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot cumulative returns comparison across assets.
    
    Shows buy-and-hold, baseline strategy, and enhanced strategy
    cumulative returns for each asset.
    
    Args:
        predictions_baseline: Dict[asset] -> predictions array
        predictions_enhanced: Dict[asset] -> predictions array
        actuals: Dict[asset] -> actual returns array
        dates: Date index for x-axis
        assets: List of assets to plot
        title: Figure title
        figsize: Figure size
    
    Returns:
        Tuple of (matplotlib Figure, performance summary DataFrame)
    """
    if assets is None:
        assets = list(actuals.keys())
    
    n_assets = len(assets)
    fig, axes = plt.subplots(1, n_assets, figsize=figsize)
    if n_assets == 1:
        axes = [axes]
    
    summary_data = []
    
    for ax, asset in zip(axes, assets):
        actual = np.asarray(actuals.get(asset, []))
        pred_base = np.asarray(predictions_baseline.get(asset, []))
        pred_enh = np.asarray(predictions_enhanced.get(asset, []))
        
        # Ensure same length
        n = min(len(actual), len(pred_base), len(pred_enh))
        actual = actual[:n]
        pred_base = pred_base[:n]
        pred_enh = pred_enh[:n]
        
        if n == 0:
            ax.text(0.5, 0.5, f"{asset}: No data", ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Strategy returns: long when predicted > 0, else flat
        strat_base = actual * np.sign(pred_base)
        strat_enh = actual * np.sign(pred_enh)
        
        # Cumulative returns
        cum_bh = (1 + actual).cumprod() - 1
        cum_base = (1 + strat_base).cumprod() - 1
        cum_enh = (1 + strat_enh).cumprod() - 1
        
        x = dates[:n] if dates is not None else np.arange(n)
        
        ax.plot(x, cum_bh * 100, label='Buy & Hold', color='#95a5a6', linewidth=1.5)
        ax.plot(x, cum_base * 100, label='Baseline', color='#e74c3c', linewidth=1.5, linestyle='--')
        ax.plot(x, cum_enh * 100, label='Enhanced', color='#2ecc71', linewidth=2)
        
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title(asset.upper())
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        if dates is not None:
            ax.tick_params(axis='x', rotation=45)
        
        # Summary metrics
        summary_data.append({
            'asset': asset,
            'buy_hold_final': cum_bh[-1] * 100,
            'baseline_final': cum_base[-1] * 100,
            'enhanced_final': cum_enh[-1] * 100,
            'baseline_vs_bh': (cum_base[-1] - cum_bh[-1]) * 100,
            'enhanced_vs_bh': (cum_enh[-1] - cum_bh[-1]) * 100,
            'enhanced_vs_baseline': (cum_enh[-1] - cum_base[-1]) * 100,
        })
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, pd.DataFrame(summary_data)
