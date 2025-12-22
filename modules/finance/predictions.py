"""
Predictions vs Actuals visualization for financial models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple

from .utils import DEFAULT_ASSETS, normalize_best_models, compute_metrics


def plot_predictions_vs_actuals(
    results: Dict[str, Dict],
    best_models: Dict[str, str],
    assets: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    title_prefix: str = "Model",
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot predictions vs actuals for each asset using the best model.
    
    Extracts predictions/actuals from results dict structure:
    - results[model_type][asset]['predictions'] / ['actuals']
    - Uses metrics from results if available, recalculates as fallback
    
    Args:
        results: Dict with structure results[model_type][asset] containing
                 'predictions', 'actuals', and optionally 'rmse', 'dir_acc'
        best_models: Dict mapping assets to their best model type,
                     or string (same model for all assets)
        assets: List of asset names to plot. Defaults to DEFAULT_ASSETS.
        figsize: Figure size (width, height)
        title_prefix: Prefix for subplot titles (e.g., "LSTM", "XGBoost")
    
    Returns:
        Tuple of (matplotlib Figure, summary DataFrame with metrics)
    """
    if assets is None:
        assets = DEFAULT_ASSETS
    
    # Normalize best_models to dict format
    best_models_dict = normalize_best_models(best_models, assets)
    
    fig, axes = plt.subplots(len(assets), 1, figsize=figsize)
    if len(assets) == 1:
        axes = [axes]
    
    summary_rows = []
    
    for idx, asset in enumerate(assets):
        ax = axes[idx]
        model_type = best_models_dict.get(asset)
        
        if model_type is None:
            ax.set_title(f"{asset.upper()} - No model specified")
            ax.text(0.5, 0.5, "No model specified", ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Get result for this asset and model
        result = results.get(model_type, {}).get(asset, {})
        
        if 'error' in result:
            ax.set_title(f"{asset.upper()} - Error: {result['error']}")
            ax.text(0.5, 0.5, result['error'], ha='center', va='center', transform=ax.transAxes)
            continue
        
        if 'predictions' not in result or 'actuals' not in result:
            ax.set_title(f"{asset.upper()} - No predictions available")
            ax.text(0.5, 0.5, "No predictions available", ha='center', va='center', transform=ax.transAxes)
            continue
        
        preds = result['predictions']
        actuals = result['actuals']
        
        # Get metrics from results if available, otherwise compute
        if 'rmse' in result and 'dir_acc' in result:
            rmse = result['rmse']
            dir_acc = result['dir_acc']
        else:
            metrics = compute_metrics(np.asarray(actuals), np.asarray(preds))
            rmse = metrics['rmse']
            dir_acc = metrics['dir_acc']
        
        # Plot
        actuals.plot(ax=ax, label='Actual', alpha=0.7, linewidth=1)
        preds.plot(ax=ax, label='Predicted', alpha=0.7, linewidth=1)
        
        ax.set_title(f"{asset.upper()} - {title_prefix} ({model_type}) | RMSE: {rmse:.5f}, Dir Acc: {dir_acc:.1f}%")
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Summary row
        summary_rows.append({
            'Asset': asset.upper(),
            'Model': model_type,
            'RMSE': rmse,
            'Dir_Acc_%': dir_acc,
            'N_Predictions': len(preds),
        })
    
    plt.tight_layout()
    
    summary_df = pd.DataFrame(summary_rows)
    
    return fig, summary_df
