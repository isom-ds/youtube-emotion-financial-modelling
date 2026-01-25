"""
Cumulative Returns visualization: Strategy vs Buy-and-Hold.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union

from .utils import DEFAULT_ASSETS, normalize_best_models


def plot_cumulative_returns(
    test_predictions: Dict[str, Dict],
    returns: pd.DataFrame,
    best_models: Union[str, Dict[str, str]],
    assets: Optional[List[str]] = None,
    results: Optional[Dict[str, Dict]] = None,
    test_start: str = "2025-04-01",
    test_end: str = "2025-05-31",
    strategy_label: str = "Strategy",
    figsize: Tuple[int, int] = (15, 4),
) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot cumulative returns comparing model strategy vs buy-and-hold.
    
    Strategy: Go long when predicted return > 0, else flat (0 return).
    Buy & Hold: Always long (use actual returns).
    
    Args:
        test_predictions: Dict with structure test_predictions[asset] containing
                          'predictions' (pd.Series with dates as index)
        returns: DataFrame with raw returns (before feature engineering),
                 must contain asset columns and be indexed by date
        best_models: Dict mapping assets to model types, or string for all assets
        assets: List of asset names. Defaults to DEFAULT_ASSETS.
        results: Optional results dict to extract model info for display
        test_start: Start date of test period (for reference only)
        test_end: End date of test period (for reference only)
        strategy_label: Label for strategy line (e.g., "LSTM Strategy", "XGBoost Strategy")
        figsize: Figure size (width, height)
    
    Returns:
        Tuple of (matplotlib Figure, summary DataFrame with performance metrics)
    """
    if assets is None:
        assets = DEFAULT_ASSETS
    
    # Normalize best_models to dict format
    best_models_dict = normalize_best_models(best_models, assets)
    
    fig, axes = plt.subplots(1, len(assets), figsize=figsize)
    if len(assets) == 1:
        axes = [axes]
    
    summary_rows = []
    
    for idx, asset in enumerate(assets):
        ax = axes[idx]
        model_type = best_models_dict.get(asset, "unknown")
        
        if asset not in test_predictions:
            ax.set_title(f"{asset.upper()} - No predictions available")
            ax.text(0.5, 0.5, "No predictions available", ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Handle both formats:
        # - Dict format: test_predictions[asset] = {'predictions': pd.Series, ...}
        # - Series format: test_predictions[asset] = pd.Series
        pred_data = test_predictions[asset]
        if isinstance(pred_data, dict):
            predictions = pred_data.get('predictions')
        elif isinstance(pred_data, pd.Series):
            predictions = pred_data
        else:
            predictions = None
        
        if predictions is None or len(predictions) == 0:
            ax.set_title(f"{asset.upper()} - Empty predictions")
            ax.text(0.5, 0.5, "Empty predictions", ha='center', va='center', transform=ax.transAxes)
            continue
        
        pred_dates = predictions.index
        
        # Get actual returns from raw returns DataFrame (ensures consistent Buy & Hold)
        if asset not in returns.columns:
            ax.set_title(f"{asset.upper()} - Asset not in returns DataFrame")
            ax.text(0.5, 0.5, f"Asset '{asset}' not found", ha='center', va='center', transform=ax.transAxes)
            continue
        
        actuals = returns.loc[pred_dates, asset]
        
        # Strategy: long if predicted > 0, else flat
        strategy_returns = actuals * np.sign(predictions)
        
        # Cumulative returns
        cum_buy_hold = (1 + actuals).cumprod() - 1
        cum_strategy = (1 + strategy_returns).cumprod() - 1
        
        # Plot
        cum_buy_hold.plot(ax=ax, label='Buy & Hold', linewidth=2)
        cum_strategy.plot(ax=ax, label=strategy_label, linewidth=2)
        
        ax.set_title(f"{asset.upper()} ({model_type})")
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Final returns
        bh_final = cum_buy_hold.iloc[-1] * 100
        strat_final = cum_strategy.iloc[-1] * 100
        
        summary_rows.append({
            'Asset': asset.upper(),
            'Model': model_type,
            'Buy_Hold_%': bh_final,
            'Strategy_%': strat_final,
            'Outperformance_pp': strat_final - bh_final,
            'N_Days': len(pred_dates),
            'Period': f"{pred_dates.min().date()} to {pred_dates.max().date()}",
        })
    
    plt.tight_layout()
    
    summary_df = pd.DataFrame(summary_rows)
    
    return fig, summary_df
