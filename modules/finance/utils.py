"""
Shared utility functions for finance visualization module.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Union

# Default assets used across financial notebooks
DEFAULT_ASSETS = ["r_btc", "r_gold", "r_spx"]


def normalize_best_models(
    best_models: Union[str, Dict[str, str]], 
    assets: List[str] = None
) -> Dict[str, str]:
    """
    Normalize best_models to a consistent dict format.
    
    Handles two input formats:
    - XGBoost: single string (e.g., "epi_l0-3") applied to all assets
    - LSTM: dict mapping assets to model types (e.g., {"r_btc": "epi_weighted_l0-3", ...})
    
    Args:
        best_models: Either a string (same model for all assets) or dict (per-asset models)
        assets: List of asset names. Defaults to DEFAULT_ASSETS if None.
    
    Returns:
        Dict mapping each asset to its best model type
    """
    if assets is None:
        assets = DEFAULT_ASSETS
    
    if isinstance(best_models, str):
        # XGBoost style: single model type for all assets
        return {asset: best_models for asset in assets}
    elif isinstance(best_models, dict):
        # LSTM style: already a dict per asset
        return best_models
    else:
        raise ValueError(f"best_models must be str or dict, got {type(best_models)}")


def compute_metrics(actuals: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """
    Compute prediction metrics for volatility models: RMSE and MAE.
    
    Args:
        actuals: Array of actual volatility values (absolute returns)
        preds: Array of predicted volatility values
    
    Returns:
        Dict with 'rmse' and 'mae'
        Note: Directional accuracy removed as it's not applicable for volatility prediction
    """
    actuals = np.asarray(actuals)
    preds = np.asarray(preds)
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    
    return {"rmse": rmse, "mae": mae}
