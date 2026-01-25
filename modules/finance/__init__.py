"""
Finance visualization module for financial modeling notebooks.

Provides reusable plotting functions for:
- Predictions vs Actuals comparison
- Cumulative Returns (Strategy vs Buy & Hold)
- Overfitting diagnostics

Compatible with notebooks 513-516 (XGBoost and LSTM models).
"""

from .utils import (
    DEFAULT_ASSETS,
    normalize_best_models,
    compute_metrics,
)

from .predictions import plot_predictions_vs_actuals

from .cumulative_returns import plot_cumulative_returns


from .overfitting import (
    plot_overfitting_diagnostic,
    diagnose_overfit_ratio,
    OVERFIT_THRESHOLDS,
)

# ARX/VAR-specific visualizations
from .arx_var import (
    best_lags_from_df,
    plot_arx_predictions,
    plot_arx_cumulative_returns,
    plot_arx_rolling_rmse,
    plot_var_predictions,
    plot_var_cumulative_returns,
    plot_var_rolling_rmse,
)

__all__ = [
    # Utils
    "DEFAULT_ASSETS",
    "normalize_best_models", 
    "compute_metrics",
    # Visualizations
    "plot_predictions_vs_actuals",
    "plot_cumulative_returns",
    "plot_overfitting_diagnostic",
    "diagnose_overfit_ratio",
    "OVERFIT_THRESHOLDS",
    # ARX/VAR
    "best_lags_from_df",
    "plot_arx_predictions",
    "plot_arx_cumulative_returns",
    "plot_arx_rolling_rmse",
    "plot_var_predictions",
    "plot_var_cumulative_returns",
    "plot_var_rolling_rmse",
]
