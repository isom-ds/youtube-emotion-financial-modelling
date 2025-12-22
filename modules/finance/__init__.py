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
]
