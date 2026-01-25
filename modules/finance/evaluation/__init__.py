"""
Evaluation module for financial modelling.

Provides metrics computation, statistical tests for model comparison,
and feature importance assessment.
"""

from .metrics import (
    compute_all_metrics,
    diebold_mariano_test,
    clark_west_test,
    bootstrap_coefficient_ci,
    permutation_importance_cv,
    apply_fdr_correction,
    compare_cv_methods,
    get_significance_stars,
)

__all__ = [
    "compute_all_metrics",
    "diebold_mariano_test",
    "clark_west_test",
    "bootstrap_coefficient_ci",
    "permutation_importance_cv",
    "apply_fdr_correction",
    "compare_cv_methods",
    "get_significance_stars",
]
