"""
Overfitting diagnostic visualization for walk-forward validation results.

Supports two formats:
1. LSTM (fold-based): results[model][asset]['fold_results'] with train_rmse/test_rmse per fold
2. XGBoost (direct): results[model][asset] with 'train_rmse' and 'test_rmse'/'rmse' directly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union

from .utils import DEFAULT_ASSETS, normalize_best_models


# Overfitting ratio thresholds
OVERFIT_THRESHOLDS = {
    'OVERFITTING': 2.0,      # ratio > 2.0
    'SLIGHT_OVERFIT': 1.5,   # 1.5 < ratio <= 2.0
    'GOOD': 0.8,             # 0.8 < ratio <= 1.5
    # ratio <= 0.8 -> UNDERFITTING
}


def diagnose_overfit_ratio(ratio: float) -> str:
    """
    Diagnose overfitting based on test/train RMSE ratio.
    
    Args:
        ratio: Test RMSE / Train RMSE
    
    Returns:
        Diagnosis string: 'OVERFITTING', 'SLIGHT OVERFIT', 'GOOD', or 'UNDERFITTING'
    """
    if ratio > OVERFIT_THRESHOLDS['OVERFITTING']:
        return 'OVERFITTING'
    elif ratio > OVERFIT_THRESHOLDS['SLIGHT_OVERFIT']:
        return 'SLIGHT OVERFIT'
    elif ratio > OVERFIT_THRESHOLDS['GOOD']:
        return 'GOOD'
    else:
        return 'UNDERFITTING'


def _extract_overfit_metrics(result: Dict) -> Optional[Dict]:
    """
    Extract train/test RMSE from a result dict.
    
    Supports multiple formats:
    1. Direct keys: result['train_rmse'], result['test_rmse'] or result['rmse']
    2. Fold-based: result['fold_results'] or result['fold_metrics'] (list of dicts)
    3. Pre-computed: result['overfit_ratio'] and result['diagnosis']
    
    Returns:
        Dict with 'train_rmse', 'test_rmse', 'ratio', 'diagnosis', 'source' or None
    """
    # Format 1: Direct keys (XGBoost style)
    if 'train_rmse' in result:
        train_rmse = result['train_rmse']
        test_rmse = result.get('test_rmse') or result.get('rmse')
        
        if train_rmse is not None and test_rmse is not None:
            # Use pre-computed ratio/diagnosis if available
            if 'overfit_ratio' in result:
                ratio = result['overfit_ratio']
                diagnosis = result.get('diagnosis', diagnose_overfit_ratio(ratio))
            else:
                ratio = test_rmse / train_rmse if train_rmse > 0 else np.inf
                diagnosis = diagnose_overfit_ratio(ratio)
            
            return {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'ratio': ratio,
                'diagnosis': diagnosis,
                'source': 'direct',
            }
    
    # Format 2: Fold-based (LSTM style)
    fold_data = result.get('fold_results') or result.get('fold_metrics')
    
    if fold_data and isinstance(fold_data, list):
        train_rmses = [f.get('train_rmse') for f in fold_data if f.get('train_rmse') is not None]
        test_rmses = [f.get('test_rmse') or f.get('rmse') for f in fold_data 
                     if (f.get('test_rmse') or f.get('rmse')) is not None]
        
        if train_rmses and test_rmses:
            avg_train = np.mean(train_rmses)
            avg_test = np.mean(test_rmses)
            ratio = avg_test / avg_train if avg_train > 0 else np.inf
            
            return {
                'train_rmse': avg_train,
                'test_rmse': avg_test,
                'ratio': ratio,
                'diagnosis': diagnose_overfit_ratio(ratio),
                'source': 'fold_data',
                'n_folds': len(train_rmses),
            }
    
    return None


def plot_overfitting_diagnostic(
    results: Dict[str, Dict],
    best_models: Union[str, Dict[str, str]],
    assets: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[Optional[Figure], pd.DataFrame]:
    """
    Visualize overfitting diagnostic by comparing train vs test RMSE.
    
    Supports multiple result formats:
    1. LSTM (fold-based): results[model][asset]['fold_results'] with train_rmse/test_rmse per fold
    2. XGBoost (direct): results[model][asset] with 'train_rmse' and 'test_rmse'/'rmse' directly
    3. Pre-computed: results[model][asset] with 'overfit_ratio' and 'diagnosis'
    
    Args:
        results: Dict with structure results[model_type][asset]
        best_models: Dict mapping assets to model types, or string for all assets
        assets: List of asset names. Defaults to DEFAULT_ASSETS.
        figsize: Figure size (width, height)
    
    Returns:
        Tuple of (matplotlib Figure or None if no data, summary DataFrame)
        
    Note:
        Ratio interpretation:
        - > 2.0: OVERFITTING (model memorizes training data)
        - 1.5-2.0: SLIGHT OVERFIT
        - 0.8-1.5: GOOD generalization
        - < 0.8: UNDERFITTING
    """
    if assets is None:
        assets = DEFAULT_ASSETS
    
    # Normalize best_models to dict format
    best_models_dict = normalize_best_models(best_models, assets)
    
    summary_rows = []
    has_overfit_data = False
    
    # First pass: collect data from all supported formats
    for asset in assets:
        model_type = best_models_dict.get(asset)
        if model_type is None:
            continue
        
        result = results.get(model_type, {}).get(asset, {})
        
        if 'error' in result:
            continue
        
        # Try to extract overfitting metrics from result
        metrics = _extract_overfit_metrics(result)
        
        if metrics is not None:
            has_overfit_data = True
            row = {
                'Asset': asset.upper(),
                'Model': model_type,
                'Train_RMSE': metrics['train_rmse'],
                'Test_RMSE': metrics['test_rmse'],
                'Ratio': metrics['ratio'],
                'Diagnosis': metrics['diagnosis'],
            }
            if 'n_folds' in metrics:
                row['N_Folds'] = metrics['n_folds']
            summary_rows.append(row)
    
    # If no overfitting data available, try alternative diagnostic
    if not has_overfit_data:
        print("⚠️ Train RMSE not available in results. Using alternative diagnostic...")
        print("\nAlternative: Comparing baseline vs model performance")
        print("Large RMSE improvement + poor directional accuracy = likely overfitting to noise\n")
        
        for asset in assets:
            model_type = best_models_dict.get(asset)
            if model_type is None or model_type == 'baseline':
                continue
            
            # Get baseline result
            base_result = results.get('baseline', {}).get(asset, {})
            model_result = results.get(model_type, {}).get(asset, {})
            
            if 'rmse' not in base_result or 'rmse' not in model_result:
                continue
            
            base_rmse = base_result['rmse']
            model_rmse = model_result['rmse']
            dir_acc = model_result.get('dir_acc', 50.0)
            
            improvement = (base_rmse - model_rmse) / base_rmse * 100 if base_rmse > 0 else 0
            
            # Flag: Large RMSE improvement but poor directional accuracy
            if improvement > 20 and dir_acc < 52:
                diagnosis = '⚠️ SUSPICIOUS'
            elif improvement > 15 and dir_acc < 50:
                diagnosis = '⚠️ QUESTIONABLE'
            else:
                diagnosis = '✓ OK'
            
            summary_rows.append({
                'Asset': asset.upper(),
                'Model': model_type,
                'Baseline_RMSE': base_rmse,
                'Model_RMSE': model_rmse,
                'RMSE_Improvement_%': improvement,
                'Dir_Acc_%': dir_acc,
                'Diagnosis': diagnosis,
            })
        
        summary_df = pd.DataFrame(summary_rows)
        return None, summary_df
    
    # Create visualization if we have fold data
    fig, axes = plt.subplots(1, len(summary_rows), figsize=figsize)
    if len(summary_rows) == 1:
        axes = [axes]
    
    colors = {
        'OVERFITTING': '#e74c3c',      # Red
        'SLIGHT OVERFIT': '#f39c12',   # Orange
        'GOOD': '#27ae60',             # Green
        'UNDERFITTING': '#3498db',     # Blue
    }
    
    for idx, row in enumerate(summary_rows):
        ax = axes[idx]
        
        x = ['Train', 'Test']
        y = [row['Train_RMSE'], row['Test_RMSE']]
        color = colors.get(row['Diagnosis'], '#95a5a6')
        
        bars = ax.bar(x, y, color=[color, color], alpha=0.7, edgecolor='black')
        
        # Add ratio annotation
        ax.annotate(f"Ratio: {row['Ratio']:.2f}\n{row['Diagnosis']}", 
                   xy=(0.5, max(y) * 0.9), 
                   ha='center', fontsize=10, fontweight='bold',
                   color=color)
        
        ax.set_title(f"{row['Asset']} ({row['Model']})")
        ax.set_ylabel('RMSE')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                   f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Overfitting Diagnostic: Train vs Test RMSE', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    summary_df = pd.DataFrame(summary_rows)
    
    return fig, summary_df
