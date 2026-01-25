"""
Trading performance metrics and volatility clustering diagnostics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, Tuple, Optional


def compute_trading_metrics(
    strategy_returns: np.ndarray,
    buy_hold_returns: np.ndarray,
    annual_factor: int = 252,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute comprehensive trading performance metrics.
    
    Parameters:
    -----------
    strategy_returns : np.ndarray
        Daily strategy returns
    buy_hold_returns : np.ndarray
        Daily buy-and-hold returns
    annual_factor : int
        Number of trading days per year (default: 252)
    risk_free_rate : float
        Annualized risk-free rate (default: 0.0)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - sharpe_ratio: Annualized Sharpe ratio
        - sortino_ratio: Annualized Sortino ratio (downside risk)
        - max_drawdown: Maximum peak-to-trough drawdown
        - calmar_ratio: Annualized return / max drawdown
        - win_rate: Percentage of positive return periods
        - annual_return: Annualized return
        - annual_volatility: Annualized volatility
        - information_ratio: IR vs buy-and-hold
    """
    # Basic statistics
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns, ddof=1)
    
    # Annualized metrics
    annual_return = mean_return * annual_factor
    annual_volatility = std_return * np.sqrt(annual_factor)
    
    # Sharpe ratio
    excess_return = mean_return - (risk_free_rate / annual_factor)
    sharpe_ratio = (excess_return * annual_factor) / (std_return * np.sqrt(annual_factor)) if std_return > 0 else 0.0
    
    # Sortino ratio (downside deviation)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else std_return
    sortino_ratio = (excess_return * annual_factor) / (downside_std * np.sqrt(annual_factor)) if downside_std > 0 else 0.0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    # Win rate
    win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
    
    # Profit factor (sum of gains / sum of losses)
    gains = np.sum(strategy_returns[strategy_returns > 0])
    losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gains / losses if losses > 0 else 0.0
    
    # Information ratio vs buy-and-hold
    active_returns = strategy_returns - buy_hold_returns
    tracking_error = np.std(active_returns, ddof=1)
    information_ratio = (np.mean(active_returns) * annual_factor) / (tracking_error * np.sqrt(annual_factor)) if tracking_error > 0 else 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'information_ratio': information_ratio,
    }


def compute_volatility_clustering(
    returns: np.ndarray,
    max_lags: int = 20,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute volatility clustering diagnostics using ACF of squared returns.
    
    Parameters:
    -----------
    returns : np.ndarray
        Return series (can be actual returns or prediction errors)
    max_lags : int
        Maximum number of lags for ACF (default: 20)
    alpha : float
        Significance level for Ljung-Box test (default: 0.05)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - acf_squared: Array of ACF values for squared returns
        - ljung_box_stat: Ljung-Box Q-statistic
        - ljung_box_pvalue: p-value for Ljung-Box test
        - persistence: Sum of significant ACF coefficients
        - half_life: Estimated half-life of ACF decay
    """
    # Square the returns
    squared_returns = returns ** 2
    
    # Compute ACF
    from statsmodels.tsa.stattools import acf
    acf_values = acf(squared_returns, nlags=max_lags, fft=False)
    
    # Ljung-Box test on squared returns
    lb_result = acorr_ljungbox(squared_returns, lags=max_lags, return_df=True)
    lb_stat = lb_result['lb_stat'].iloc[-1]  # Last lag statistic
    lb_pvalue = lb_result['lb_pvalue'].iloc[-1]  # Last lag p-value
    
    # Persistence measure: sum of significant ACF coefficients
    # Significance threshold: 1.96 / sqrt(n)
    n = len(returns)
    threshold = 1.96 / np.sqrt(n)
    significant_acf = acf_values[1:][np.abs(acf_values[1:]) > threshold]
    persistence = np.sum(significant_acf) if len(significant_acf) > 0 else 0.0
    
    # Half-life estimation (lag where ACF drops below 0.5)
    half_life_idx = np.where(acf_values[1:] < 0.5)[0]
    half_life = half_life_idx[0] + 1 if len(half_life_idx) > 0 else max_lags
    
    return {
        'acf_values': acf_values,
        'acf_lag1': acf_values[1] if len(acf_values) > 1 else 0.0,
        'acf_lag5': acf_values[5] if len(acf_values) > 5 else 0.0,
        'acf_lag10': acf_values[10] if len(acf_values) > 10 else 0.0,
        'ljung_box_stat': lb_stat,
        'ljung_box_pvalue': lb_pvalue,
        'persistence': persistence,
        'half_life': half_life,
    }


def compute_overfitting_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    window_size: int = 12,
) -> Dict[str, float]:
    """
    Compute overfitting diagnostics including rolling IC and coefficient stability.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Model predictions
    dates : Optional[pd.DatetimeIndex]
        Date index for rolling calculations
    window_size : int
        Window size for rolling IC (default: 12 months)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - ic: Overall Spearman IC
        - ic_rolling_mean: Mean of rolling IC
        - ic_rolling_std: Std of rolling IC
        - ic_stability: IC mean / IC std (higher is better)
        - ic_rolling_values: Array of rolling IC values
        - ic_rolling_dates: Dates for rolling IC
    """
    # Overall Information Coefficient (Spearman correlation)
    ic, _ = stats.spearmanr(y_pred, y_true)
    
    # Rolling IC
    if dates is not None and len(dates) == len(y_pred):
        df = pd.DataFrame({'pred': y_pred, 'actual': y_true}, index=dates)
        
        # Calculate rolling IC manually
        rolling_ic_list = []
        rolling_dates_list = []
        
        for i in range(window_size, len(df) + 1):
            window_data = df.iloc[i - window_size:i]
            if len(window_data) >= 5:
                ic_val, _ = stats.spearmanr(window_data['pred'], window_data['actual'])
                rolling_ic_list.append(ic_val)
                rolling_dates_list.append(window_data.index[-1])
        
        rolling_ic_values = np.array(rolling_ic_list)
        rolling_ic_dates = pd.DatetimeIndex(rolling_dates_list)
        
        ic_mean = np.mean(rolling_ic_values)
        ic_std = np.std(rolling_ic_values, ddof=1)
        ic_stability = ic_mean / ic_std if ic_std > 0 else 0.0
        
        return {
            'ic': ic,
            'ic_rolling_mean': ic_mean,
            'ic_rolling_std': ic_std,
            'ic_stability': ic_stability,
            'ic_rolling_values': rolling_ic_values,
            'ic_rolling_dates': rolling_ic_dates,
        }
    else:
        return {
            'ic': ic,
            'ic_rolling_mean': ic,
            'ic_rolling_std': 0.0,
            'ic_stability': 0.0,
            'ic_rolling_values': np.array([ic]),
            'ic_rolling_dates': dates if dates is not None else np.array([0]),
        }



def generate_volatility_scaled_positions(
    momentum_signals: np.ndarray,
    predicted_volatility: np.ndarray,
    percentile_cap: float = 95.0,
    percentile_floor: float = 5.0,
    max_position_change: Optional[float] = None
) -> np.ndarray:
    """
    Generate position sizes scaled by inverse volatility with percentile bounds.
    
    Parameters:
    -----------
    momentum_signals : np.ndarray
        Directional signals (-1, 0, or 1)
    predicted_volatility : np.ndarray
        Predicted volatility (absolute return forecasts)
    percentile_cap : float
        Upper percentile for capping inverse volatility (default: 95)
    percentile_floor : float
        Lower percentile for flooring inverse volatility (default: 5)
    max_position_change : float, optional
        Maximum allowed position change between periods (e.g., 0.5)
        
    Returns:
    --------
    np.ndarray
        Scaled position sizes
    """
    # Compute inverse volatility
    inverse_vol = 1.0 / predicted_volatility
    
    # Apply percentile bounds to inverse volatility
    cap_value = np.percentile(inverse_vol, percentile_cap)
    floor_value = np.percentile(inverse_vol, percentile_floor)
    inverse_vol_bounded = np.clip(inverse_vol, floor_value, cap_value)
    
    # Normalize to have mean = 1 for easier interpretation
    inverse_vol_normalized = inverse_vol_bounded / np.mean(inverse_vol_bounded)
    
    # Apply momentum signals
    positions = momentum_signals * inverse_vol_normalized
    
    # Apply position change limit if specified
    if max_position_change is not None:
        for i in range(1, len(positions)):
            position_change = positions[i] - positions[i-1]
            if abs(position_change) > max_position_change:
                positions[i] = positions[i-1] + np.sign(position_change) * max_position_change
    
    return positions


def compute_coefficient_stability(
    coefficients_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute stability metrics for model coefficients across CV folds or bootstrap samples.
    
    Parameters:
    -----------
    coefficients_dict : dict
        Dictionary mapping feature names to arrays of coefficient values
        
    Returns:
    --------
    dict
        Dictionary containing:
        - mean_coef_std: Mean standard deviation across features
        - max_coef_std: Maximum standard deviation
        - stability_score: 1 - (mean_std / mean_abs_coef)
    """
    stds = []
    mean_abs_coefs = []
    
    for feature, coefs in coefficients_dict.items():
        if len(coefs) > 1:
            stds.append(np.std(coefs, ddof=1))
            mean_abs_coefs.append(np.mean(np.abs(coefs)))
    
    mean_coef_std = np.mean(stds) if stds else 0.0
    max_coef_std = np.max(stds) if stds else 0.0
    mean_abs_coef = np.mean(mean_abs_coefs) if mean_abs_coefs else 1.0
    
    stability_score = 1.0 - (mean_coef_std / mean_abs_coef) if mean_abs_coef > 0 else 0.0
    
    return {
        'mean_coef_std': mean_coef_std,
        'max_coef_std': max_coef_std,
        'stability_score': stability_score,
    }
