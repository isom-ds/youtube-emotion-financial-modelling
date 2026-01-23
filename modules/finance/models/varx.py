import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_squared_error

def make_varx_design(Y: pd.DataFrame, X: pd.DataFrame, p: int, q: int):
    """
    Y: (T, k) endogenous returns (SPX, BTC, Gold) on trading-day index
    X: (T, m) exogenous features aligned to Y index (socials+macro+fill flags)
    Build Z = [Y_{t-1..t-p}, X_{t..t-q}] and target Y_t
    """
    Z_parts = []

    # lagged Y
    for i in range(1, p+1):
        Yi = Y.shift(i)
        Yi.columns = [f"{c}_lag{i}" for c in Y.columns]
        Z_parts.append(Yi)

    # current + lagged X
    for j in range(0, q+1):
        Xj = X.shift(j)
        Xj.columns = [f"{c}_xlag{j}" for c in X.columns]
        Z_parts.append(Xj)

    Z = pd.concat(Z_parts, axis=1)
    data = pd.concat([Z, Y], axis=1).dropna()
    Z = data[Z.columns]
    Yt = data[Y.columns]
    return Z, Yt

def rolling_splits(T, initial_train, step, val_size):
    splits = []
    train_end = initial_train
    while train_end + val_size <= T:
        tr = np.arange(0, train_end)
        va = np.arange(train_end, train_end + val_size)
        splits.append((tr, va))
        train_end += step
    return splits

def cv_score_varx(Y, X, p, q, model_kind, params,
                 initial_train=40, step=5, val_size=5):

    Z, Yt = make_varx_design(Y, X, p, q)
    # map rolling splits on original Y index to the reduced index after dropna
    idx_map = {d:i for i, d in enumerate(Z.index)}

    splits_raw = rolling_splits(len(Y.index), initial_train, step, val_size)
    splits = []
    for tr, va in splits_raw:
        tr_dates = Y.index[tr]
        va_dates = Y.index[va]
        tr2 = [idx_map[d] for d in tr_dates if d in idx_map]
        va2 = [idx_map[d] for d in va_dates if d in idx_map]
        if len(tr2) > 20 and len(va2) > 0:
            splits.append((np.array(tr2), np.array(va2)))

    if model_kind == "ridge":
        reg = Ridge(**params)
    elif model_kind == "mt_enet":
        reg = MultiTaskElasticNet(**params, max_iter=20000)
    else:
        raise ValueError("model_kind must be 'ridge' or 'mt_enet'")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", reg)
    ])

    rmses = []
    for tr2, va2 in splits:
        model.fit(Z.iloc[tr2], Yt.iloc[tr2])
        pred = model.predict(Z.iloc[va2])
        rmse = np.sqrt(mean_squared_error(Yt.iloc[va2], pred))
        rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else np.inf

def automl_regularized_varx(Y, X,
                            p_grid=(1,2,3,5,7,10,15),
                            q_grid=(0,1,2,3,5,7,10),
                            ridge_alphas=np.logspace(-4, 4, 17),
                            enet_alphas=np.logspace(-4, 2, 13),
                            enet_l1s=(0.1,0.3,0.5,0.7,0.9)):

    best = {"score": np.inf}
    for p in p_grid:
        for q in q_grid:
            # Ridge
            for a in ridge_alphas:
                s = cv_score_varx(Y, X, p, q, "ridge", {"alpha": a})
                if s < best["score"]:
                    best = {"score": s, "p": p, "q": q, "model": "ridge", "params": {"alpha": float(a)}}

            # MultiTask ElasticNet (joint sparsity across outputs)
            for a in enet_alphas:
                for l1 in enet_l1s:
                    s = cv_score_varx(Y, X, p, q, "mt_enet", {"alpha": a, "l1_ratio": l1})
                    if s < best["score"]:
                        best = {"score": s, "p": p, "q": q, "model": "mt_enet",
                                "params": {"alpha": float(a), "l1_ratio": float(l1)}}

    return best
