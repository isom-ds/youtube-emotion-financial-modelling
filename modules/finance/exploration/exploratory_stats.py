import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({"font.size": 14})

# -----------------------------
# Helpers
# -----------------------------
def savefig(path: str, save: bool = False):
    if save:
        plt.savefig(path, dpi=200, bbox_inches="tight")

def savecsv(df_out: pd.DataFrame, path: str, save: bool = False):
    if save:
        df_out.to_csv(path)

def safe_numeric(df_in: pd.DataFrame, cols):
    df_out = df_in.copy()
    for c in cols:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
    return df_out

def describe_block(df_in: pd.DataFrame, cols):
    x = df_in[cols].copy()
    desc = x.describe().T
    desc["skew"] = x.skew(numeric_only=True)
    desc["kurt"] = x.kurtosis(numeric_only=True)
    desc["missing_pct"] = x.isna().mean() * 100
    desc["zero_pct"] = (x.fillna(0) == 0).mean() * 100
    return desc.sort_index()

def rolling_mean_std_plot(df_in: pd.DataFrame, cols, window: int, title: str, outdir:str=None, save: bool = False):
    plt.figure(figsize=(18, 6))
    for c in cols:
        plt.plot(df_in.index, df_in[c].rolling(window).mean(), label=f"{c}")
    plt.title(f"{title} — rolling mean (w={window})")
    plt.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()
    if outdir:
        savefig(os.path.join(outdir, f"{title}_rollmean.png".replace(" ", "_")), save=save)

    plt.figure(figsize=(18, 6))
    for c in cols:
        plt.plot(df_in.index, df_in[c].rolling(window).std(), label=f"{c}")
    plt.title(f"{title} — rolling std (w={window})")
    plt.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()
    if outdir:
        savefig(os.path.join(outdir, f"{title}_rollstd.png".replace(" ", "_")), save=save)

def corr_heatmap(df_in: pd.DataFrame, cols, title: str, outdir:str=None, save: bool = False) -> pd.DataFrame:
    corr = df_in[cols].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    if outdir:
        savefig(os.path.join(outdir, f"{title}_corrheat.png".replace(" ", "_")), save=save)
    return corr

def rolling_corr_plot(df_in: pd.DataFrame, a: str, b: str, window: int, outdir:str=None, save: bool = False):
    s = df_in[a].rolling(window).corr(df_in[b])
    plt.figure(figsize=(18, 5))
    plt.plot(df_in.index, s)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"Rolling Corr(w={window}): {a} vs {b}")
    plt.tight_layout()
    plt.show()
    if outdir:
        savefig(os.path.join(outdir, f"rollcorr_{a}_vs_{b}.png"), save=save)

def acf_plot(series: pd.Series, nlags: int, title: str, outdir:str=None, save: bool = False):
    vals = acf(series.dropna(), nlags=nlags, fft=True)
    plt.figure(figsize=(12, 4))
    plt.stem(range(len(vals)), vals)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    if outdir:
        savefig(os.path.join(outdir, f"{title}_acf.png".replace(" ", "_")), save=save)

def lead_lag_corr(df_in: pd.DataFrame, x: str, y: str, max_lag: int) -> pd.DataFrame:
    out = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = df_in[x].corr(df_in[y].shift(-lag))
        else:
            corr = df_in[x].shift(lag).corr(df_in[y])
        out.append({"lag": lag, "corr": corr})
    return pd.DataFrame(out).set_index("lag")

def lead_lag_plot(ll: pd.DataFrame, title: str, outdir:str=None, save: bool = False):
    plt.figure(figsize=(12, 4))
    plt.plot(ll.index, ll["corr"], marker="o")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(title + " (x leads when lag>0)")
    plt.tight_layout()
    plt.show()
    if outdir:
        savefig(os.path.join(outdir, f"{title}_leadlag.png".replace(" ", "_")), save=save)

def find_topic_cols(cols, prefix: str):
    # Keep: prefix_<topic> columns. Exclude base + weighted.
    out = []
    for c in cols:
        if c.startswith(prefix + "_") and c not in [prefix, f"{prefix}_weighted"]:
            out.append(c)
    out = [c for c in out if c not in [f"{prefix}_weighted", prefix]]
    return out

def topic_share_entropy(df_in: pd.DataFrame, topic_cols):
    X = df_in[topic_cols].copy()
    absX = X.abs()
    mean_abs = absX.mean().sort_values(ascending=False)
    share = (mean_abs / mean_abs.sum()).rename("share_of_total_mean_abs")
    shares_df = pd.DataFrame({"mean_abs": mean_abs, "share": share})

    row_sum = absX.sum(axis=1).replace(0, np.nan)
    P = absX.div(row_sum, axis=0)
    K = len(topic_cols)
    eps = 1e-12
    H = -(P * np.log(P + eps)).sum(axis=1)
    H_norm = H / np.log(K) if K > 1 else np.nan

    entropy_df = pd.DataFrame({
        "entropy": H,
        "entropy_norm": H_norm,
        "concentration_1_minus_entropy_norm": 1 - H_norm
    })
    return shares_df, entropy_df

def creator_community_divergence(df_in: pd.DataFrame, creator_cols, community_cols) -> pd.DataFrame:
    def strip_prefix(c: str, pref: str) -> str:
        return c[len(pref) + 1:]

    c_map = {strip_prefix(c, "ei_creator"): c for c in creator_cols}
    m_map = {strip_prefix(c, "ei_community"): c for c in community_cols}
    common_topics = sorted(set(c_map.keys()).intersection(m_map.keys()))

    out = pd.DataFrame(index=df_in.index)
    for t in common_topics:
        out[f"div_{t}"] = df_in[c_map[t]] - df_in[m_map[t]]
    return out

def run_pca(df_in: pd.DataFrame, cols, n_components: int, title_prefix: str, outdir:str=None, save: bool = False):
    X = df_in[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if X.shape[0] < 5 or X.shape[1] < 2:
        print(f"[PCA skipped] Not enough data for {title_prefix}")
        return None, None, None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    pcs = pca.fit_transform(Xs)

    # Explained variance
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker="o")
    plt.title(f"{title_prefix} — Explained variance ratio")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.tight_layout()
    plt.show()
    if outdir:
        savefig(os.path.join(outdir, f"{title_prefix}_pca_explained_variance.png".replace(" ", "_")), save=save)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )

    pcs_df = pd.DataFrame(
        pcs,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )

    # Plot PC1/PC2 time series
    for i in range(min(2, pca.n_components_)):
        plt.figure(figsize=(18, 5))
        plt.plot(pcs_df.index, pcs_df[f"PC{i+1}"])
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title(f"{title_prefix} — PC{i+1} over time")
        plt.tight_layout()
        plt.show()
        if outdir:
            savefig(os.path.join(outdir, f"{title_prefix}_PC{i+1}_timeseries.png".replace(" ", "_")), save=save)

    if outdir:
        savecsv(loadings, os.path.join(outdir, f"{title_prefix}_pca_loadings.csv".replace(" ", "_")))
        savecsv(pcs_df, os.path.join(outdir, f"{title_prefix}_pca_pcs.csv".replace(" ", "_")))
    return pca, loadings, pcs_df