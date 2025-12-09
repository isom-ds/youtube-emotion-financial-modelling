import numpy as np
import pandas as pd

# Define emotion columns once at module level
POS_EMO = ["anticipation", "joy", "trust", "surprise"]
NEG_EMO = ["anger", "disgust", "fear", "sadness"]


def ei(data: pd.DataFrame, type: str) -> pd.DataFrame:
    """Calculate Emotion Index for a DataFrame."""
    df = data.copy()

    # Sum by row for each group
    df["positive"] = df[POS_EMO].sum(axis=1)
    df["negative"] = df[NEG_EMO].sum(axis=1)

    # Calculate EI
    total = df["positive"] + df["negative"]
    df[f"ei_{type}"] = (df["positive"] - df["negative"]) / total.replace(0, np.nan)

    return df


def epi(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame, epsilon=1e-6):
    """Calculate Emotion Polarity Index between creator and community."""
    # Select only needed columns to reduce memory
    df_trans = data_transcripts[["date", "ei_creator"]].copy()
    df_comm = data_comments[["date", "ei_community"]].copy()

    # Convert to datetime
    df_trans["date"] = pd.to_datetime(df_trans["date"])
    df_comm["date"] = pd.to_datetime(df_comm["date"])

    # Merge two tables keep all dates and set missing values to 0
    df = pd.merge(df_trans, df_comm, on="date", how="outer").fillna(0)

    # Compute AEPI with Magnitude Scaling (vectorized)
    ei_creator = df["ei_creator"].values
    ei_community = df["ei_community"].values
    
    abs_creator = np.abs(ei_creator)
    abs_community = np.abs(ei_community)
    
    agreement_score = 1 - 2 * np.abs(
        (ei_creator - ei_community) / (abs_creator + abs_community + epsilon)
    )

    # Magnitude weight based on the size of epi values
    magnitude_weight = (abs_creator + abs_community) * 1.25

    # Final AEPI calculation
    df["epi"] = np.clip(agreement_score * magnitude_weight, -1, 1)
    # df['epi'] = (2 * (df['epi'] - df['epi'].min()) / ((df['epi'].max()) - (df['epi'].min()))) - 1

    # Cases where there the epi_creator or epi_community is 0 set aepi to 0
    # df['epi'] = np.where((df['ei_creator'] == 0) | (df['ei_community'] == 0), 0, df['epi'])

    return df.drop_duplicates(subset=["date"], keep="first")


def ei_topic(
    data: pd.DataFrame,
    type: str,
    topic_list: list
):
    """
    Calculate Emotion Index by topic - MEMORY OPTIMIZED VERSION.
    Uses groupby + pivot instead of multiple merges to avoid memory explosion.
    """
    # Only copy the columns we need
    needed_cols = ["date", "topics_corrected"] + POS_EMO + NEG_EMO
    df = data[needed_cols].copy()

    # Pre-calculate positive and negative sums once for all rows
    df["positive"] = df[POS_EMO].sum(axis=1)
    df["negative"] = df[NEG_EMO].sum(axis=1)
    
    # Calculate EI for all rows at once
    total = df["positive"] + df["negative"]
    df["ei"] = np.where(total > 0, (df["positive"] - df["negative"]) / total, 0)

    # Filter to only topics in topic_list
    df = df[df["topics_corrected"].isin(topic_list)]
    
    # Group by date and topic, then take mean of EI
    grouped = df.groupby(["date", "topics_corrected"])["ei"].mean().reset_index()
    
    # Pivot to wide format - this is much more memory efficient than multiple merges
    df_pivot = grouped.pivot(index="date", columns="topics_corrected", values="ei")
    
    # Rename columns to include type
    df_pivot.columns = [f"ei_{type}_{col}" for col in df_pivot.columns]
    
    # Reset index and fill NaN with 0
    df_out = df_pivot.reset_index().fillna(0)

    return df_out


def epi_topic(
    data_transcripts: pd.DataFrame, 
    data_comments: pd.DataFrame, 
    topic_list: list, 
    epsilon=1e-6
):
    """
    Calculate Emotion Polarity Index by topic - MEMORY OPTIMIZED VERSION.
    Only selects needed columns to avoid memory issues.
    """
    # Get only the columns we need (date + ei columns for each topic)
    trans_cols = ["date"] + [f"ei_creator_{t}" for t in topic_list if f"ei_creator_{t}" in data_transcripts.columns]
    comm_cols = ["date"] + [f"ei_community_{t}" for t in topic_list if f"ei_community_{t}" in data_comments.columns]
    
    df_trans = data_transcripts[trans_cols].copy()
    df_comm = data_comments[comm_cols].copy()
    
    df_trans["date"] = pd.to_datetime(df_trans["date"])
    df_comm["date"] = pd.to_datetime(df_comm["date"])

    # Merge two tables keep all dates and set missing values to 0
    df = pd.merge(df_trans, df_comm, on="date", how="outer").fillna(0)

    # Vectorized computation for all topics
    for topic in topic_list:
        creator_col = f"ei_creator_{topic}"
        community_col = f"ei_community_{topic}"
        
        # Skip if columns don't exist
        if creator_col not in df.columns or community_col not in df.columns:
            continue
        
        ei_creator = df[creator_col].to_numpy()
        ei_community = df[community_col].to_numpy()
        
        abs_creator = np.abs(ei_creator)
        abs_community = np.abs(ei_community)
        
        # Compute AEPI with Magnitude Scaling
        denominator = abs_creator + abs_community + epsilon
        agreement_score = 1 - 2 * np.abs((ei_creator - ei_community) / denominator)

        # Magnitude weight based on the size of epi values
        magnitude_weight = (abs_creator + abs_community) * 1.25

        # Final AEPI calculation with clipping
        df[f"epi_{topic}"] = np.clip(agreement_score * magnitude_weight, -1, 1)

    return df.drop_duplicates(subset=["date"], keep="first")
