import numpy as np
import pandas as pd
from copy import deepcopy
from constants import topic_list


def ei(data: pd.DataFrame, type: str) -> pd.DataFrame:

    df = deepcopy(data)

    pos_emo = ["anticipation", "joy", "trust", "surprise"]
    neg_emo = ["anger", "disgust", "fear", "sadness"]

    # Sum by row for each group
    df["positive"] = df[pos_emo].sum(axis=1)
    df["negative"] = df[neg_emo].sum(axis=1)

    # Calculate EPI
    df[f"ei_{type}"] = (df["positive"] - df["negative"]) / (
        df["positive"] + df["negative"]
    )

    return df


def epi(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame, epsilon=1e-6):
    df_transcripts = deepcopy(data_transcripts)
    df_comments = deepcopy(data_comments)

    # Convert to datetime
    df_transcripts["date"] = pd.to_datetime(df_transcripts["date"])
    df_comments["date"] = pd.to_datetime(df_comments["date"])

    # Merge two tables keep all dates and set missing values to 0
    df = pd.merge(
        df_transcripts[["date", "ei_creator"]],
        df_comments[["date", "ei_community"]],
        on="date",
        how="outer",
    ).fillna(0)

    # Compute AEPI with Magnitude Scaling
    agreement_score = 1 - 2 * abs(
        (df["ei_creator"] - df["ei_community"])
        / (abs(df["ei_creator"]) + abs(df["ei_community"]) + epsilon)
    )

    # Magnitude weight based on the size of epi values
    magnitude_weight = (abs(df["ei_creator"]) + abs(df["ei_community"])) * 1.25

    # Final AEPI calculation
    df["epi"] = agreement_score * magnitude_weight

    # Limit AEPI between -1 and 1
    df["epi"] = np.clip(df["epi"], -1, 1)
    # df['epi'] = (2 * (df['epi'] - df['epi'].min()) / ((df['epi'].max()) - (df['epi'].min()))) - 1

    # Cases where there the epi_creator or epi_community is 0 set aepi to 0
    # df['epi'] = np.where((df['ei_creator'] == 0) | (df['ei_community'] == 0), 0, df['epi'])

    return df.drop_duplicates(subset=["date"], keep="first")


def ei_topic(
    data: pd.DataFrame,
    type: str,
):
    df = deepcopy(data)

    pos_emo = ["anticipation", "joy", "trust", "surprise"]
    neg_emo = ["anger", "disgust", "fear", "sadness"]

    all_dfs = []

    # Calculate EPI
    for i in topic_list:
        topic_df = df[df["topics_corrected"] == i]
        topic_df["positive"] = topic_df[pos_emo].sum(axis=1)
        topic_df["negative"] = topic_df[neg_emo].sum(axis=1)
        topic_df[f"ei_{type}_{i}"] = (topic_df["positive"] - topic_df["negative"]) / (
            topic_df["positive"] + topic_df["negative"]
        )

        all_dfs.append(topic_df[["date", f"ei_{type}_{i}"]])

    df_out = all_dfs[0]
    for i in range(1, len(all_dfs)):
        df_out = pd.merge(df_out, all_dfs[i], on="date", how="outer").fillna(0)

    return df_out


def epi_topic(
    data_transcripts: pd.DataFrame, data_comments: pd.DataFrame, epsilon=1e-6
):
    df_transcripts = deepcopy(data_transcripts)
    df_comments = deepcopy(data_comments)

    # Convert to datetime
    df_transcripts["date"] = pd.to_datetime(df_transcripts["date"])
    df_comments["date"] = pd.to_datetime(df_comments["date"])

    # Merge two tables keep all dates and set missing values to 0
    df = pd.merge(df_transcripts, df_comments, on="date", how="outer").fillna(0)

    # Compute AEPI with Magnitude Scaling
    for i in topic_list:
        agreement_score = 1 - 2 * abs(
            (df[f"ei_creator_{i}"] - df[f"ei_community_{i}"])
            / (abs(df[f"ei_creator_{i}"]) + abs(df[f"ei_community_{i}"]) + epsilon)
        )

        # Magnitude weight based on the size of epi values
        magnitude_weight = (
            abs(df[f"ei_creator_{i}"]) + abs(df[f"ei_community_{i}"])
        ) * 1.25

        # Final AEPI calculation
        df[f"epi_{i}"] = agreement_score * magnitude_weight

        # Limit AEPI between -1 and 1
        df[f"epi_{i}"] = np.clip(df[f"epi_{i}"], -1, 1)
    # df['epi'] = (2 * (df['epi'] - df['epi'].min()) / ((df['epi'].max()) - (df['epi'].min()))) - 1

    # Cases where there the epi_creator or epi_community is 0 set aepi to 0
    # df['epi'] = np.where((df['ei_creator'] == 0) | (df['ei_community'] == 0), 0, df['epi'])

    return df.drop_duplicates(subset=["date"], keep="first")
