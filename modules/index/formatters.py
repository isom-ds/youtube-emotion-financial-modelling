import multiprocessing
import pandas as pd
from copy import deepcopy
from itertools import chain
from tqdm.notebook import tqdm
from constants import plutchik, topic_list


def create_col_from_emo(data: pd.DataFrame, cols: list = plutchik) -> pd.DataFrame:

    df = deepcopy(data)

    # Create separate columns using .map()
    for emotion in cols:
        df[emotion] = df["emotions"].map(
            lambda x: 1 if isinstance(x, list) and emotion in x else 0
        )

    return df


def create_col_from_topic(data: pd.DataFrame, cols: list = topic_list) -> pd.DataFrame:

    df = deepcopy(data)

    # Create separate columns using .map()
    for topic in cols:
        df[topic] = df["topics"].map(
            lambda x: 1 if isinstance(x, list) and topic in x else 0
        )

    return df


def convert_json_to_list(data):
    """
    Converts the nested dictionary structure into a list of dictionaries with separate topic-emotion pairs.

    Parameters:
    - data (dict): The input dictionary containing 'id', 'topics', and 'emotions'.

    Returns:
    - list: A list of dictionaries with 'id', 'topics', and 'emotions'.
    """
    results = []
    entry_id = data["id"]

    if isinstance(data["content"], dict):
        topics = data["content"]["topics"]
        emotions_list = data["content"]["emotions"]

        for topic, emotions in zip(topics, emotions_list):
            results.append({"id": entry_id, "topics": topic, "emotions": emotions})
    else:
        results.append({"id": entry_id, "topics": None, "emotions": None})

    return results


def convert_json_to_df(data: list, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the nested dictionary structure into a DataFrame with separate topic-emotion pairs.

    Parameters:
    - data (dict): The input dictionary containing 'id', 'topics', and 'emotions'.
    - dataframe (pd.DataFrame): The DataFrame to be merged.

    Returns:
    - pd.DataFrame: A DataFrame with 'id', 'topics', and 'emotions'.
    """
    ctx = multiprocessing.get_context("fork")

    with ctx.Pool() as pool:
        processed = list(tqdm(pool.imap(convert_json_to_list, data), total=len(data)))

    flattened_list = list(chain.from_iterable(processed))

    flattened_df = pd.DataFrame(flattened_list)
    df = deepcopy(dataframe)

    return flattened_df.merge(df, on="id", how="left")
