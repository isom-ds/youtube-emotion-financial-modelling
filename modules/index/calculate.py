import numpy as np
import pandas as pd

# Define emotion columns once at module level (per index.tex Equation 6)
POS_EMO = ["anticipation", "joy", "trust"]
NEG_EMO = ["anger", "disgust", "fear", "sadness"]
SURPRISE = "surprise"


def apply_engagement_weights(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply engagement weighting to creator-level data per index.tex Equation 4.
    
    Engagement weight: a_v = log(1 + viewCount + likeCount + commentCount)
    
    This function should be applied to creator data BEFORE grouping by date.
    
    Parameters:
    - data: DataFrame with emotion columns and engagement metrics
    
    Returns:
    - DataFrame with engagement-weighted emotion columns
    """
    df = data.copy()
    
    # Calculate engagement weight per index.tex Equation 4
    df['engagement_weight'] = np.log(
        1 + df['viewCount'].fillna(0) + 
        df['likeCount'].fillna(0) + 
        df['commentCount'].fillna(0)
    )
    
    # Apply weight to emotion columns (row-level weighting)
    for emotion in POS_EMO + NEG_EMO + [SURPRISE]:
        if emotion in df.columns:
            df[f'{emotion}_weighted'] = df[emotion] * df['engagement_weight']
    
    return df


def ei(data: pd.DataFrame, type: str, weighted: bool = False, epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Calculate Emotion Index per index.tex Equation 6.
    
    EI = (sum of positive emotions - sum of negative emotions) / total emotions
    
    Parameters:
    - data: DataFrame with emotion columns (already grouped by date if needed)
    - type: 'creator' or 'community'
    - weighted: If True, use engagement-weighted emotion columns
    - epsilon: Small constant for numerical stability
    
    Returns:
    - DataFrame with EI column
    """
    df = data.copy()
    
    # Select appropriate emotion columns based on weighting
    suffix = '_weighted' if weighted else ''
    pos_cols = [f"{e}{suffix}" for e in POS_EMO if f"{e}{suffix}" in df.columns]
    neg_cols = [f"{e}{suffix}" for e in NEG_EMO if f"{e}{suffix}" in df.columns]
    
    # Sum by row for each group
    df["positive"] = df[pos_cols].sum(axis=1)
    df["negative"] = df[neg_cols].sum(axis=1)
    
    # Calculate EI per index.tex Equation 6
    total = df["positive"] + df["negative"]
    col_name = f"ei_{type}" if not weighted else f"ei_{type}_engweighted"
    df[col_name] = (df["positive"] - df["negative"]) / (total + epsilon)
    
    return df


def ccd_intensity(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame, 
                  weighted: bool = False) -> pd.DataFrame:
    """
    Calculate Creator-Community Divergence (CCD) and Emotional Intensity per index.tex Equations 8-9.
    
    CCD_t = |EI_creator - EI_community| / 2
    Int_t = (|EI_creator| + |EI_community|) / 2
    
    Parameters:
    - data_transcripts: DataFrame with ei_creator column
    - data_comments: DataFrame with ei_community column
    - weighted: If True, use engagement-weighted EI for creator (community is never engagement-weighted)
    
    Returns:
    - DataFrame with CCD and Intensity columns
    """
    creator_suffix = '_engweighted' if weighted else ''
    creator_col = f'ei_creator{creator_suffix}'
    community_col = 'ei_community'  # Community is never engagement-weighted
    
    # Merge on date
    df_trans = data_transcripts[["date", creator_col]].copy()
    df_comm = data_comments[["date", community_col]].copy()
    
    df_trans["date"] = pd.to_datetime(df_trans["date"])
    df_comm["date"] = pd.to_datetime(df_comm["date"])
    
    df = pd.merge(df_trans, df_comm, on="date", how="outer").fillna(0)
    
    ei_creator = df[creator_col].values
    ei_community = df[community_col].values
    
    # Calculate CCD and Intensity per index.tex Equations 8-9
    output_suffix = creator_suffix if creator_suffix else ''
    df[f"ccd{output_suffix}"] = np.abs(ei_creator - ei_community) / 2
    df[f"intensity{output_suffix}"] = (np.abs(ei_creator) + np.abs(ei_community)) / 2
    
    return df.drop_duplicates(subset=["date"], keep="first")


def epi(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame, 
        weighted: bool = False) -> pd.DataFrame:
    """
    Calculate Emotion Polarity Index per index.tex Equation 10.
    
    EPI_t = Int_t * CCD_t
    
    This is the NEW Iteration 2 formulation that separates intensity and divergence.
    Output range: [0, 1]
    
    Parameters:
    - data_transcripts: DataFrame with ei_creator column
    - data_comments: DataFrame with ei_community column
    - weighted: If True, use engagement-weighted EI columns
    
    Returns:
    - DataFrame with EPI, CCD, and Intensity columns
    """
    # First calculate CCD and Intensity
    df = ccd_intensity(data_transcripts, data_comments, weighted)
    
    suffix = '_engweighted' if weighted else ''
    ccd_col = f"ccd{suffix}"
    intensity_col = f"intensity{suffix}"
    
    # Calculate EPI per index.tex Equation 10
    df[f"epi{suffix}"] = df[intensity_col] * df[ccd_col]
    
    return df


def epi_signed(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame, 
               weighted: bool = False) -> pd.DataFrame:
    """
    Calculate Signed Emotion Polarity Index per index.tex (after Equation 10).
    
    EPI±_t = sign(EI_creator + EI_community) * EPI_t
    
    Parameters:
    - data_transcripts: DataFrame with ei_creator column
    - data_comments: DataFrame with ei_community column
    - weighted: If True, use engagement-weighted EI for creator
    
    Returns:
    - DataFrame with signed EPI column
    """
    df = epi(data_transcripts, data_comments, weighted)
    
    creator_suffix = '_engweighted' if weighted else ''
    creator_col = f'ei_creator{creator_suffix}'
    community_col = 'ei_community'
    epi_suffix = creator_suffix if creator_suffix else ''
    epi_col = f"epi{epi_suffix}"
    
    # Merge in the original EI columns if needed
    if creator_col not in df.columns:
        df_trans_ei = data_transcripts[['date', creator_col]].copy()
        df_trans_ei['date'] = pd.to_datetime(df_trans_ei['date'])
        df = pd.merge(df, df_trans_ei, on='date', how='left')
    
    if community_col not in df.columns:
        df_comm_ei = data_comments[['date', community_col]].copy()
        df_comm_ei['date'] = pd.to_datetime(df_comm_ei['date'])
        df = pd.merge(df, df_comm_ei, on='date', how='left')
    
    # Calculate signed variant
    df[f"epi_signed{epi_suffix}"] = np.sign(df[creator_col] + df[community_col]) * df[epi_col]
    
    return df


def surprise_index(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame,
                   topic_col: str = None, weighted: bool = False, 
                   epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Calculate Surprise Index per index.tex Equation 11.
    
    Measures informational novelty and unexpectedness.
    
    If topic_col is provided, computes volume-weighted surprise across topics.
    Otherwise, computes aggregate surprise index.
    
    Parameters:
    - data_transcripts: DataFrame with surprise emotion column
    - data_comments: DataFrame with surprise emotion column
    - topic_col: Column name for topic grouping (optional)
    - weighted: If True, use engagement-weighted for creators (community never weighted)
    - epsilon: Small constant for numerical stability
    
    Returns:
    - DataFrame with surprise index
    """
    trans_suffix = '_weighted' if weighted else ''
    comm_suffix = ''  # Community is never engagement-weighted
    trans_surprise_col = f'{SURPRISE}{trans_suffix}'
    comm_surprise_col = f'{SURPRISE}{comm_suffix}'
    
    if topic_col:
        # Topic-weighted version per index.tex Equation 11
        # Calculate topic volumes w_{k,t}
        trans_volumes = data_transcripts.groupby(['date', topic_col]).size().reset_index(name='vol_creator')
        comm_volumes = data_comments.groupby(['date', topic_col]).size().reset_index(name='vol_community')
        
        volumes = pd.merge(trans_volumes, comm_volumes, on=['date', topic_col], how='outer').fillna(0)
        volumes['topic_volume'] = volumes['vol_creator'] + volumes['vol_community']
        
        # Get surprise counts per topic
        trans_surprise = data_transcripts.groupby(['date', topic_col])[trans_surprise_col].sum().reset_index(name='surprise_creator')
        comm_surprise = data_comments.groupby(['date', topic_col])[comm_surprise_col].sum().reset_index(name='surprise_community')
        
        # Get total emotion counts per topic
        all_trans_cols = [f"{e}{trans_suffix}" for e in POS_EMO + NEG_EMO + [SURPRISE] if f"{e}{trans_suffix}" in data_transcripts.columns]
        all_comm_cols = [f"{e}{comm_suffix}" for e in POS_EMO + NEG_EMO + [SURPRISE] if f"{e}{comm_suffix}" in data_comments.columns]
        trans_total = data_transcripts.groupby(['date', topic_col])[all_trans_cols].sum().sum(axis=1).reset_index(name='total_creator')
        comm_total = data_comments.groupby(['date', topic_col])[all_comm_cols].sum().sum(axis=1).reset_index(name='total_community')
        
        # Merge everything
        df = volumes.copy()
        df = pd.merge(df, trans_surprise, on=['date', topic_col], how='left').fillna(0)
        df = pd.merge(df, comm_surprise, on=['date', topic_col], how='left').fillna(0)
        df = pd.merge(df, trans_total, on=['date', topic_col], how='left').fillna(0)
        df = pd.merge(df, comm_total, on=['date', topic_col], how='left').fillna(0)
        
        df['surprise_numerator'] = df['topic_volume'] * (df['surprise_creator'] + df['surprise_community'])
        df['surprise_denominator'] = df['topic_volume'] * (df['total_creator'] + df['total_community'])
        
        # Aggregate across topics
        result = df.groupby('date').agg({
            'surprise_numerator': 'sum',
            'surprise_denominator': 'sum'
        }).reset_index()
        
        output_suffix = '_engweighted_topicweighted' if weighted else '_topicweighted'
        result[f'surprise{output_suffix}'] = result['surprise_numerator'] / (result['surprise_denominator'] + epsilon)
        result['date'] = pd.to_datetime(result['date'])
        
        return result[['date', f'surprise{output_suffix}']]
    
    else:
        # Non-topic-weighted version (aggregate)
        df_trans = data_transcripts.groupby('date')[trans_surprise_col].sum().reset_index(name='surprise_creator')
        df_comm = data_comments.groupby('date')[comm_surprise_col].sum().reset_index(name='surprise_community')
        
        all_trans_cols = [f"{e}{trans_suffix}" for e in POS_EMO + NEG_EMO + [SURPRISE] if f"{e}{trans_suffix}" in data_transcripts.columns]
        all_comm_cols = [f"{e}{comm_suffix}" for e in POS_EMO + NEG_EMO + [SURPRISE] if f"{e}{comm_suffix}" in data_comments.columns]
        trans_total = data_transcripts.groupby('date')[all_trans_cols].sum().sum(axis=1).reset_index(name='total_creator')
        comm_total = data_comments.groupby('date')[all_comm_cols].sum().sum(axis=1).reset_index(name='total_community')
        
        df = pd.merge(df_trans, df_comm, on='date', how='outer').fillna(0)
        df = pd.merge(df, trans_total, on='date', how='left').fillna(0)
        df = pd.merge(df, comm_total, on='date', how='left').fillna(0)
        
        output_suffix = '_engweighted' if weighted else ''
        df[f'surprise{output_suffix}'] = (df['surprise_creator'] + df['surprise_community']) / \
                                          (df['total_creator'] + df['total_community'] + epsilon)
        df['date'] = pd.to_datetime(df['date'])
        
        return df[['date', f'surprise{output_suffix}']]


def split_index(data_comments: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    """
    Calculate Split Index for binary emotion classification per index.tex.
    
    Measures within-community polarization using dispersion of binary valence.
    
    Split_t = 4 * p_t * (1 - p_t)
    
    where p_t is the proportion of positive-valence comments on day t.
    Range: [0, 1], maximum at p_t = 0.5 (even split), minimum at p_t = 0 or 1 (unanimous).
    
    Parameters:
    - data_comments: DataFrame with comment-level emotion data
    - weighted: If True, use engagement-weighted EI (note: not typically used for comments)
    
    Returns:
    - DataFrame with split index for each date
    """
    df = data_comments.copy()
    
    suffix = '_weighted' if weighted else ''
    pos_cols = [f"{e}{suffix}" for e in POS_EMO if f"{e}{suffix}" in df.columns]
    neg_cols = [f"{e}{suffix}" for e in NEG_EMO if f"{e}{suffix}" in df.columns]
    
    # Calculate comment-level EI (binary: -1 or +1)
    df["positive"] = df[pos_cols].sum(axis=1)
    df["negative"] = df[neg_cols].sum(axis=1)
    
    total = df["positive"] + df["negative"]
    df['ei_comment'] = (df["positive"] - df["negative"]) / total.replace(0, np.nan)
    
    # Calculate proportion of positive comments per day
    result = df.groupby('date').agg(
        total_comments=('ei_comment', 'count'),
        positive_comments=('ei_comment', lambda x: (x == 1.0).sum())
    ).reset_index()
    
    # Calculate p_t (proportion with positive valence)
    result['p_t'] = result['positive_comments'] / result['total_comments']
    
    # Calculate Split_t = 4 * p_t * (1 - p_t)
    output_suffix = '_engweighted' if weighted else ''
    result[f'split{output_suffix}'] = 4 * result['p_t'] * (1 - result['p_t'])
    result['date'] = pd.to_datetime(result['date'])
    
    return result[['date', f'split{output_suffix}']]


def ei_topic(data: pd.DataFrame, type: str, topic_list: list, 
             topic_col: str = 'topics_corrected', weighted: bool = False,
             epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Calculate topic-level Emotion Index per index.tex Equation 6.
    
    Returns individual topic EIs (not aggregated).
    
    Parameters:
    - data: DataFrame with emotion and topic columns
    - type: 'creator' or 'community'
    - topic_list: List of topics to calculate EI for
    - topic_col: Column name containing topic assignments
    - weighted: If True, use engagement-weighted emotion columns
    - epsilon: Small constant for numerical stability
    
    Returns:
    - DataFrame with ei_{type}_{topic} columns for each topic
    """
    needed_cols = ["date", topic_col]
    suffix = '_weighted' if weighted else ''
    
    for emo in POS_EMO + NEG_EMO:
        col = f"{emo}{suffix}"
        if col in data.columns:
            needed_cols.append(col)
    
    df = data[needed_cols].copy()
    
    pos_cols = [f"{e}{suffix}" for e in POS_EMO if f"{e}{suffix}" in df.columns]
    neg_cols = [f"{e}{suffix}" for e in NEG_EMO if f"{e}{suffix}" in df.columns]
    
    # Pre-calculate positive and negative sums
    df["positive"] = df[pos_cols].sum(axis=1)
    df["negative"] = df[neg_cols].sum(axis=1)
    
    # Calculate EI for all rows
    total = df["positive"] + df["negative"]
    df["ei"] = np.where(total > 0, (df["positive"] - df["negative"]) / (total + epsilon), 0)
    
    # Filter to topics in topic_list
    df = df[df[topic_col].isin(topic_list)]
    
    # Group by date and topic
    grouped = df.groupby(["date", topic_col])["ei"].mean().reset_index()
    
    # Pivot to wide format
    df_pivot = grouped.pivot(index="date", columns=topic_col, values="ei")
    
    # Rename columns
    output_suffix = '_engweighted' if weighted else ''
    df_pivot.columns = [f"ei_{type}{output_suffix}_{col}" for col in df_pivot.columns]
    
    df_out = df_pivot.reset_index().fillna(0)
    
    return df_out


def ei_topic_aggregated(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame,
                        topic_list: list, topic_col: str = 'topics_corrected',
                        weighted: bool = False, epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Calculate volume-weighted aggregate EI across topics per index.tex Equation 7.
    
    EI_t = Σ(w_{k,t} * EI_{k,t}) / Σ(w_{k,t})
    
    where w_{k,t} = topic volume (number of textual items)
    
    Parameters:
    - data_transcripts: Creator-level data
    - data_comments: Community-level data
    - topic_list: List of topics
    - topic_col: Column name for topics
    - weighted: If True, use engagement-weighted emotion columns
    - epsilon: Small constant
    
    Returns:
    - DataFrame with volume-weighted aggregate EI for creator and community
    """
    # Get topic-level EIs
    df_trans_ei = ei_topic(data_transcripts, 'creator', topic_list, topic_col, weighted, epsilon)
    df_comm_ei = ei_topic(data_comments, 'community', topic_list, topic_col, weighted, epsilon)
    
    # Calculate topic volumes (unique textual items per topic per day)
    trans_volumes = data_transcripts.groupby(['date', topic_col]).size().reset_index(name='vol_creator')
    comm_volumes = data_comments.groupby(['date', topic_col]).size().reset_index(name='vol_community')
    
    volumes = pd.merge(trans_volumes, comm_volumes, on=['date', topic_col], how='outer').fillna(0)
    volumes['topic_volume'] = volumes['vol_creator'] + volumes['vol_community']
    
    # Pivot volumes to wide format
    vol_pivot = volumes.pivot(index='date', columns=topic_col, values='topic_volume').fillna(0)
    
    # Merge EIs with volumes
    df_trans = pd.merge(df_trans_ei, vol_pivot.reset_index(), on='date', how='left')
    df_comm = pd.merge(df_comm_ei, vol_pivot.reset_index(), on='date', how='left')
    
    # Calculate weighted sums
    suffix = '_engweighted' if weighted else ''
    
    for source_type, df_source in [('creator', df_trans), ('community', df_comm)]:
        numerator = 0
        denominator = 0
        
        for topic in topic_list:
            ei_col = f"ei_{source_type}{suffix}_{topic}"
            vol_col = topic
            
            if ei_col in df_source.columns and vol_col in df_source.columns:
                numerator += df_source[ei_col] * df_source[vol_col]
                denominator += df_source[vol_col]
        
        output_col = f"ei_{source_type}{suffix}_topicweighted"
        df_source[output_col] = numerator / (denominator + epsilon)
        
        if source_type == 'creator':
            df_result = df_source[['date', output_col]].copy()
        else:
            df_result = pd.merge(df_result, df_source[['date', output_col]], on='date', how='outer')
    
    return df_result


def epi_topic(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame, 
              topic_list: list, topic_col: str = 'topics_corrected',
              weighted: bool = False) -> pd.DataFrame:
    """
    Calculate topic-level EPI per index.tex Equation 10 applied to each topic.
    
    Returns individual topic EPIs (not aggregated) with signed variants.
    
    Parameters:
    - data_transcripts: DataFrame with topic-level EI columns
    - data_comments: DataFrame with topic-level EI columns
    - topic_list: List of topics
    - topic_col: Column name for topics
    - weighted: If True, use engagement-weighted EI columns
    
    Returns:
    - DataFrame with epi_{topic}, ccd_{topic}, intensity_{topic}, epi_signed_{topic} columns
    """
    # Get topic-level EIs
    df_trans = ei_topic(data_transcripts, 'creator', topic_list, topic_col, weighted)
    df_comm = ei_topic(data_comments, 'community', topic_list, topic_col, weighted)
    
    # Merge
    df = pd.merge(df_trans, df_comm, on='date', how='outer').fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    
    suffix = '_engweighted' if weighted else ''
    
    # Calculate CCD, Intensity, EPI, and SIGNED EPI for each topic
    for topic in topic_list:
        creator_col = f"ei_creator{suffix}_{topic}"
        community_col = f"ei_community{suffix}_{topic}"
        
        if creator_col in df.columns and community_col in df.columns:
            ei_creator = df[creator_col].values
            ei_community = df[community_col].values
            
            # Per index.tex Equations 8-10
            df[f"ccd{suffix}_{topic}"] = np.abs(ei_creator - ei_community) / 2
            df[f"intensity{suffix}_{topic}"] = (np.abs(ei_creator) + np.abs(ei_community)) / 2
            df[f"epi{suffix}_{topic}"] = df[f"intensity{suffix}_{topic}"] * df[f"ccd{suffix}_{topic}"]
            
            # ADD SIGNED VARIANT for each topic
            df[f"epi_signed{suffix}_{topic}"] = np.sign(ei_creator + ei_community) * df[f"epi{suffix}_{topic}"]
    
    return df.drop_duplicates(subset=["date"], keep="first")


def epi_topic_aggregated(data_transcripts: pd.DataFrame, data_comments: pd.DataFrame,
                         topic_list: list, topic_col: str = 'topics_corrected',
                         weighted: bool = False, epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Calculate volume-weighted aggregate EPI across topics.
    
    Uses volume weights based on number of textual items per topic.
    
    Parameters:
    - data_transcripts: Creator-level data
    - data_comments: Community-level data
    - topic_list: List of topics
    - topic_col: Column name for topics
    - weighted: If True, use engagement-weighted versions
    - epsilon: Small constant
    
    Returns:
    - DataFrame with volume-weighted EPI, CCD, Intensity, and signed EPI
    """
    # First get volume-weighted EIs
    df_ei_agg = ei_topic_aggregated(data_transcripts, data_comments, topic_list, topic_col, weighted, epsilon)
    
    suffix = '_engweighted' if weighted else ''
    creator_col = f"ei_creator{suffix}_topicweighted"
    community_col = f"ei_community{suffix}_topicweighted"
    
    # Calculate CCD, Intensity, and EPI from aggregated EIs
    ei_creator = df_ei_agg[creator_col].values
    ei_community = df_ei_agg[community_col].values
    
    df_ei_agg[f"ccd{suffix}_topicweighted"] = np.abs(ei_creator - ei_community) / 2
    df_ei_agg[f"intensity{suffix}_topicweighted"] = (np.abs(ei_creator) + np.abs(ei_community)) / 2
    df_ei_agg[f"epi{suffix}_topicweighted"] = df_ei_agg[f"intensity{suffix}_topicweighted"] * df_ei_agg[f"ccd{suffix}_topicweighted"]
    
    # Add signed variant
    df_ei_agg[f"epi_signed{suffix}_topicweighted"] = np.sign(ei_creator + ei_community) * df_ei_agg[f"epi{suffix}_topicweighted"]
    
    return df_ei_agg
