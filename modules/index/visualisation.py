import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from constants import topic_list
import datetime as dt

# Define emotion groups and their colors
emocols = [
    "date",
    "anger",
    "disgust",
    "fear",
    "sadness",
    "anticipation",
    "joy",
    "trust",
    "surprise",
]
emocolours = {
    "joy": "#023047",
    "trust": "#126782",
    "anticipation": "#219EBC",
    "surprise": "#DAA520",
    "anger": "#B62B2B",
    "fear": "#D42929",
    "disgust": "#F04242",
    "sadness": "#FF8787",
}

# Define topic groups and their colors
topiccols = [
    'date',
    'Conspiracy and Historical Events',
    'Disaster and Resilience',
    'Disputes and Opinions',
    'Economic and Financial Matters',
    'Education and Knowledge',
    'Environmental and Infrastructure',
    'Health and Social Welfare',
    'Labor and Workforce',
    'Leadership and Governance',
    'Legal and Regulatory Framework',
    'Miscellaneous and Other',
    'Power and Influence Dynamics',
    'Psychological and Emotional Aspects',
    'Quality of Life and Standards',
    'Security and Conflict',
    'Social and Cultural Aspects',
    'Sports and Recreation',
    'Trust, Credibility, and Communication'
]
topiccolours = {
    'Conspiracy and Historical Events': '#B62B2B',
    'Disaster and Resilience': '#8ECAE6',
    'Disputes and Opinions': '#F04242',
    'Economic and Financial Matters': '#023047',
    'Education and Knowledge': '#219EBC',
    'Environmental and Infrastructure': '#38B000',
    'Health and Social Welfare': '#FFB703',
    'Labor and Workforce': '#F4A259',
    'Leadership and Governance': '#FB8500',
    'Legal and Regulatory Framework': '#8D99AE',
    'Miscellaneous and Other': '#6c757d',
    'Power and Influence Dynamics': '#126782',
    'Psychological and Emotional Aspects': '#A7C957',
    'Quality of Life and Standards': '#B5838D',
    'Security and Conflict': '#D42929',
    'Social and Cultural Aspects': '#FF8787',
    'Sports and Recreation': '#58B4D1',
    'Trust, Credibility, and Communication': '#219EBC',
}

# Dates for filtering
min_date = dt.date(2025, 1, 6)
max_date = dt.date(2025, 5, 31)
filter_date = dt.date(2025, 1, 13)

def plot_stack100(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    datatype: str,
    legend_cols: int = 8,
    legend_add_gap: float = 0.0,
):
    if datatype == "emotion":
        columns = emocols
        colours = emocolours

    if datatype == "topic":
        columns = topiccols
        colours = topiccolours

    df1 = deepcopy(data1)
    df2 = deepcopy(data2)
    df1 = df1[columns]
    df2 = df2[columns]

    # Set 'date' as index
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df1 = df1[df1["date"].dt.date >= filter_date]
    df1.set_index("date", inplace=True)
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")
    df2 = df2[df2["date"].dt.date >= filter_date]
    df2.set_index("date", inplace=True)

    # Create common index
    common_idx = df1.index.intersection(df2.index).sort_values()

    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]

    plt.rcParams.update({"font.size": 18})

    # Create figure with two subplots (side by side)
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(20, 9 if datatype == "emotion" else 11), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # Normalize values so each row sums to 1 (100% stacked)
    df1_normalized = df1.div(df1.sum(axis=1), axis=0) * 100
    df2_normalized = df2.div(df2.sum(axis=1), axis=0) * 100

    # Plot 100% stacked bar chart
    df1_normalized.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        width=0.8,
        color=[colours[col] for col in df1_normalized.columns],
    )
    df2_normalized.plot(
        kind="bar",
        stacked=True,
        ax=ax3,
        width=0.8,
        color=[colours[col] for col in df2_normalized.columns],
    )

    # Formatting
    ax1.set_xlabel("")
    ax3.set_xlabel("")
    ax1.set_ylabel("Content Creators")
    ax3.set_ylabel("Community")

    # Format x-axis dates as 'DD-Mon'
    week_start_dates = df2.index[
        df2.index.weekday == 0
    ]  # Get only the Monday dates in the dataset
    ax3.set_xticks(
        [df2.index.get_loc(date) for date in week_start_dates]
    )  # Set ticks only at those dates
    ax3.set_xticklabels(week_start_dates.strftime("%d-%b"))  # Format labels correctly

    # Clear top ticks (shared axis will show them on bottom only)
    ax1.tick_params(labelbottom=False)

    # Turn off legends for all ax
    ax1.get_legend().remove()
    ax3.get_legend().remove()

    # Add shared titles for groups
    # ax1.set_title(titlestr, fontsize=20, pad=20, fontweight="bold")

    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # # Move legend outside
    plt.legend(
        title="",
        bbox_to_anchor=(0.95, -0.4 + legend_add_gap),
        ncol=legend_cols,
        loc="right",
        fontsize="18",
    )

    plt.tight_layout()

    # Show plot
    plt.show()


def plot_stack100_split(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    data3: pd.DataFrame,
    data4: pd.DataFrame,
    datatype: str,
    legend_cols: int = 8,
    legend_add_gap: float = 0.0,
):
    if datatype == "emotion":
        columns = emocols
        colours = emocolours

    if datatype == "topic":
        columns = topiccols
        colours = topiccolours

    df1 = deepcopy(data1)
    df2 = deepcopy(data2)
    df3 = deepcopy(data3)
    df4 = deepcopy(data4)
    df1 = df1[columns]
    df2 = df2[columns]
    df3 = df3[columns]
    df4 = df4[columns]

    # Set 'date' as index
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df1 = df1[df1["date"].dt.date >= filter_date]
    df1.set_index("date", inplace=True)
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")
    df2 = df2[df2["date"].dt.date >= filter_date]
    df2.set_index("date", inplace=True)
    df3["date"] = pd.to_datetime(df3["date"], format="%Y-%m-%d")
    df3 = df3[df3["date"].dt.date >= filter_date]
    df3.set_index("date", inplace=True)
    df4["date"] = pd.to_datetime(df4["date"], format="%Y-%m-%d")
    df4 = df4[df4["date"].dt.date >= filter_date]
    df4.set_index("date", inplace=True)

    common_idx1 = df1.index.intersection(df2.index).sort_values()
    common_idx2 = df3.index.intersection(df4.index).sort_values()

    df1 = df1.loc[common_idx1]
    df2 = df2.loc[common_idx1]
    df3 = df3.loc[common_idx2]
    df4 = df4.loc[common_idx2]

    plt.rcParams.update({"font.size": 18})

    # Create figure with two subplots (side by side)
    fig, (ax1, ax3, ax5, ax7) = plt.subplots(
        1, 4, figsize=(20, 4), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    # Normalize values so each row sums to 1 (100% stacked)
    df1_normalized = df1.div(df1.sum(axis=1), axis=0) * 100
    df2_normalized = df2.div(df2.sum(axis=1), axis=0) * 100
    df3_normalized = df3.div(df3.sum(axis=1), axis=0) * 100
    df4_normalized = df4.div(df4.sum(axis=1), axis=0) * 100

    # Plot 100% stacked bar chart
    df1_normalized.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        width=0.8,
        color=[colours[col] for col in df1_normalized.columns],
    )
    df2_normalized.plot(
        kind="bar",
        stacked=True,
        ax=ax3,
        width=0.8,
        color=[colours[col] for col in df2_normalized.columns],
    )
    df3_normalized.plot(
        kind="bar",
        stacked=True,
        ax=ax5,
        width=0.8,
        color=[colours[col] for col in df3_normalized.columns],
    )
    df4_normalized.plot(
        kind="bar",
        stacked=True,
        ax=ax7,
        width=0.8,
        color=[colours[col] for col in df4_normalized.columns],
    )

    # Formatting
    ax1.set_xlabel("")
    ax3.set_xlabel("")
    ax5.set_xlabel("")
    ax7.set_xlabel("")
    ax1.set_ylabel("Proportion (%)")

    # Format x-axis dates as 'DD-Mon'
    week_start_dates = df1.index[
        df1.index.weekday == 0
    ]  # Get only the Monday dates in the dataset
    ax1.set_xticks(
        [df1.index.get_loc(date) for date in week_start_dates]
    )  # Set ticks only at those dates
    ax1.set_xticklabels(week_start_dates.strftime("%d-%b"))  # Format labels correctly
    ax1.set_title("Content Creator")
    week_start_dates = df2.index[
        df2.index.weekday == 0
    ]  # Get only the Monday dates in the dataset
    ax3.set_xticks(
        [df2.index.get_loc(date) for date in week_start_dates]
    )  # Set ticks only at those dates
    ax3.set_xticklabels(week_start_dates.strftime("%d-%b"))  # Format labels correctly
    ax3.set_title("Community")
    week_start_dates = df3.index[
        df3.index.weekday == 0
    ]  # Get only the Monday dates in the dataset
    ax5.set_xticks(
        [df3.index.get_loc(date) for date in week_start_dates]
    )  # Set ticks only at those dates
    ax5.set_xticklabels(week_start_dates.strftime("%d-%b"))  # Format labels correctly
    ax5.set_title("Content Creator")
    week_start_dates = df4.index[
        df4.index.weekday == 0
    ]  # Get only the Monday dates in the dataset
    ax7.set_xticks(
        [df4.index.get_loc(date) for date in week_start_dates]
    )  # Set ticks only at those dates
    ax7.set_xticklabels(week_start_dates.strftime("%d-%b"))  # Format labels correctly
    ax7.set_title("Community")

    # Turn off legends for all ax
    ax1.get_legend().remove()
    ax3.get_legend().remove()
    ax5.get_legend().remove()
    ax7.get_legend().remove()

    # Add shared titles for groups
    fig.text(
        0.305, 0.97, "February 2025", ha="center", fontsize=20, fontweight="bold"
    )
    fig.text(
        0.715, 0.97, "May 2025", ha="center", fontsize=20, fontweight="bold"
    )

    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

    # Move legend outside
    plt.legend(
        title="",
        bbox_to_anchor=(-1.125, -0.4 + legend_add_gap),
        ncol=legend_cols,
        loc="center",
        fontsize="18",
    )

    plt.tight_layout()

    # Show plot
    plt.show()

def plot_indices(
    df: pd.DataFrame,
    index_type: str = 'epi',
    include_variants: list = None,
    figsize: tuple = (20, 10),
    ylim: tuple = None,
    add_events: bool = True,
):
    """
    Plot indices over time grouped by type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the index data with 'date' column
    index_type : str
        Type of index to plot. Options: 'ei', 'epi', 'epi_signed', 'ccd', 
        'intensity', 'surprise', 'split'
    include_variants : list, optional
        Specific column names to plot. If None, plots all variants of the index_type
    figsize : tuple
        Figure size (width, height)
    ylim : tuple, optional
        Y-axis limits (min, max). If None, auto-scales
    add_events : bool
        Whether to add event annotations and shading
    """
    
    df1 = deepcopy(df)
    
    # Convert 'date' column to datetime format
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df1 = df1[df1["date"].dt.date >= filter_date]
    
    plt.rcParams.update({"font.size": 20})
    
    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    # Define index configurations
    index_configs = {
        'ei': {
            'columns': ['ei_creator', 'ei_community_x', 'ei_creator_engweighted', 'ei_creator_topicweighted', 'ei_community_topicweighted'],
            'colors': ['#023047', '#219EBC', '#126782', '#023047', '#219EBC'],
            'linestyles': ['-', '-', '-.', '--', '--'],
            'linewidths': [4, 4, 2, 3, 3],
            'labels': ['Creator EI', 'Community EI', 'Creator EI (Eng. Weighted)', 'Creator EI (Topic Weighted)', 'Community EI (Topic Weighted)'],
            'ylim': (-1, 1),
            'ylabel': 'Emotion Index'
        },
        'epi': {
            'columns': ['epi', 'epi_engweighted', 'epi_topicweighted', 'epi_engweighted_topicweighted'],
            'colors': ['#B62B2B', '#D42929', '#F04242', '#FF8787'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['EPI', 'EPI (Eng. Weighted)', 'EPI (Topic Weighted)', 'EPI (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Emotion Polarity Index'
        },
        'epi_signed': {
            'columns': ['epi_signed', 'epi_signed_engweighted', 'epi_signed_topicweighted', 'epi_signed_engweighted_topicweighted'],
            'colors': ['#B62B2B', '#D42929', '#F04242', '#FF8787'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['EPI Signed', 'EPI Signed (Eng. Weighted)', 'EPI Signed (Topic Weighted)', 'EPI Signed (Both Weighted)'],
            'ylim': (-1, 1),
            'ylabel': 'Signed Emotion Polarity Index'
        },
        'ccd': {
            'columns': ['ccd', 'ccd_engweighted', 'ccd_topicweighted', 'ccd_engweighted_topicweighted'],
            'colors': ['#FB8500', '#F4A259', '#FAA307', '#FFD60A'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['CCD', 'CCD (Eng. Weighted)', 'CCD (Topic Weighted)', 'CCD (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Creator-Community Divergence'
        },
        'intensity': {
            'columns': ['intensity', 'intensity_engweighted', 'intensity_topicweighted', 'intensity_engweighted_topicweighted'],
            'colors': ['#126782', '#219EBC', '#126782', '#219EBC'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['Intensity', 'Intensity (Eng. Weighted)', 'Intensity (Topic Weighted)', 'Intensity (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Emotional Intensity'
        },
        'surprise': {
            'columns': ['surprise', 'surprise_engweighted', 'surprise_topicweighted', 'surprise_engweighted_topicweighted'],
            'colors': ['#DAA520', '#FFB703', '#F4A259', '#FF8787'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['Surprise', 'Surprise (Eng. Weighted)', 'Surprise (Topic Weighted)', 'Surprise (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Surprise Index (Informational Novelty)'
        },
        'split': {
            'columns': ['split'],
            'colors': ['#8D99AE'],
            'linestyles': ['-'],
            'linewidths': [4],
            'labels': ['Split Index'],
            'ylim': (0, 1),
            'ylabel': 'Split Index (Within-Community Polarization)'
        }
    }
    
    # Get configuration for selected index type
    if index_type not in index_configs:
        raise ValueError(f"Invalid index_type. Choose from: {list(index_configs.keys())}")
    
    config = index_configs[index_type]
    
    # Determine which columns to plot
    if include_variants:
        columns_to_plot = [col for col in include_variants if col in df1.columns]
        # Find corresponding indices for colors, linestyles, etc.
        plot_indices_list = [config['columns'].index(col) if col in config['columns'] else 0 
                             for col in columns_to_plot]
        colors = [config['colors'][i] for i in plot_indices_list]
        linestyles = [config['linestyles'][i] for i in plot_indices_list]
        linewidths = [config['linewidths'][i] for i in plot_indices_list]
        labels = [config['labels'][config['columns'].index(col)] if col in config['columns'] 
                  else col for col in columns_to_plot]
    else:
        columns_to_plot = [col for col in config['columns'] if col in df1.columns]
        colors = config['colors'][:len(columns_to_plot)]
        linestyles = config['linestyles'][:len(columns_to_plot)]
        linewidths = config['linewidths'][:len(columns_to_plot)]
        labels = config['labels'][:len(columns_to_plot)]
    
    # Plot each column
    for col, color, linestyle, linewidth, label in zip(columns_to_plot, colors, linestyles, linewidths, labels):
        ax1.plot(
            df1["date"],
            df1[col],
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=linewidth,
        )
    
    # Calculate y-axis limits based on actual data
    data_max = df1[columns_to_plot].max().max()
    data_min = df1[columns_to_plot].min().min()
    
    # Set y-axis limits
    if ylim:
        ax1.set_ylim(ylim)
    else:
        # Use data-driven limits with some padding
        y_range = data_max - data_min
        y_padding = y_range * 0.2  # 20% padding
        ul_padding = min(data_max + y_padding, 1.04)
        ax1.set_ylim(data_min - (y_padding/10) , ul_padding)
    
    # Add event annotations if requested
    if add_events:
        # Get y-position for all text (aligned to top)
        y_max = ax1.get_ylim()[1]
        y_text_pos = y_max * 0.95 if ul_padding < 1.04 else y_max * 0.99  # Position text at 95% of y-axis height
        
        # Initial Tariff Announcements (20 Jan - 26 Jan)
        ax1.axvspan(
            pd.to_datetime("2025-01-20"),
            pd.to_datetime("2025-01-26"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-01-23"), y_text_pos, "Initial Tariff\nAnnouncements", 
                color="black", ha="center", va="top", fontsize=18)
        
        # Tariffs Postponed (04 Mar - 06 Mar)
        ax1.axvspan(
            pd.to_datetime("2025-03-04"),
            pd.to_datetime("2025-03-06"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-02-28"), y_text_pos, "Tariffs\nPostponed", 
                color="black", ha="center", va="top", fontsize=18)
        
        # China Retaliation & US Tariffs Imposed (10 Mar - 13 Mar)
        ax1.axvspan(
            pd.to_datetime("2025-03-10"),
            pd.to_datetime("2025-03-13"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-03-12"), y_text_pos, 
                "China\nRetaliation\n& US Tariffs\nImposed", 
                color="black", ha="center", va="top", fontsize=18)
        
        # Global Tariffs & China Retaliation (2 Apr - 6 Apr)
        ax1.axvspan(
            pd.to_datetime("2025-04-02"),
            pd.to_datetime("2025-04-06"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-03-30"), y_text_pos, 
                "Global Tariffs\n& China\nRetaliation", 
                color="black", ha="center", va="top", fontsize=18)
        
        # US-China Tariff Threats (9 Apr - 12 Apr)
        ax1.axvspan(
            pd.to_datetime("2025-04-09"),
            pd.to_datetime("2025-04-12"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-04-12"), y_text_pos, "US-China\nTariff\nThreats", 
                color="black", ha="center", va="top", fontsize=18)
        
        # Tariffs Relaxed (27 Apr - 04 May)
        ax1.axvspan(
            pd.to_datetime("2025-04-27"),
            pd.to_datetime("2025-05-04"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-04-28"), y_text_pos, "Tariffs\nRelaxed", 
                color="black", ha="center", va="top", fontsize=18)
        
        # Trade Deals & Tariff Rollbacks (05 May - 14 May)
        ax1.axvspan(
            pd.to_datetime("2025-05-05"),
            pd.to_datetime("2025-05-14"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-05-10"), y_text_pos, 
                "Trade Deals\n& US-China\nTariff\nRollbacks", 
                color="black", ha="center", va="top", fontsize=18)
        
        # Non-US Apple Tariff Threat (20 May - 25 May)
        ax1.axvspan(
            pd.to_datetime("2025-05-20"),
            pd.to_datetime("2025-05-25"),
            color="gray",
            alpha=0.3,
        )
        ax1.text(pd.to_datetime("2025-05-25"), y_text_pos, 
                "Non-US\nManufactured\nApple\nTariff Threat", 
                color="black", ha="center", va="top", fontsize=18)
    
    # Formatting
    week_start_dates = (
        df1["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )
    ax1.set_xticks(week_start_dates)
    ax1.set_xticklabels(week_start_dates.dt.strftime("%d-%b"))
    
    # Set labels
    ax1.set_ylabel(config['ylabel'])
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(
        lines1,
        labels1,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(labels1), 4),
        fontsize=18,
    )
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    # Show plot
    plt.show()

def plot_indices_with_returns(
    df: pd.DataFrame,
    index_type: str = 'epi',
    include_variants: list = None,
    figsize: tuple = (20, 20),
    ylim_indices: tuple = None,
    add_events: bool = True,
    returns_cols: list = ['r_btc', 'r_gold', 'r_spx', 'r_spx_auto', 'r_spx_oilgas', 'r_spx_mach', 'r_spx_elec', 'r_aapl'],
):
    """
    Plot indices over time with asset returns on dual y-axes.
    Creates a 3x1 subplot for each asset (Bitcoin, Gold, S&P 500).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the index data and returns. Can have 'date' column or datetime index
    index_type : str
        Type of index to plot. Options: 'ei', 'epi', 'epi_signed', 'ccd', 
        'intensity', 'surprise', 'split'
    include_variants : list, optional
        Specific column names to plot. If None, plots all variants of the index_type
    figsize : tuple
        Figure size (width, height)
    ylim_indices : tuple, optional
        Y-axis limits for indices (min, max). If None, auto-scales
    add_events : bool
        Whether to add event annotations and shading
    returns_cols : list
        List of return column names [btc, gold, spx, spx_auto, spx_oilgas, spx_mach, spx_elec, aapl]
    """
    
    df1 = deepcopy(df)
    
    # Handle date column - convert to column if index
    if 'date' not in df1.columns:
        if isinstance(df1.index, pd.DatetimeIndex):
            df1 = df1.reset_index()
            # Rename the index column to 'date' regardless of its original name
            index_col = df1.columns[0]  # First column after reset_index is the old index
            if index_col != 'date':
                df1.rename(columns={index_col: 'date'}, inplace=True)
        else:
            raise ValueError("DataFrame must have 'date' column or DatetimeIndex")
    
    # Convert 'date' column to datetime format
    df1["date"] = pd.to_datetime(df1["date"])
    df1 = df1[df1["date"].dt.date >= filter_date]
    
    # Validate returns columns
    asset_info = []
    for i, ret_col in enumerate(returns_cols):
        if ret_col not in df1.columns:
            print(f"Warning: Return column '{ret_col}' not found in DataFrame, skipping")
            continue
        
        # Map column names to asset labels
        if 'btc' in ret_col.lower():
            label = 'Bitcoin'
        elif 'gold' in ret_col.lower():
            label = 'Gold'
        elif 'spx_auto' in ret_col.lower():
            label = 'Automobiles'
        elif 'spx_oilgas' in ret_col.lower():
            label = 'Oil & Gas'
        elif 'spx_mach' in ret_col.lower():
            label = 'Machinery'
        elif 'spx_elec' in ret_col.lower():
            label = 'Electronics'
        elif 'spx' in ret_col.lower():
            label = 'S&P 500'
        elif 'aapl' in ret_col.lower():
            label = 'Apple Inc.'
        else:
            label = ret_col
        
        asset_info.append({'col': ret_col, 'label': label})
    
    if not asset_info:
        raise ValueError("No valid return columns found in DataFrame")
    
    plt.rcParams.update({"font.size": 20})
    
    # Create figure with 3 subplots
    n_assets = len(asset_info)
    fig, axes = plt.subplots(n_assets, 1, figsize=figsize, sharex=True)
    
    # Ensure axes is always a list
    if n_assets == 1:
        axes = [axes]
    
    # Define index configurations (same as plot_indices)
    index_configs = {
        'ei': {
            'columns': ['ei_creator', 'ei_community', 'ei_creator_engweighted', 'ei_creator_topicweighted', 'ei_community_topicweighted'],
            'colors': ['#023047', '#219EBC', '#126782', '#023047', '#219EBC'],
            'linestyles': ['-', '-', '-.', '--', '--'],
            'linewidths': [4, 4, 2, 3, 3],
            'labels': ['Creator EI', 'Community EI', 'Creator EI (Eng. Weighted)', 'Creator EI (Topic Weighted)', 'Community EI (Topic Weighted)'],
            'ylim': (-1, 1),
            'ylabel': 'Emotion Index'
        },
        'epi': {
            'columns': ['epi', 'epi_engweighted', 'epi_topicweighted', 'epi_engweighted_topicweighted'],
            'colors': ['#B62B2B', '#D42929', '#F04242', '#FF8787'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['EPI', 'EPI (Eng. Weighted)', 'EPI (Topic Weighted)', 'EPI (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Emotion Polarity Index'
        },
        'epi_signed': {
            'columns': ['epi_signed', 'epi_signed_engweighted', 'epi_signed_topicweighted', 'epi_signed_engweighted_topicweighted'],
            'colors': ['#B62B2B', '#D42929', '#F04242', '#FF8787'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['EPI Signed', 'EPI Signed (Eng. Weighted)', 'EPI Signed (Topic Weighted)', 'EPI Signed (Both Weighted)'],
            'ylim': (-1, 1),
            'ylabel': 'Signed Emotion Polarity Index'
        },
        'ccd': {
            'columns': ['ccd', 'ccd_engweighted', 'ccd_topicweighted', 'ccd_engweighted_topicweighted'],
            'colors': ['#FB8500', '#F4A259', '#FAA307', '#FFD60A'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['CCD', 'CCD (Eng. Weighted)', 'CCD (Topic Weighted)', 'CCD (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Creator-Community Divergence'
        },
        'intensity': {
            'columns': ['intensity', 'intensity_engweighted', 'intensity_topicweighted', 'intensity_engweighted_topicweighted'],
            'colors': ['#126782', '#219EBC', '#126782', '#219EBC'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['Intensity', 'Intensity (Eng. Weighted)', 'Intensity (Topic Weighted)', 'Intensity (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Emotional Intensity'
        },
        'surprise': {
            'columns': ['surprise', 'surprise_engweighted', 'surprise_topicweighted', 'surprise_engweighted_topicweighted'],
            'colors': ['#DAA520', '#FFB703', '#F4A259', '#FF8787'],
            'linestyles': ['-', '-', '--', '--'],
            'linewidths': [4, 3, 2, 2],
            'labels': ['Surprise', 'Surprise (Eng. Weighted)', 'Surprise (Topic Weighted)', 'Surprise (Both Weighted)'],
            'ylim': (0, 1),
            'ylabel': 'Surprise Index (Informational Novelty)'
        },
        'split': {
            'columns': ['split'],
            'colors': ['#8D99AE'],
            'linestyles': ['-'],
            'linewidths': [4],
            'labels': ['Split Index'],
            'ylim': (0, 1),
            'ylabel': 'Split Index (Within-Community Polarization)'
        }
    }
    
    # Get configuration for selected index type
    if index_type not in index_configs:
        raise ValueError(f"Invalid index_type. Choose from: {list(index_configs.keys())}")
    
    config = index_configs[index_type]
    
    # Determine which columns to plot
    if include_variants:
        columns_to_plot = [col for col in include_variants if col in df1.columns]
        # Find corresponding indices for colors, linestyles, etc.
        plot_indices_list = [config['columns'].index(col) if col in config['columns'] else 0 
                             for col in columns_to_plot]
        colors = [config['colors'][i] for i in plot_indices_list]
        linestyles = [config['linestyles'][i] for i in plot_indices_list]
        linewidths = [config['linewidths'][i] for i in plot_indices_list]
        labels = [config['labels'][config['columns'].index(col)] if col in config['columns'] 
                  else col for col in columns_to_plot]
    else:
        columns_to_plot = [col for col in config['columns'] if col in df1.columns]
        colors = config['colors'][:len(columns_to_plot)]
        linestyles = config['linestyles'][:len(columns_to_plot)]
        linewidths = config['linewidths'][:len(columns_to_plot)]
        labels = config['labels'][:len(columns_to_plot)]
    
    # Calculate y-axis limits for indices based on actual data
    data_max = df1[columns_to_plot].max().max()
    data_min = df1[columns_to_plot].min().min()
    
    # Set y-axis limits for indices
    if ylim_indices:
        indices_ylim = ylim_indices
    else:
        # Use data-driven limits with some padding
        y_range = data_max - data_min
        y_padding = y_range * 0.2  # 20% padding
        ul_padding = min(data_max + y_padding, 1.04)
        indices_ylim = (data_min - (y_padding/10), ul_padding)
    
    # Plot each asset in separate subplot
    twin_axes = []  # Store secondary axes for legend collection
    for idx, (ax, asset) in enumerate(zip(axes, asset_info)):
        # Create secondary y-axis for indices
        ax2 = ax.twinx()
        twin_axes.append(ax2)
        
        # Plot returns on primary y-axis
        ax.plot(
            df1["date"],
            df1[asset['col']],
            linestyle='-',
            color='black',
            label='Asset Return',
            linewidth=5,
        )
        
        # Plot indices on secondary y-axis
        for col, color, linestyle, linewidth, label in zip(columns_to_plot, colors, linestyles, linewidths, labels):
            ax2.plot(
                df1["date"],
                df1[col],
                linestyle=linestyle,
                color=color,
                label=label,
                linewidth=linewidth,
            )
        
        # Calculate y-axis limits for returns based on actual data
        ret_data = df1[asset['col']].dropna()
        ret_max = ret_data.max()
        ret_min = ret_data.min()
        ret_range = ret_max - ret_min
        ret_padding = ret_range * 0.2  # 20% padding
        ret_ul_padding = min(ret_max + ret_padding, 1.04)
        ret_ylim = (ret_min - (ret_padding/10), ret_ul_padding)
        
        # Set y-axis limits (swapped - asset on primary, indices on secondary)
        ax.set_ylim(ret_ylim)
        ax2.set_ylim(indices_ylim)
        
        # Set labels (swapped)
        ax.set_ylabel(asset['label'])  # Asset name as primary y-axis label
        # Only show secondary y-axis label on middle subplot
        if idx == 1:
            ax2.set_ylabel(config['ylabel'])
        
        # Add event shading if requested (no text inside plots)
        if add_events:
            # Initial Tariff Announcements (20 Jan - 26 Jan)
            ax.axvspan(
                pd.to_datetime("2025-01-20"),
                pd.to_datetime("2025-01-26"),
                color="gray",
                alpha=0.3,
            )
            
            # Tariffs Postponed (04 Mar - 06 Mar)
            ax.axvspan(
                pd.to_datetime("2025-03-04"),
                pd.to_datetime("2025-03-06"),
                color="gray",
                alpha=0.3,
            )
            
            # China Retaliation & US Tariffs Imposed (10 Mar - 13 Mar)
            ax.axvspan(
                pd.to_datetime("2025-03-10"),
                pd.to_datetime("2025-03-13"),
                color="gray",
                alpha=0.3,
            )
            
            # Global Tariffs & China Retaliation (2 Apr - 6 Apr)
            ax.axvspan(
                pd.to_datetime("2025-04-02"),
                pd.to_datetime("2025-04-06"),
                color="gray",
                alpha=0.3,
            )
            
            # US-China Tariff Threats (9 Apr - 12 Apr)
            ax.axvspan(
                pd.to_datetime("2025-04-09"),
                pd.to_datetime("2025-04-12"),
                color="gray",
                alpha=0.3,
            )
            
            # Tariffs Relaxed (27 Apr - 04 May)
            ax.axvspan(
                pd.to_datetime("2025-04-27"),
                pd.to_datetime("2025-05-04"),
                color="gray",
                alpha=0.3,
            )
            
            # Trade Deals & Tariff Rollbacks (05 May - 14 May)
            ax.axvspan(
                pd.to_datetime("2025-05-05"),
                pd.to_datetime("2025-05-14"),
                color="gray",
                alpha=0.3,
            )
            
            # Non-US Apple Tariff Threat (20 May - 25 May)
            ax.axvspan(
                pd.to_datetime("2025-05-20"),
                pd.to_datetime("2025-05-25"),
                color="gray",
                alpha=0.3,
            )
    
    # Add event text annotations above plots if requested
    if add_events:
        # Get date range for positioning
        date_min = df1["date"].min()
        date_max = df1["date"].max()
        date_range = (date_max - date_min).days
        
        # Helper function to convert date to figure x-coordinate
        def date_to_fig_x(date):
            date_pd = pd.to_datetime(date)
            days_from_start = (date_pd - date_min).days
            # Map to figure coordinates (0.125 to 0.9 typically for plot area)
            return 0.125 + (days_from_start / date_range) * 0.775
        
        # Y position for text (above plots)
        text_y = 0.97
        
        # Add event labels
        fig.text(date_to_fig_x("2025-01-23"), text_y, "Initial Tariff\nAnnouncements", 
                ha="center", va="top", fontsize=18, color="black")
        fig.text(date_to_fig_x("2025-02-24"), text_y, "Tariffs\nPostponed", 
                ha="center", va="top", fontsize=18, color="black")
        fig.text(date_to_fig_x("2025-03-10"), 0.9894, "China\nRetaliation\n& US Tariffs", 
                ha="center", va="top", fontsize=18, color="black")
        fig.text(date_to_fig_x("2025-03-29"), 1.006, "Global\nTariffs &\nChina\nRetaliation", 
                ha="center", va="top", fontsize=18, color="black")
        fig.text(date_to_fig_x("2025-04-13"), text_y, "US-China\nTariff Threats", 
                ha="center", va="top", fontsize=18, color="black")
        fig.text(date_to_fig_x("2025-04-28"), text_y, "Tariffs\nRelaxed", 
                ha="center", va="top", fontsize=18, color="black")
        fig.text(date_to_fig_x("2025-05-10"), 1.006, "Trade\nDeals &\nTariff\nRollbacks", 
                ha="center", va="top", fontsize=18, color="black")
        fig.text(date_to_fig_x("2025-05-23"), text_y, "Apple Tariff\nThreat", 
                ha="center", va="top", fontsize=18, color="black")
    
    # Formatting x-axis (only for bottom subplot)
    week_start_dates = (
        df1["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )
    axes[-1].set_xticks(week_start_dates)
    axes[-1].set_xticklabels(week_start_dates.dt.strftime("%d-%b"))
    
    # Rotate x-axis labels
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    
    # Collect all legend handles and labels
    all_handles = []
    all_labels = []
    
    # Get return handle from first primary axis (only one "Asset Return" label)
    lines1, labels1 = axes[0].get_legend_handles_labels()
    if lines1:
        all_handles.append(lines1[0])
        all_labels.append(labels1[0])
    
    # Get index handles from first secondary axis
    lines2, labels2 = twin_axes[0].get_legend_handles_labels()
    all_handles.extend(lines2)
    all_labels.extend(labels2)
    
    # Add combined legend at bottom
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=min(len(all_labels), 5),
        fontsize=18,
    )
    
    # Adjust layout (more space at top for event labels, normal space at bottom)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Show plot
    plt.show()
