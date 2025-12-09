import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from copy import deepcopy

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
    "surprise": "#58B4D1",
    "anger": "#B62B2B",
    "fear": "#D42929",
    "disgust": "#F04242",
    "sadness": "#FF8787",
}


def get_topic_columns(df: pd.DataFrame) -> list:
    """Extract topic columns from dataframe (all columns except 'date')."""
    return ["date"] + [col for col in df.columns if col != "date"]


def get_topic_colours(topics: list, cmap_name: str = "tab20") -> dict:
    """Generate a colour mapping for topics dynamically using a matplotlib colormap.
    
    Args:
        topics: List of topic names (may include 'date' which will be excluded)
        cmap_name: Name of matplotlib colormap. Good options for many categories:
            - 'tab20' (20 distinct colors)
            - 'tab20b', 'tab20c' (20 more distinct colors)
            - 'Set1', 'Set2', 'Set3' (9-12 colors)
            - 'Paired' (12 colors)
    """
    # Exclude 'date' from topics if present
    topic_names = [t for t in topics if t != "date"]
    n_topics = len(topic_names)
    
    # Get colormap and generate evenly spaced colors
    cmap = cm.get_cmap(cmap_name, max(n_topics, 1))
    colors = [cmap(i) for i in range(n_topics)]
    
    # Convert RGBA to hex
    return {
        topic: "#{:02x}{:02x}{:02x}".format(
            int(colors[i][0] * 255),
            int(colors[i][1] * 255),
            int(colors[i][2] * 255)
        )
        for i, topic in enumerate(topic_names)
    }


def plot_stack100(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    datatype: str,
    titlestr: str,
    legend_cols: int = 8,
    legend_add_gap: float = 0.0,
    topic_colours: dict | None = None,
):
    if datatype == "emotion":
        columns = emocols
        colours = emocolours

    if datatype == "topic":
        columns = get_topic_columns(data1)
        colours = topic_colours if topic_colours else get_topic_colours(columns)

    df1 = deepcopy(data1)
    df2 = deepcopy(data2)
    df1 = df1[columns]
    df2 = df2[columns]

    # Set 'date' as index
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df1.set_index("date", inplace=True)
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")
    df2.set_index("date", inplace=True)

    plt.rcParams.update({"font.size": 18})

    # Create figure with two subplots (side by side)
    fig, (ax1, ax3) = plt.subplots(
        1, 2, figsize=(20, 4), sharey=True, gridspec_kw={"wspace": 0.05}
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

    # Turn off legends for all ax
    ax1.get_legend().remove()
    ax3.get_legend().remove()

    # Add shared titles for groups
    fig.text(
        0.5, 0.97, titlestr, ha="center", fontsize=20, fontweight="bold"
    )

    # Add a black box around hurricane periods
    hurricane_periods = [
        ("2024-09-24", "2024-09-27", "Hurricane Helene"),
        ("2024-10-05", "2024-10-10", "Hurricane Milton"),
    ]

    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Move legend outside
    plt.legend(
        title="",
        bbox_to_anchor=(-0.05, -0.4 + legend_add_gap),
        ncol=legend_cols,
        loc="center",
        fontsize="18",
    )

    plt.tight_layout()

    # Show plot
    plt.show()


def epi_plot(
    df: pd.DataFrame,
):

    df1 = deepcopy(df)

    # Convert 'date' column to datetime format
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")

    plt.rcParams.update({"font.size": 20})

    # Create figure with two subplots (side by side)
    fig, ax1 = plt.subplots(
        1, 1, figsize=(20, 10), sharey=True
    )

    # --- Graph ---
    ax1.plot(
        df1["date"],
        df1["ei_creator"],
        linestyle="--",
        color="b",
        label="Creator EI",
        linewidth=2,
    )
    ax1.plot(
        df1["date"],
        df1["ei_community"],
        linestyle="--",
        color="g",
        label="Community EI",
        linewidth=2,
    )
    ax1.plot(
        df1["date"], df1["epi"], linestyle="-", color="r", label="EPI", linewidth=4
    )
    # ax1.set_title("Hurricane Helene")
    ax1.set_ylim(-1, 1)

    # Highlight Hurricane Helene (24 Sep - 27 Sep)
    # ax1.axvspan(
    #     pd.to_datetime("2024-09-24"),
    #     pd.to_datetime("2024-09-27"),
    #     color="gray",
    #     alpha=0.3,
    # )
    # ax1.text(pd.to_datetime("2024-09-23"), -0.95, "Helene", color="black", ha="left")

    # # Highlight Hurricane Milton (5 Oct - 10 Oct)
    # ax1.axvspan(
    #     pd.to_datetime("2024-10-05"),
    #     pd.to_datetime("2024-10-10"),
    #     color="gray",
    #     alpha=0.3,
    # )
    # ax1.text(pd.to_datetime("2024-10-07"), -0.95, "Milton", color="black", ha="center")

    # Formatting
    week_start_dates = (
        df1["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )  # Get start of each week
    ax1.set_xticks(week_start_dates)  # Set ticks at these dates
    ax1.set_xticklabels(
        week_start_dates.dt.strftime("%d-%b")
    )  # Format labels correctly
    plt.xticks(rotation=360)  # Rotate x-ticks for better readability

    # Remove x-axis label
    ax1.set_ylabel("Index")

    # Combine legends from both graphs and place at the bottom outside the plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(
        lines1,
        labels1,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        ncol=1,
        fontsize=20,
    )

    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Show plot
    plt.show()


def plot_ei_topic(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    topic_list: list,
    legend_cols: int = 8,
    legend_add_gap: float = 0.0,
    topic_colours: dict | None = None,
):

    df1 = deepcopy(data1)
    df2 = deepcopy(data2)
    df1 = df1[["date"] + [f"ei_creator_{i}" for i in topic_list if i != "other"]]
    df2 = df2[["date"] + [f"ei_community_{i}" for i in topic_list if i != "other"]]

    # Set 'date' as index
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")

    # Generate dynamic topic colours if not provided
    colours = topic_colours if topic_colours else get_topic_colours(topic_list)

    plt.rcParams.update({"font.size": 18})

    # Create figure with two subplots (side by side)
    fig, (ax1, ax3) = plt.subplots(
        1, 2, figsize=(20, 4), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    # Plot
    for i in topic_list:
        if i != "other":
            ax1.plot(
                df1["date"],
                df1[f"ei_creator_{i}"],
                color=colours[i],
                label=i,
                linewidth=2,
            )
            ax3.plot(
                df2["date"],
                df2[f"ei_community_{i}"],
                color=colours[i],
                label=i,
                linewidth=2,
            )

    # Formatting
    ax1.set_xlabel("")
    ax3.set_xlabel("")
    ax1.set_ylabel("Index")

    # Format x-axis dates as 'DD-Mon'
    week_start_dates = (
        df1["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )  # Get start of each week
    ax1.set_xticks(week_start_dates)  # Set ticks at these dates
    ax1.set_xticklabels(
        week_start_dates.dt.strftime("%d-%b")
    )  # Format labels correctly
    week_start_dates = (
        df2["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )  # Get start of each week
    ax3.set_xticks(week_start_dates)  # Set ticks at these dates
    ax3.set_xticklabels(
        week_start_dates.dt.strftime("%d-%b")
    )  # Format labels correctly

    ax1.set_title("Content Creator")
    ax3.set_title("Community")

    # # Turn off legends for all ax
    # ax1.get_legend().remove()
    # ax3.get_legend().remove()
    # ax5.get_legend().remove()
    # ax7.get_legend().remove()

    # Add shared titles for groups
    fig.text(
        0.305, 0.97, "Hurricane Helene", ha="center", fontsize=20, fontweight="bold"
    )
    fig.text(
        0.715, 0.97, "Hurricane Milton", ha="center", fontsize=20, fontweight="bold"
    )

    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

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


def plot_epi_topic_weighted(
    df: pd.DataFrame,
):

    df1 = deepcopy(df)

    # Convert 'date' column to datetime format
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")

    plt.rcParams.update({"font.size": 20})

    # Create figure with a single subplot
    fig, ax1 = plt.subplots(
        1, 1, figsize=(20, 5), sharey=True
    )

    # --- Graph ---
    ax1.plot(df1["date"], df1["epi_weighted"], linewidth=2)
    ax1.set_title("Hurricane Helene")
    ax1.set_ylim(-1, 1)

    # Highlight Hurricane Helene (24 Sep - 27 Sep)
    ax1.axvspan(
        pd.to_datetime("2024-09-24"),
        pd.to_datetime("2024-09-27"),
        color="gray",
        alpha=0.3,
    )
    ax1.text(pd.to_datetime("2024-09-23"), -0.95, "Helene", color="black", ha="left")

    # Highlight Hurricane Milton (5 Oct - 10 Oct)
    ax1.axvspan(
        pd.to_datetime("2024-10-05"),
        pd.to_datetime("2024-10-10"),
        color="gray",
        alpha=0.3,
    )
    ax1.text(pd.to_datetime("2024-10-07"), -0.95, "Milton", color="black", ha="center")

    # Formatting
    week_start_dates = (
        df1["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )  # Get start of each week
    ax1.set_xticks(week_start_dates)  # Set ticks at these dates
    ax1.set_xticklabels(
        week_start_dates.dt.strftime("%d-%b")
    )

    plt.xticks(rotation=360)  # Rotate x-ticks for better readability

    # add horizontal line at y=0
    ax1.axhline(0, color="black", linewidth=2, linestyle="--")

    # Remove x-axis label
    ax1.set_ylabel("Index")

    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Show plot
    plt.show()
