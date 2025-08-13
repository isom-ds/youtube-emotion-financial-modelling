import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from constants import topic_list

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

# Define topic groups and their colors
topiccols = [
    "date",
    "damages",
    "hurricane advice",
    "hurricane relief services",
    "personal opinion",
    "weather information",
]
topiccolours = {
    "damages": "#B62B2B",
    "hurricane advice": "#fb8500",
    "hurricane relief services": "#219ebc",
    "personal opinion": "#ffb703",
    "weather information": "#023047",
    #'other': '#6c757d'
}


def plot_stack100(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    data3: pd.DataFrame,
    data4: pd.DataFrame,
    datatype: str,
    legend_cols: int = 8,
    legend_add_gap: float = 0.0,
):
    if datatype == "emotions":
        columns = emocols
        colours = emocolours

    if datatype == "topics":
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
    df1.set_index("date", inplace=True)
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")
    df2.set_index("date", inplace=True)
    df3["date"] = pd.to_datetime(df3["date"], format="%Y-%m-%d")
    df3.set_index("date", inplace=True)
    df4["date"] = pd.to_datetime(df4["date"], format="%Y-%m-%d")
    df4.set_index("date", inplace=True)

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
        0.305, 0.97, "Hurricane Helene", ha="center", fontsize=20, fontweight="bold"
    )
    fig.text(
        0.715, 0.97, "Hurricane Milton", ha="center", fontsize=20, fontweight="bold"
    )

    # Add a black box around hurricane periods
    hurricane_periods = [
        ("2024-09-24", "2024-09-27", "Hurricane Helene"),
        ("2024-10-05", "2024-10-10", "Hurricane Milton"),
    ]

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


def epi_plot(
    helene: pd.DataFrame,
    milton: pd.DataFrame,
):

    df1 = deepcopy(helene)
    df2 = deepcopy(milton)

    # Convert 'date' column to datetime format
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")

    plt.rcParams.update({"font.size": 20})

    # Create figure with two subplots (side by side)
    fig, (ax1, ax3) = plt.subplots(
        1, 2, figsize=(20, 5), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    # --- First Graph: Milton ---
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
    ax1.set_title("Hurricane Helene")
    ax1.set_ylim(-1, 1)

    # --- Second Graph: AEPI ---
    ax3.plot(
        df2["date"],
        df2["ei_creator"],
        linestyle="--",
        color="b",
        label="Creator EI",
        linewidth=2,
    )
    ax3.plot(
        df2["date"],
        df2["ei_community"],
        linestyle="--",
        color="g",
        label="Community EI",
        linewidth=2,
    )
    ax3.plot(
        df2["date"], df2["epi"], linestyle="-", color="r", label="EPI", linewidth=4
    )
    ax3.set_title("Hurricane Milton")
    ax3.set_ylim(-1, 1)

    # Highlight Hurricane Helene (24 Sep - 27 Sep)
    ax1.axvspan(
        pd.to_datetime("2024-09-24"),
        pd.to_datetime("2024-09-27"),
        color="gray",
        alpha=0.3,
    )
    ax3.axvspan(
        pd.to_datetime("2024-09-24"),
        pd.to_datetime("2024-09-27"),
        color="gray",
        alpha=0.3,
    )
    ax1.text(pd.to_datetime("2024-09-23"), -0.95, "Helene", color="black", ha="left")
    ax3.text(pd.to_datetime("2024-09-24"), -0.95, "Helene", color="black", ha="left")

    # Highlight Hurricane Milton (5 Oct - 10 Oct)
    ax1.axvspan(
        pd.to_datetime("2024-10-05"),
        pd.to_datetime("2024-10-10"),
        color="gray",
        alpha=0.3,
    )
    ax3.axvspan(
        pd.to_datetime("2024-10-05"),
        pd.to_datetime("2024-10-10"),
        color="gray",
        alpha=0.3,
    )
    ax1.text(pd.to_datetime("2024-10-07"), -0.95, "Milton", color="black", ha="center")
    ax3.text(pd.to_datetime("2024-10-06"), -0.95, "Milton", color="black", ha="left")

    # Formatting
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

    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Show plot
    plt.show()


def plot_ei_topic(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    data3: pd.DataFrame,
    data4: pd.DataFrame,
    legend_cols: int = 8,
    legend_add_gap: float = 0.0,
):

    df1 = deepcopy(data1)
    df2 = deepcopy(data2)
    df3 = deepcopy(data3)
    df4 = deepcopy(data4)
    df1 = df1[["date"] + [f"ei_creator_{i}" for i in topic_list if i != "other"]]
    df2 = df2[["date"] + [f"ei_community_{i}" for i in topic_list if i != "other"]]
    df3 = df3[["date"] + [f"ei_creator_{i}" for i in topic_list if i != "other"]]
    df4 = df4[["date"] + [f"ei_community_{i}" for i in topic_list if i != "other"]]

    # Set 'date' as index
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")
    df3["date"] = pd.to_datetime(df3["date"], format="%Y-%m-%d")
    df4["date"] = pd.to_datetime(df4["date"], format="%Y-%m-%d")

    plt.rcParams.update({"font.size": 18})

    # Create figure with two subplots (side by side)
    fig, (ax1, ax3, ax5, ax7) = plt.subplots(
        1, 4, figsize=(20, 4), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    # Plot
    for i in topic_list:
        if i != "other":
            ax1.plot(
                df1["date"],
                df1[f"ei_creator_{i}"],
                color=topiccolours[i],
                label=i,
                linewidth=2,
            )
            ax3.plot(
                df2["date"],
                df2[f"ei_community_{i}"],
                color=topiccolours[i],
                label=i,
                linewidth=2,
            )
            ax5.plot(
                df3["date"],
                df3[f"ei_creator_{i}"],
                color=topiccolours[i],
                label=i,
                linewidth=2,
            )
            ax7.plot(
                df4["date"],
                df4[f"ei_community_{i}"],
                color=topiccolours[i],
                label=i,
                linewidth=2,
            )

    # Formatting
    ax1.set_xlabel("")
    ax3.set_xlabel("")
    ax5.set_xlabel("")
    ax7.set_xlabel("")
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
    week_start_dates = (
        df3["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )  # Get start of each week
    ax5.set_xticks(week_start_dates)  # Set ticks at these dates
    ax5.set_xticklabels(
        week_start_dates.dt.strftime("%d-%b")
    )  # Format labels correctly
    week_start_dates = (
        df4["date"].dt.to_period("W").drop_duplicates().dt.start_time
    )  # Get start of each week
    ax7.set_xticks(week_start_dates)  # Set ticks at these dates
    ax7.set_xticklabels(
        week_start_dates.dt.strftime("%d-%b")
    )  # Format labels correctly

    ax1.set_title("Content Creator")
    ax3.set_title("Community")
    ax5.set_title("Content Creator")
    ax7.set_title("Community")

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


def plot_epi_topic_weighted(
    helene: pd.DataFrame,
    milton: pd.DataFrame,
):

    df1 = deepcopy(helene)
    df2 = deepcopy(milton)

    # Convert 'date' column to datetime format
    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    df2["date"] = pd.to_datetime(df2["date"], format="%Y-%m-%d")

    plt.rcParams.update({"font.size": 20})

    # Create figure with two subplots (side by side)
    fig, (ax1, ax3) = plt.subplots(
        1, 2, figsize=(20, 5), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    # --- First Graph: Milton ---
    ax1.plot(df1["date"], df1["epi_weighted"], linewidth=2)
    ax1.set_title("Hurricane Helene")
    ax1.set_ylim(-1, 1)

    # --- Second Graph: AEPI ---
    ax3.plot(df2["date"], df2["epi_weighted"], linewidth=2)
    ax3.set_title("Hurricane Milton")
    ax3.set_ylim(-1, 1)

    # Highlight Hurricane Helene (24 Sep - 27 Sep)
    ax1.axvspan(
        pd.to_datetime("2024-09-24"),
        pd.to_datetime("2024-09-27"),
        color="gray",
        alpha=0.3,
    )
    ax3.axvspan(
        pd.to_datetime("2024-09-24"),
        pd.to_datetime("2024-09-27"),
        color="gray",
        alpha=0.3,
    )
    ax1.text(pd.to_datetime("2024-09-23"), -0.95, "Helene", color="black", ha="left")
    ax3.text(pd.to_datetime("2024-09-24"), -0.95, "Helene", color="black", ha="left")

    # Highlight Hurricane Milton (5 Oct - 10 Oct)
    ax1.axvspan(
        pd.to_datetime("2024-10-05"),
        pd.to_datetime("2024-10-10"),
        color="gray",
        alpha=0.3,
    )
    ax3.axvspan(
        pd.to_datetime("2024-10-05"),
        pd.to_datetime("2024-10-10"),
        color="gray",
        alpha=0.3,
    )
    ax1.text(pd.to_datetime("2024-10-07"), -0.95, "Milton", color="black", ha="center")
    ax3.text(pd.to_datetime("2024-10-06"), -0.95, "Milton", color="black", ha="left")

    # Formatting
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

    plt.xticks(rotation=360)  # Rotate x-ticks for better readability

    # add horizontal line at y=0
    ax1.axhline(0, color="black", linewidth=2, linestyle="--")
    ax3.axhline(0, color="black", linewidth=2, linestyle="--")

    # Remove x-axis label
    ax1.set_ylabel("Index")

    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Show plot
    plt.show()
