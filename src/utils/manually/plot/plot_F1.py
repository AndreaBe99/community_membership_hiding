from calendar import c
from statistics import mean
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import cv2
import math
import os
import numpy as np


def plot_singleBeta_singleTau_allDataset(
    file_path: str,
    log_name: str,
    algs: List[str],
    detection_alg: str,
    metrics: List[str],
    datasets: List[str],
    beta: float,
    tau: float,
):
    """
    Given a path, loop over all the datasets (given arguments) folders, and
    given the beta and tau values, make a group-plot for each metric, whith
    N groups (with N the number of datasets) and each group contains a M bars
    (with M the number of algorithms).

    JSON file structure:
        - first level: algorithm (e.g. "Agent", "Random", "Degree", "Roam")
        - second level: metric (e.g. "goal", "steps", "reward", "time")

    FOLDER structure:
        - first level: 2 folder, one for each dataset (e.g. "words", "karate", "football")
        - second level: 3 folder, one for each  detection algorithms (greedy, louvain, walktrap)
        - third level: 2 folder, one for node hiding and one for community hiding
        - fourth level: 3 folders, one for each tau value
        - fifth level: 3 folders, one for each beta value
    """
    # Renae the "Agent" key to "DRL-Agent (ours)"
    agent_renamed = "DRL-Agent (ours)"
    centrality_renamed = "Betweenness"

    # Add the metrics that is the ratio between the SR and the NMI
    metrics.append(METRIC_NAME)

    # Save a dictionary with the first level keys is the metric, the second
    # level keys is the dataset, the third level keys is the algorithm
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = {}
        for dataset in datasets:
            metrics_dict[metric][dataset] = {}

    for dataset in datasets:
        # Load the path of the json file: dataset/detection_alg/node_hiding/tau/beta
        if log_name == "evaluation_node_hiding":
            json_path = f"{file_path}/{dataset}/{detection_alg}/node_hiding/tau_{tau}/beta_{beta}/{log_name}.json"
        else:
            json_path = f"{file_path}/{dataset}/{detection_alg}/community_hiding/tau_{tau}/beta_{beta}/{log_name}.json"
        # Load the json file
        with open(json_path, "r") as f:
            data = json.load(f)

        for metric in metrics:
            for alg in algs:
                if metric == METRIC_NAME:
                    # Compute the ratio between the SR and the NMI
                    if log_name == "evaluation_node_hiding":
                        temp_metric = "goal"
                    else:
                        temp_metric = "deception_score"

                    if metric == "SR*NMI":
                        data[alg][metric] = [
                            x * y for x, y in zip(data[alg]["goal"], data[alg]["nmi"])
                        ]
                    elif metric == "SR/(1-NMI)":
                        data[alg][metric] = [
                            x / (1 - y if y != 1 else 0.9999999)
                            for x, y in zip(data[alg]["goal"], data[alg]["nmi"])
                        ]
                    elif metric == "F1":
                        data[alg][metric] = [
                            (2 * x * y) / (x + y)
                            for x, y in zip(data[alg][temp_metric], data[alg]["nmi"])
                        ]

                if alg == "Agent":
                    metrics_dict[metric][dataset][agent_renamed] = data[alg][metric]
                elif alg == "Centrality":
                    metrics_dict[metric][dataset][centrality_renamed] = data[alg][
                        metric
                    ]
                else:
                    metrics_dict[metric][dataset][alg] = data[alg][metric]

                if log_name != "evaluation_node_hiding" and alg == "Agent":
                    # Agent renamed has 300 values, the others 3, so we need to
                    # compute the mean of the first 100 values, the second 100 values,
                    # and the third 100 values
                    # Get the first 100 values, and compute the mean
                    first_100 = metrics_dict[metric][dataset][agent_renamed][:100]
                    first_100_mean = mean(first_100)
                    # Get the second 100 values, and compute the mean
                    second_100 = metrics_dict[metric][dataset][agent_renamed][100:200]
                    second_100_mean = mean(second_100)
                    # Get the third 100 values, and compute the mean
                    third_100 = metrics_dict[metric][dataset][agent_renamed][200:]
                    third_100_mean = mean(third_100)

                    # Replace the values with the mean
                    metrics_dict[metric][dataset][agent_renamed] = [
                        first_100_mean,
                        second_100_mean,
                        third_100_mean,
                    ]

    # Delete all metric from metrics, except "ratio"
    metrics = [METRIC_NAME]
    for metric in metrics:
        plot_data = []
        for dataset in datasets:
            # Create a dataframe from dict_metrics
            df = pd.DataFrame(metrics_dict[metric][dataset])
            # Convert the column "goal" to percentages for each algorithm
            if metric == "goal":
                df = df.apply(lambda x: x * 100)
            # Rename the columns called "Agent" to "DRL-Agent (ours)"
            df = df.rename(columns={"Agent": agent_renamed})
            # Rename the column called "Centrality" to "Betweenness"
            df = df.rename(columns={"Centrality": "Betweenness"})
            # Add the dataframe to the plot_data
            plot_data.append(df)

        # Concatenate the dataframes
        df = pd.concat(plot_data, axis=1)
        # in algs list replace "Agent" with "DRL-Agent (ours)"
        algs = [agent_renamed if alg == "Agent" else alg for alg in algs]
        # in algs list replace "Centrality" with "Betweenness"
        algs = [centrality_renamed if alg == "Centrality" else alg for alg in algs]
        df.columns = pd.MultiIndex.from_product([datasets, algs])
        # Melt the dataframe
        df = df.melt(var_name=["Dataset", "Algorithm"], value_name=metric)

        # Set theme
        sns.set_theme(style="darkgrid")
        # Increase the font size
        sns.set(font_scale=1.6)
        # Set palette
        if log_name == "evaluation_node_hiding":
            palette = sns.set_palette("Set1")
        elif log_name == "evaluation_community_hiding":
            palette = sns.set_palette("Set2")

        # if the metric is goal don't plot the error bars
        if metric == "nmi" or metric == "deception_score":
            errorbar = "sd"
        else:
            errorbar = None

        if metric == "goal" or metric == METRIC_NAME:
            g = sns.catplot(
                data=df,
                kind="bar",
                x="Dataset",
                y=metric,
                hue="Algorithm",
                aspect=2,
                palette=palette,
                errorbar="ci",
                # errorbar=df_confidence_binary_test,
            )

        else:
            # Plot the data
            g = sns.catplot(
                data=df,
                kind="bar",
                x="Dataset",
                y=metric,
                hue="Algorithm",
                aspect=2,
                palette=palette,
                errorbar=errorbar,
            )
        # Set labels as Betas and Metrics
        g.set_axis_labels("Datasets", f"Mean {metric.capitalize()}")
        # if the metric is goal set the y axis to percentages
        if metric == "goal":
            metric = "sr"
            g.set(ylim=(0, 100))
            g.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
            g.set_ylabels("Success Rate")
        elif metric == METRIC_NAME:
            # g.set(ylim=(0, 1))
            if log_name == "evaluation_node_hiding":
                g.set_ylabels(METRIC_NAME)
            elif log_name == "evaluation_community_hiding":
                g.set_ylabels(METRIC_NAME)
        elif metric == "nmi":
            g.set(ylim=(0, 1))
            g.set_ylabels("NMI (avg)")
        elif metric == "deception_score":
            metric = "ds"
            g.set(ylim=(0, 1))
            g.set_ylabels("Deception Score (avg)")
        elif metric == "steps":
            g.set_ylabels("Steps (avg)")
        elif metric == "time":
            g.set_ylabels("Time in sec. (avg)")

        sns.move_legend(g, "upper left", bbox_to_anchor=(0.75, 0.8), frameon=False)

        # Change the text of the first field of the legend
        # replace labels
        for i, t in enumerate(g._legend.texts):
            if i == 0:
                t.set_text("DRl-Agent\n(ours)")
            t.set_fontsize(15)

        g.set_xticklabels(rotation=45, ha="center")

        # Save the plot
        g.savefig(
            f"{file_path}/allDataset_{log_name}_{metric.replace('/', '-')}_tau{tau}_beta{beta}_group.png",
            bbox_inches="tight",
            dpi=300,
        )


def df_confidence_binary_test(x: pd.DataFrame):
    x = x.apply(lambda x: 1 if x == 100 else 0)
    n = x.shape[0]
    p = sum(x.tolist()) / n
    z = 1.96  # 95% confidence level

    std_error = math.sqrt(p * (1 - p) / n)
    margin_of_error = z * std_error

    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error

    lower_bound *= 100
    upper_bound *= 100
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

    return (lower_bound, upper_bound)


def confidence_binary_test(x: List[int]):
    n = len(x)
    p = sum(x) / n
    z = 1.96  # 95% confidence level
    std_error = math.sqrt(p * (1 - p) / n)
    margin_of_error = z * std_error

    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error

    # lower_bound *= 100
    # upper_bound *= 100
    return margin_of_error  # , (lower_bound, upper_bound)


if __name__ == "__main__":

    ################# SINGLE BETA - SINGLE TAU - ALL DATASET #################
    DETECTION_ALG = "walktrap"
    PATH = "test"
    TYPE = 1  # 0: allBeta, 1: allDataset
    BETA = 1
    TAU = 0.5

    metrics_test = ["F1", "SR*NMI", "SR/(1-NMI)"]
    METRIC_NAME = metrics_test[0]
    # NODE HIDING
    plot_singleBeta_singleTau_allDataset(
        PATH,
        log_name="evaluation_node_hiding",
        algs=["Agent", "Random", "Degree", "Centrality", "Roam"],
        detection_alg=DETECTION_ALG,
        metrics=["goal", "nmi", "steps", "time"],
        datasets=["kar", "words", "vote", "pow", "fb-75"],
        beta=BETA,
        tau=TAU,
    )
    # join_images(PATH, task="node_hiding", nd_box_start_r=1.58, beta=BETA, tau=TAU)
    # COMMUNITY HIDING
    BETA = 1
    TAU = 0.3
    plot_singleBeta_singleTau_allDataset(
        PATH,
        log_name="evaluation_community_hiding",
        algs=["Agent", "Safeness", "Modularity"],
        detection_alg=DETECTION_ALG,
        metrics=["goal", "nmi", "deception_score", "steps", "time"],
        datasets=["kar", "words", "vote", "pow", "fb-75"],
        beta=BETA,
        tau=TAU,
    )
