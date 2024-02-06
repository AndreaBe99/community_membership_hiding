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
    # Replace the "Agent" key with "DRL-Agent (ours)"
    algs[algs.index("Agent")] = agent_renamed

    df_sr = pd.DataFrame(columns=["algorithms", "mean", "std", "dataset"])
    df_nmi = pd.DataFrame(columns=["algorithms", "mean", "std", "dataset"])
    # Loop over the datasets
    for dataset in datasets:
        path = f"{file_path}/{dataset}/{detection_alg}/{log_name}/tau_{tau}/allBetas_evaluation_{log_name}_mean_std.json"

        # Load JSON data from file
        with open(path, "r") as f:
            json_data = json.load(f)

        # Get JSON data given a beta value
        json_sr = json_data["goal"][str(beta)]
        json_nmi = json_data["nmi"][str(beta)]

        # Add a new key to the dictionary representing the dataset
        # json_sr[dataset] = dataset

        # Create a dataframe from the JSON data, with the following columns:
        # - algorithms (e.g. "Agent", "Random", "Degree", "Roam")
        # - mean
        # - std
        # - dataset (e.g. "words", "karate", ...)

        for algorithm in json_sr.keys():
            if algorithm == "Agent":
                algorithm = agent_renamed

            df_sr = df_sr._append(
                {
                    "dataset": dataset,
                    "algorithms": algorithm,
                    "mean": json_sr[algorithm]["mean"],
                    "std": json_sr[algorithm]["ci"],
                },
                ignore_index=True,
            )
            # Do the same for NMI
            df_nmi = df_nmi._append(
                {
                    "dataset": dataset,
                    "algorithms": algorithm,
                    "mean": json_nmi[algorithm]["mean"],
                    "std": json_nmi[algorithm]["std"],
                },
                ignore_index=True,
            )

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Set font scale
    sns.set(font_scale=1.5)

    palette = sns.color_palette("Set1", n_colors=4)
    # Get list of colors from the palette
    colors = palette.as_hex()

    # Creare un dizionario per mappare gli algoritmi ai colori
    color_dict = {
        "DRL-Agent (ours)": colors[0],
        "Random": colors[1],
        "Degree": colors[2],
        "Roam": colors[3],
    }
    # Creare un dizionario per mappare i dataset ai marker
    marker_dict = {"kar": "o", "words": "^", "vote": "s", "pow": "D", "fb-75": "X"}

    # Creare un dizionario per memorizzare i punti per ogni algoritmo
    points_dict = {"DRL-Agent (ours)": [], "Random": [], "Degree": [], "Roam": []}

    # Iterare su ogni riga nei dataframe
    for (index_sr, row_sr), (index_nmi, row_nmi) in zip(
        df_sr.iterrows(), df_nmi.iterrows()
    ):
        # Ottenere il colore e il marker corrispondenti
        color = color_dict[row_sr["algorithms"]]
        marker = marker_dict[row_sr["dataset"]]

        # Aggiungere un punto al grafico
        plt.scatter(row_nmi["mean"], row_sr["mean"], color=color, marker=marker, s=100)

        # Aggiungere il punto al dizionario
        points_dict[row_sr["algorithms"]].append((row_nmi["mean"], row_sr["mean"]))

    # Iterare su ogni algoritmo
    for algorithm, points in points_dict.items():
        # Ordinare i punti in base al valore NMI
        points.sort()

        # Ottenere i valori NMI e SR separati
        nmi_values, sr_values = zip(*points)

        # Collegare i punti con una linea
        plt.plot(nmi_values, sr_values, color=color_dict[algorithm])

    # Impostare i titoli degli assi
    # Convert y-axis values to percentage
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:.0f}%".format(x * 100))
    )

    plt.xlabel("NMI")
    plt.ylabel("SR")

    # Creare una lista di oggetti "Patch" per ogni algoritmo
    algorithm_patches = [
        plt.Line2D([0], [0], color=color, lw=2, label=algorithm)
        for algorithm, color in color_dict.items()
    ]

    # Creare una lista di oggetti "Patch" per ogni dataset
    dataset_patches = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor="k",
            markersize=10,
            label=dataset,
        )
        for dataset, marker in marker_dict.items()
    ]

    # Creare la legenda per gli algoritmi
    lgnd = plt.legend(
        loc="center left",
        handles=algorithm_patches,
        title="Algorithms",
        bbox_to_anchor=(1.02, 0.8),  # Move the legend outside the plot
        facecolor='white'  # Set the background color of the legend to white
        
    )
    plt.gca().add_artist(lgnd)
    # Creare la legenda per i dataset
    
    plt.legend(
        loc="center left",
        handles=dataset_patches,
        title="Datasets",
        bbox_to_anchor=(1.02, 0.3),  # Move the legend outside the plot
        facecolor='white'  # Set the background color of the legend to white
    )
    
    # aggiungi più spazio a destra per la legenda
    plt.subplots_adjust(right=0.69)

    #plt.tight_layout()

    # Mostrare il grafico
    # plt.show()
    plt.savefig(f"{file_path}/allDatasets_tau_{tau}_beta_{beta}_sr_nmi_lineplot.png")


if __name__ == "__main__":
    ################# SINGLE BETA - SINGLE TAU - ALL DATASET #################
    DETECTION_ALG = "greedy"
    PATH = "test"
    TYPE = 1  # 0: allBeta, 1: allDataset
    BETA = 1
    TAU = 0.3
    # NODE HIDING
    plot_singleBeta_singleTau_allDataset(
        PATH,
        log_name="node_hiding",
        algs=["Agent", "Random", "Degree", "Roam"],
        detection_alg=DETECTION_ALG,
        metrics=["goal", "nmi", "steps", "time"],
        datasets=["kar", "words", "vote", "pow", "fb-75"],
        beta=BETA,
        tau=TAU,
    )

    # COMMUNITY HIDING
    # BETA = 1
    # TAU = 0.3
    # plot_singleBeta_singleTau_allDataset(
    #     PATH,
    #     log_name="evaluation_community_hiding",
    #     algs=["Agent", "Safeness", "Modularity"],
    #     detection_alg=DETECTION_ALG,
    #     metrics=["goal", "nmi", "deception_score", "steps", "time"],
    #     datasets=["kar", "words", "vote", "pow", "fb-75"],
    #     beta=BETA,
    #     tau=TAU,
    # )
