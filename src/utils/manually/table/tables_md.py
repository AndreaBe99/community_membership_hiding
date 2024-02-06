import json
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


def process_json_community(data, tau):
    data = json.loads(data)
    metrics = ["deception_score", "nmi"]

    taus = [0.3, 0.5, 0.8]

    results_sr = {
        "τ": [],
        "β": [],
        "DRL-Agent (ours)": [],
        "Safeness": [],
        "Modularity": [],
    }

    data_1 = data[metrics[0]]
    for beta in data_1:
        for alg in data_1[beta]:
            mean = data_1[beta][alg]["mean"]
            std = data_1[beta][alg]["std"]
            results_sr[alg].append(f"{mean:.2f} ± {std:.2f}")
        results_sr["τ"].append(tau)
        results_sr["β"].append(beta)

    df_sr = pd.DataFrame(results_sr)

    results_nmi = {
        "τ": [],
        "β": [],
        "DRL-Agent (ours)": [],
        "Safeness": [],
        "Modularity": [],
    }

    data_2 = data[metrics[1]]
    for beta in data_2:
        for alg in data_2[beta]:
            mean = data_2[beta][alg]["mean"]
            std = data_2[beta][alg]["std"]
            results_nmi[alg].append(f"{mean:.2f} ± {std:.2f}")
        results_nmi["τ"].append(tau)
        results_nmi["β"].append(beta)
    df_nmi = pd.DataFrame(results_nmi)

    return df_sr, df_nmi


def process_json(data, tau):
    data = json.loads(data)
    metrics = ["goal", "nmi"]

    results_sr = {
        "τ": [],
        "β": [],
        "DRL-Agent (ours)": [],
        "Random": [],
        "Degree": [],
        "Centrality": [],
        "Roam": [],
    }

    data_1 = data[metrics[0]]
    for beta in data_1:
        for alg in data_1[beta]:
            mean = data_1[beta][alg]["mean"] * 100
            std = data_1[beta][alg]["ci"] * 100
            results_sr[alg].append(f"{mean:.2f}% ± {std:.2f}%")
        results_sr["τ"].append(tau)
        results_sr["β"].append(str(beta) + " $\mu$")

    df_sr = pd.DataFrame(results_sr)

    results_nmi = {
        "τ": [],
        "β": [],
        "DRL-Agent (ours)": [],
        "Random": [],
        "Degree": [],
        "Centrality": [],
        "Roam": [],
    }

    data_2 = data[metrics[1]]
    for beta in data_2:
        for alg in data_2[beta]:
            mean = data_2[beta][alg]["mean"]
            std = data_2[beta][alg]["std"]
            results_nmi[alg].append(f"{mean:.2f} ± {std:.2f}")
        results_nmi["τ"].append(tau)
        results_nmi["β"].append(str(beta) + " $\mu$")

    df_nmi = pd.DataFrame(results_nmi)

    # Make a copy of the dataframe, where the column of each algorithm converted
    # to float values, considering the mean value only
    df_sr_float = df_sr.copy()
    for alg in df_sr_float.columns[2:]:
        df_sr_float[alg] = df_sr_float[alg].apply(lambda x: float(x.split("%")[0]))

    # Do the same for the NMI dataframe
    df_nmi_float = df_nmi.copy()
    for alg in df_nmi_float.columns[2:]:
        df_nmi_float[alg] = df_nmi_float[alg].apply(lambda x: float(x.split("±")[0]))

    return df_sr, df_nmi, df_sr_float, df_nmi_float


def save_lineplot(df_sr, df_nmi):
    # Fissare un valore di tau (ad esempio, tau=0.3)
    tau_value = 0.5
    df_sr_tau = df_sr[df_sr["τ"] == tau_value]
    df_nmi_tau = df_nmi[df_nmi["τ"] == tau_value]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    palette = sns.color_palette("Set1", n_colors=4)
    # Get list of colors from the palette
    colors = palette.as_hex()

    # colors = ["red", "green", "blue", "orange"]
    markers = ["o", "v", "s"]
    j = 0
    for algorithm in df_sr_tau.columns[2:]:
        for i in range(len(df_sr_tau[algorithm])):
            plt.plot(
                df_nmi_tau[algorithm].iloc[i],
                df_sr_tau[algorithm].iloc[i],
                marker=markers[i],
                color=colors[j],
                markersize=15,
            )

        # Connect the markers with a line
        plt.plot(
            df_nmi_tau[algorithm],
            df_sr_tau[algorithm],
            color=colors[j],
            linewidth=2,
        )

        j = j + 1

    plt.title(f"SR vs NMI for τ = {tau_value}")
    plt.xlabel("NMI")
    plt.ylabel("SR")

    # Add another legend for the markers, where:
    # - "o" is for β = 0.5 μ
    # - "v" is for β = 1 μ
    # - "s" is for β = 2 μ
    lgnd = plt.legend(
        loc="center left",
        handles=[
            plt.Line2D([], [], color="black", marker="o", linestyle="None"),
            plt.Line2D([], [], color="black", marker="v", linestyle="None"),
            plt.Line2D([], [], color="black", marker="s", linestyle="None"),
        ],
        labels=["0.5 μ", "1 μ", "2 μ"],
        bbox_to_anchor=(1.02, 0.65),  # Move the legend outside the plot
        borderaxespad=0,
    )
    lgnd.set_title("Beta Values")  # Add a title to the legend
    # add the legend manually to the current Axes
    plt.gca().add_artist(lgnd)

    # Create a legend for the lines, where the colors of the lines are:
    # - red for DRL-Agent (ours)
    # - green for Random
    # - blue for Degree
    # - orange for Roam

    plt.legend(
        loc="center left",
        handles=[
            plt.Line2D([], [], color=colors[0], linestyle="-"),
            plt.Line2D([], [], color=colors[1], linestyle="-"),
            plt.Line2D([], [], color=colors[2], linestyle="-"),
            plt.Line2D([], [], color=colors[3], linestyle="-"),
        ],
        labels=["DRL-Agent (ours)", "Random", "Degree", "Centrality", "Roam"],
        bbox_to_anchor=(1.02, 0.9),  # Move the legend outside the plot
        borderaxespad=0,
        title="Algorithms",
    )
    # plt.legend(
    #     bbox_to_anchor=(1.02, 0.9),
    #     loc="center left",
    #     borderaxespad=0,
    #     title="Algorithms",
    # )
    plt.tight_layout()
    # Save plot
    plt.savefig(f"test/node_hiding_tau_{tau_value}_sr_nmi.png")


if __name__ == "__main__":
    # Specify the tau values
    tau_values = [0.3, 0.5, 0.8]
    # tau_values = [0.3]

    # Load JSON data and process for each tau value
    for tau_value in tau_values:
        # Load JSON data from file
        path = f"test/"
        dataset = "fb-75/"
        algorithm = "walktrap/"
        task = "node_hiding"
        tau = f"/tau_{tau_value}/"
        json_file = f"allBetas_evaluation_{task}_mean_std.json"

        path = path + dataset + algorithm + task + tau + json_file
        with open(path, "r") as f:
            json_data = f.read()

        # Process JSON data
        if task == "node_hiding":
            df_sr, df_nmi, df_sr_float, df_nmi_float = process_json(
                json_data, tau_value
            )
        else:
            df_sr, df_nmi = process_json_community(json_data, tau_value)

        # Append results to the dataframes
        if tau_value == tau_values[0]:
            df_sr_all = df_sr
            df_nmi_all = df_nmi

            # df_sr_float_all = df_sr_float
            # df_nmi_float_all = df_nmi_float
        else:
            df_sr_all = df_sr_all._append(df_sr, ignore_index=True)
            df_nmi_all = df_nmi_all._append(df_nmi, ignore_index=True)

            # df_sr_float_all = df_sr_float_all._append(df_sr_float, ignore_index=True)
            # df_nmi_float_all = df_nmi_float_all._append(df_nmi_float, ignore_index=True)

    # save_lineplot(df_sr_float_all, df_nmi_float_all)
    # Convert dataframes to markdown tables
    if task == "node_hiding":
        print("Success Rate:")
    else:
        print("Deception Score:")
    print(df_sr_all.to_markdown(index=False))
    print("\n")
    print("NMI:")
    print(df_nmi_all.to_markdown(index=False))
