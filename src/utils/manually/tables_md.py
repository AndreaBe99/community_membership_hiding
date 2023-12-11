import json
import pandas as pd
import json
import pandas as pd


def process_json_community(data, tau):
    data = json.loads(data)
    metrics = ["deception_score", "nmi"]

    taus = [0.3, 0.5, 0.8]

    results_sr = {
        # "τ": [],
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
        # results_sr["τ"].append(tau)
        results_sr["β"].append(beta)

    df_sr = pd.DataFrame(results_sr)

    results_nmi = {
        # "τ": [],
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
        # results_nmi["τ"].append(tau)
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

    return df_sr, df_nmi


# JSON data for tau=0.8


if __name__ == "__main__":
    # Specify the tau values
    tau_values = [0.3]  # , 0.5, 0.8]

    # Load JSON data and process for each tau value
    for tau_value in tau_values:
        # Load JSON data from file
        path = f"test/"
        dataset = "fb-75/"
        algorithm = "greedy/"
        task = "community_hiding"
        tau = f"/tau_{tau_value}/"
        json_file = f"allBetas_evaluation_{task}_mean_std.json"

        path = path + dataset + algorithm + task + tau + json_file
        with open(path, "r") as f:
            json_data = f.read()

        # Process JSON data
        if task == "node_hiding":
            df_sr, df_nmi = process_json(json_data, tau_value)
        else:
            df_sr, df_nmi = process_json_community(json_data, tau_value)

        # Append results to the dataframes
        if tau_value == tau_values[0]:
            df_sr_all = df_sr
            df_nmi_all = df_nmi
        else:
            df_sr_all = df_sr_all._append(df_sr, ignore_index=True)
            df_nmi_all = df_nmi_all._append(df_nmi, ignore_index=True)

    # Convert dataframes to markdown tables
    if task == "node_hiding":
        print("Success Rate:")
    else:
        print("Deception Score:")
    print(df_sr_all.to_markdown(index=False))
    print("\n")
    print("NMI:")
    print(df_nmi_all.to_markdown(index=False))
