import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generates the boxplot for single axis tracking
def plot_box_singlets(groups: dict) -> None:
    labels = ['SAC', 'RED3Q', 'RED5Q']

    for title, data_list in groups.items():
        dfs = []
        for agent_label, data in zip(labels, data_list):
            # Make sure data is a numpy array for convenience and take abs if needed
            arr = np.array(data).flatten()  
            dfs.append(pd.DataFrame({
                'Error': arr,
                'Agent': agent_label
            }))

        df = pd.concat(dfs, ignore_index=True)

        plt.figure(figsize=(7, 5))
        sns.boxplot(x='Agent', y='Error', data=df, palette='Set2', fill=False)
        plt.yscale('log')
        plt.xlabel("Agent")
        plt.ylabel("Mean absolute error (deg)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"{title}.png"), dpi=300)
        plt.close()
    #plt.show()

def plot_box_pairs(nom_data: list[np.ndarray], eff_data: list[np.ndarray], jolt_data: list[np.ndarray]) -> None:
    df_nom = _prepare_pitchroll(nom_data, 'Nominal')
    df_eff = _prepare_pitchroll(eff_data, 'Quarter')
    df_jolt = _prepare_pitchroll(jolt_data, 'Jolt')

    _plot_condition(df_nom)
    _plot_condition(df_eff)
    _plot_condition(df_jolt)

def _plot_condition(df):
    condition = df["Condition"].iloc[0]  # all rows share the same condition

    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x='Agent', y='Value', hue='Angle', palette='Set2', fill=False)
    plt.yscale('log')  # Optional: use if error values vary widely
    plt.ylabel("Mean absolute error (deg)")
    plt.xlabel("Agent")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Error')
    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"box_pitchroll_{condition}.png"), dpi=300)
    plt.close()


def _prepare_pitchroll(data_list, condition):
    """
    data_list: list of np arrays, each shape (N_samples, 2), ordered as [SAC, RED3Q, RED5Q]
    condition: string for condition label
    """
    rows = []
    agents = ['SAC', 'RED3Q', 'RED5Q']
    for i, agent in enumerate(agents):
        arr = data_list[i]  # shape (N_samples, 2)
        pitch_vals = arr[:, 0]
        roll_vals = arr[:, 1]
        for v in pitch_vals:
            rows.append({'Agent': agent, 'Angle': '$\\theta_e$', 'Value': v, 'Condition': condition})
        for v in roll_vals:
            rows.append({'Agent': agent, 'Angle': '$\\phi_e$', 'Value': v, 'Condition': condition})
    return pd.DataFrame(rows)