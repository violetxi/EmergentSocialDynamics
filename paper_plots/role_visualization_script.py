#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
cleanup_dir_path = "cleanup/"  # Modify this to your directory path

# Define a consistent color palette based on the PPO plot using Seaborn's colorblind palette
colorblind_palette = sns.color_palette("colorblind", 15)
color_mapping_blind = {
    "ppo": colorblind_palette[0],
    "icm": colorblind_palette[1],
    "im_reward": colorblind_palette[2],
    "social_influence_visible": colorblind_palette[3],
    "SVO_hetero_75": colorblind_palette[4],
    "SVO_homog_30": colorblind_palette[5],
    "SVO_homog_15": colorblind_palette[6],
    "SVO_homog_45": colorblind_palette[7],
    "SVO_homog_60": colorblind_palette[8],
    "SVO_homog_75": colorblind_palette[9],
    "SVO_hetero_15": colorblind_palette[10],
    "SVO_hetero_30": colorblind_palette[11],
    "SVO_hetero_45": colorblind_palette[12],
    "SVO_hetero_60": colorblind_palette[13],
    "SVO_hetero_75": colorblind_palette[14]
}
agent_marker_mapping = {
    'agent_0': 'o', 
    'agent_1': 'x', 
    'agent_2': 's', 
    'agent_3': '+', 
    'agent_4': 'D'
    }
models2plots = {
    'ppo': 'PPO',
    'im_reward': 'ICM-R',
    'icm': 'ICM',        
    'social_influence_visible': 'Influence',
    'SVO_hetero_75': 'SVO-HE',
    'SVO_homog_30': 'SVO-HO'
}

# Define the plotting function with consistent color palette
def plot_consistent_color_quadrant_analysis(subset):
    # Data preparation
    all_data_subset = []
    for model_dir in subset:
        csv_files = [f for f in os.listdir(os.path.join(cleanup_dir_path, model_dir)) if f.endswith('.csv')]
        model_data = pd.concat([pd.read_csv(os.path.join(cleanup_dir_path, model_dir, csv)) for csv in csv_files])
        model_data["apple_eaten_z_score"] = (model_data["apple_eaten"] - model_data["apple_eaten"].mean()) / model_data["apple_eaten"].std()
        model_data["effective_clean_beam_z_score"] = (model_data["effective_clean_beam"] - model_data["effective_clean_beam"].mean()) / model_data["effective_clean_beam"].std()
        avg_z_scores_by_agent = model_data.groupby("Unnamed: 0")[["apple_eaten_z_score", "effective_clean_beam_z_score"]].mean().reset_index()
        avg_z_scores_by_agent["model"] = model_dir
        all_data_subset.append(avg_z_scores_by_agent)
    aggregated_df_subset = pd.concat(all_data_subset)
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="apple_eaten_z_score", 
        y="effective_clean_beam_z_score", 
        data=aggregated_df_subset, 
        hue="model", 
        style="Unnamed: 0", 
        s=150, 
        palette=color_mapping_blind, 
        alpha=0.8, 
        legend=None)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Apple Eaten (z-score)")
    plt.ylabel("Effective Clean Beam (z-score)")
    handles, labels = [], []
    for model in subset:
        handles.append(plt.Line2D([0], [0], color=color_mapping_blind[model], marker='s', markersize=10, label=models2plots[model], linestyle="None"
        ))
    for agent in aggregated_df_subset["Unnamed: 0"].unique():
        handles.append(plt.Line2D([0], [0], color="black", marker=agent_marker_mapping[agent], markersize=10, label=f"{agent}", linestyle="None"))
    plt.legend(handles=handles, title="Model & Agent", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./role_scatter.pdf")
    plt.show()

# Define your subsets here
subset_1 = ["ppo", "icm", "im_reward", "social_influence_visible", "SVO_hetero_75", "SVO_homog_30"]
#subset_2 = ["SVO_homog_15", "SVO_homog_30", "SVO_homog_45", "SVO_homog_60", "SVO_homog_75"]
#subset_3 = ["SVO_hetero_15", "SVO_hetero_30", "SVO_hetero_45", "SVO_hetero_60", "SVO_hetero_75"]

# Generate the plots
plot_consistent_color_quadrant_analysis(subset_1)
# plot_consistent_color_quadrant_analysis(subset_2)
# plot_consistent_color_quadrant_analysis(subset_3)


# %%
