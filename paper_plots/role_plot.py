# Write the full script to a .py file for the user to download
#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the subsets
subset_1 = ['ppo', 'icm', 'im_reward', 'social_influence_visible', 'SVO_hetero_75', 'SVO_homog_30']
subset_2 = ['svo_homog_15', 'svo_homog_30', 'svo_homog_45', 'svo_homog_60', 'svo_homog_75']
subset_3 = ['svo_hetero_15', 'svo_hetero_30', 'svo_hetero_45', 'svo_hetero_60', 'svo_hetero_75']
model_plot_names = {
    'ppo': 'PPO',
    'im_reward': 'ICM-R',
    'icm': 'ICM',        
    'social_influence_visible': 'Influence',
    'SVO_hetero_75': 'SVO-HE',
    'SVO_homog_30': 'SVO-HO'
}

# Directory path
cleanup_dir_path = "cleanup"

def plot_pie_charts_all_models(subset):
    all_data_subset = []

    for model_dir in subset:
        csv_files = [f for f in os.listdir(os.path.join(cleanup_dir_path, model_dir)) if f.endswith('.csv')]
        model_data = pd.concat([pd.read_csv(os.path.join(cleanup_dir_path, model_dir, csv)) for csv in csv_files])
        model_data = model_data.groupby("Unnamed: 0").mean().reset_index()
        model_data["apple_eaten_z_score"] = (model_data["apple_eaten"] - model_data["apple_eaten"].mean()) / model_data["apple_eaten"].std()
        model_data["effective_clean_beam_z_score"] = (model_data["effective_clean_beam"] - model_data["effective_clean_beam"].mean()) / model_data["effective_clean_beam"].std()
        model_data["Quadrant"] = model_data.apply(lambda row: "Eat More & Clean More" if row["apple_eaten_z_score"] > 0 and row["effective_clean_beam_z_score"] > 0 else ("Eat Less & Clean More" if row["apple_eaten_z_score"] <= 0 and row["effective_clean_beam_z_score"] > 0 else ("Eat Less & Clean Less" if row["apple_eaten_z_score"] <= 0 and row["effective_clean_beam_z_score"] <= 0 else "Eat More & Clean Less")), axis=1)
        model_data["Model"] = model_dir
        all_data_subset.append(model_data)

    aggregated_df_subset = pd.concat(all_data_subset)
    quadrant_colors = {
        "Eat More & Clean More": sns.color_palette("pastel", n_colors=4)[0],
        "Eat Less & Clean More": sns.color_palette("pastel", n_colors=4)[1],
        "Eat Less & Clean Less": sns.color_palette("pastel", n_colors=4)[2],
        "Eat More & Clean Less": sns.color_palette("pastel", n_colors=4)[3]
    }

    fig, axes = plt.subplots(1, len(subset), figsize=(20, 4))
    for idx, model in enumerate(subset):
        data = aggregated_df_subset[aggregated_df_subset["Model"] == model]
        quadrant_counts = data["Quadrant"].value_counts()
        colors = [quadrant_colors[q] for q in quadrant_counts.index]
        axes[idx].pie(quadrant_counts, labels=None, colors=colors, startangle=90)
        axes[idx].set_title(model_plot_names[model], fontsize=20)
    
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=quadrant, markersize=10, markerfacecolor=quadrant_colors[quadrant]) for quadrant in quadrant_colors]
    fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.12, 0.5), title="Roles", fontsize='large', title_fontsize='large')
    plt.tight_layout()
    # save plot as a high res pdf file
    plt.savefig("./role_pie_charts_all_models.pdf")
    plt.show()

# bar to represent the distance
def plot_bar_charts_all_models(subset):
    all_data_subset = []

    for model_dir in subset:
        csv_files = [f for f in os.listdir(os.path.join(cleanup_dir_path, model_dir)) if f.endswith('.csv')]
        model_data = pd.concat([pd.read_csv(os.path.join(cleanup_dir_path, model_dir, csv)) for csv in csv_files])
        model_data = model_data.groupby("Unnamed: 0").mean().reset_index()
        model_data["apple_eaten_z_score"] = (model_data["apple_eaten"] - model_data["apple_eaten"].mean()) / model_data["apple_eaten"].std()
        model_data["effective_clean_beam_z_score"] = (model_data["effective_clean_beam"] - model_data["effective_clean_beam"].mean()) / model_data["effective_clean_beam"].std()
        model_data["Quadrant"] = model_data.apply(lambda row: "Eat More & Clean More" if row["apple_eaten_z_score"] > 0 and row["effective_clean_beam_z_score"] > 0 else ("Eat Less & Clean More" if row["apple_eaten_z_score"] <= 0 and row["effective_clean_beam_z_score"] > 0 else ("Eat Less & Clean Less" if row["apple_eaten_z_score"] <= 0 and row["effective_clean_beam_z_score"] <= 0 else "Eat More & Clean Less")), axis=1)
        model_data["Distance"] = model_data.apply(lambda row: ((row["apple_eaten_z_score"] ** 2) + (row["effective_clean_beam_z_score"] ** 2)) ** 0.5, axis=1)
        model_data["Model"] = model_dir
        all_data_subset.append(model_data)

    aggregated_df_subset = pd.concat(all_data_subset)
    quadrant_colors = {
        "Eat More & Clean More": sns.color_palette("pastel", n_colors=4)[0],
        "Eat Less & Clean More": sns.color_palette("pastel", n_colors=4)[1],
        "Eat Less & Clean Less": sns.color_palette("pastel", n_colors=4)[2],
        "Eat More & Clean Less": sns.color_palette("pastel", n_colors=4)[3]
    }

    y_max = aggregated_df_subset["Distance"].max() + 0.1
    y_min = 0

    fig, axes = plt.subplots(1, len(subset), figsize=(20, 4), sharey=True)
    for idx, model in enumerate(subset):
        data = aggregated_df_subset[aggregated_df_subset["Model"] == model]
        avg_distances = data.groupby("Quadrant")["Distance"].mean().reset_index()
        sns.barplot(
            data=avg_distances, x="Quadrant", y="Distance", 
            ax=axes[idx], palette=quadrant_colors, 
            order=["Eat More & Clean More", "Eat Less & Clean More", "Eat Less & Clean Less", "Eat More & Clean Less"]
            )
        axes[idx].set_title(model_plot_names[model])
        axes[idx].set_ylim(y_min, y_max)
        axes[idx].set_xticks([]) 
        if idx == 0:
            axes[idx].set_ylabel("Average Distance")            
        else:
            axes[idx].set_ylabel("")
    
    plt.tight_layout()
    plt.show()


# Plot pie charts and bar charts for all subsets
for s in [subset_1]:#, subset_2, subset_3]:
    plot_pie_charts_all_models(s)
    #plot_bar_charts_all_models(s)


# %%
