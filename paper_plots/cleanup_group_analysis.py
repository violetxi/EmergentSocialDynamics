# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:24:27 2023

@author: locro
"""
#%%
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cleanup_path = 'cleanup/'
cleanup_contents = os.listdir(cleanup_path)
models2plots = {
    'ppo': 'PPO',
    'im_reward': 'ICM-R',
    'icm': 'ICM',        
    'social_influence_visible': 'Influence',
    'SVO_hetero_75': 'SVO-HE',
    'SVO_homog_30': 'SVO-HO'
}
# Initialize an empty DataFrame to store the data for 'Effective_Clean_Beam' and 'Reward' summed across agents for each episode and model
summed_data_df = pd.DataFrame(columns=['Model', 'Episode', 'Summed_Effective_Clean_Beam', 'Summed_Reward'])

# Iterate through each model folder to load the CSV files and collect the required data
for model in cleanup_contents:
    if model in models2plots.keys():
        model_path = os.path.join(cleanup_path, model)
        print(os.listdir(model_path))

        # Skip files, only process directories (i.e., models)
        if not os.path.isdir(model_path):
            continue
        
        # List the files in each model's folder
        episode_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]
        
        # Iterate through each episode file to read and collect the data
        for episode_file in episode_files:
            episode_path = os.path.join(model_path, episode_file)
            episode_data = pd.read_csv(episode_path)
            print(episode_data.head())    
            # Standardize column names by replacing spaces with underscores and making them lowercase
            episode_data.columns = episode_data.columns.str.replace(' ', '_').str.lower()
            
            # Calculate the sums for 'effective_clean_beam' and 'reward'
            summed_effective_clean_beam = episode_data['effective_clean_beam'].sum()
            summed_reward = episode_data['reward'].sum()
            
            # Append this episode's summed data to the global DataFrame
            summed_data_df.loc[len(summed_data_df)] = {
                'Model': model,
                'Episode': episode_file,
                'Summed_Effective_Clean_Beam': summed_effective_clean_beam,
                'Summed_Reward': summed_reward
            }

# Convert 'Summed_Effective_Clean_Beam' and 'Summed_Reward' to numeric types to ensure they can be used for correlation calculation
summed_data_df['Summed_Effective_Clean_Beam'] = pd.to_numeric(summed_data_df['Summed_Effective_Clean_Beam'], errors='coerce')
summed_data_df['Summed_Reward'] = pd.to_numeric(summed_data_df['Summed_Reward'], errors='coerce')

# Calculate the correlation between 'Summed_Effective_Clean_Beam' and 'Summed_Reward'
correlation_summed = summed_data_df['Summed_Effective_Clean_Beam'].corr(summed_data_df['Summed_Reward'])

# Create the scatterplot
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(x='Summed_Effective_Clean_Beam', y='Summed_Reward', data=summed_data_df, hue='Model', palette='tab10')
# Add grid lines
ax.grid(False)

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# add legend
handles, labels = [], []
for model in models2plots.keys():
    handles.append(plt.Line2D([0], [0], marker='o', markersize=10, label=models2plots[model], linestyle="None"
    ))
# Add the correlation as text to the plot
plt.text(max(summed_data_df['Summed_Effective_Clean_Beam']) * 0.7, min(summed_data_df['Summed_Reward']) + (max(summed_data_df['Summed_Reward']) - min(summed_data_df['Summed_Reward'])) * 0.1, f'Correlation: {correlation_summed:.2f}', fontsize=12)

# Add title and labels
#plt.title('Summed Effective Clean Beam and Summed Reward per Episode')
plt.xlabel('Summed Effective Clean Beam')
plt.ylabel('Summed Reward')

plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./clean_reward_corr.pdf")
plt.show()

# %%
