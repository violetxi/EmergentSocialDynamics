import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typeguard import typechecked

from social_rl.utils.utils import ensure_dir


# format model names
def format_model_name(name, verbose=False):
    parameters = name.split('-')
    # add all hyperparameters to the model name
    if verbose:        
        formatted_params = [parameters[0]]  # keep the primary descriptor as is    
        for param in parameters[1:]:        
            key = param.split('=')[0].split('.')[-1]    #param.split('=').split('.')  #param.split('.')[-1].split('=')[0]
            value = param.split('=')[-1]
            try:
                value = float(value)
                if value.is_integer():
                    formatted_value = f"{int(value):d}"
                else:
                    formatted_value = f"{value:.5f}"
            except ValueError:
                formatted_value = value
            formatted_params.append(f"{key}={formatted_value}")
        out_name = '-'.join(formatted_params)
    else:
        out_name = parameters[0]
    return out_name    


@typechecked
def plot_avg_cumulative_rewards(result_path: str) -> None:
    """
    Plot cumulative rewards for each agent across differe
    """
    data = pickle.load(open(result_path, 'rb'))    
    filename = os.path.basename(result_path)
    result_folder = os.path.dirname(result_path)
    # Create folder for the plots if it doesn't exist
    folder_name = os.path.join(
        result_folder, filename.split('.pkl')[0]
        )
    ensure_dir(folder_name)        
    # Prepare data for plotting
    plot_data = {}        
    for run in data:
        for model, episodes in run.items():       
            if model not in plot_data:
                model = format_model_name(model)                
                plot_data[model] = {agent: [] for agent in episodes[0].keys()}
            for episode in episodes:
                for agent, rewards in episode.items():
                    plot_data[model][agent].append(np.cumsum(rewards))  # Cumulative rewards for each episode
    
    sns.set_palette("colorblind")
    # Calculate average cumulative reward and standard error for each episode across all runs
    for model, agents in plot_data.items():        
        if model in plot_data:            
            agents = plot_data[model]
            fig, ax = plt.subplots(figsize=(10, 6))
            for agent, rewards in agents.items():
                rewards = np.array(rewards)  # Convert rewards to numpy array for broadcasting
                avg_rewards = np.mean(rewards, axis=0)  # Average rewards for each episode across all runs
                std_error = np.std(rewards, axis=0) / np.sqrt(len(rewards))  # Standard error for each episode across all runs

                # Plot the average cumulative rewards over episodes with standard error as shaded area using seaborn
                ax.plot(range(len(avg_rewards)), avg_rewards, label=agent)
                ax.fill_between(range(len(avg_rewards)), avg_rewards - std_error, avg_rewards + std_error, alpha=0.2)

            ax.set_title(f"{model}")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Average Cumulative Reward")
            ax.legend(title="Agent",  prop={'size': 12}, title_fontsize='14')
            sns.despine()
            # save the plot 
            fig_save_path = os.path.join(folder_name, f"{model}.png")
            plt.savefig(fig_save_path)
            print(f"Plot saved to {fig_save_path}")
            plt.close()


def plot_eval_metrics(result_path: str) -> None:        
    def compute_gini_coefficient_with_negative_rewards_shifted(rewards):
        # compute Gini coefficient handling negative values by shifting the rewards 
        # distribution to make all values positive
        shift_value = abs(min(rewards)) + 1e-5  # small epsilon to ensure no zero value
        shifted_rewards = [r + shift_value for r in rewards]        
        # compute Gini coefficient for shifted rewards
        sorted_rewards = sorted(shifted_rewards)
        n = len(sorted_rewards)
        numer = sum([(i + 1) * reward for i, reward in enumerate(sorted_rewards)])
        denom = n * sum(sorted_rewards)        
        if denom == 0:
            return 0
        else:
            return (2 * numer / denom) - (n + 1) / n

    # Load data from the pickle file
    with open(result_path, 'rb') as f:
        data = pd.read_pickle(f)    
    # extracting model names and their corresponding rewards data
    model_names = [list(item.keys())[0] for item in data]
    model_rewards = [list(item.values())[0] for item in data]
    
    # compute average total population return and standard error for each model
    average_returns = []
    standard_errors = []
    for rewards in model_rewards:
        # Calculate total return for each episode
        episode_totals = [sum([episode[agent].sum() for agent in episode]) for episode in rewards]        
        # Compute average and standard error
        average_returns.append(np.mean(episode_totals))
        standard_errors.append(np.std(episode_totals) / np.sqrt(len(episode_totals)))
    # Compute average Gini coefficient for each model
    average_ginis = []
    for rewards in model_rewards:
        # get the difference between 1 and the Gini coefficient 
        gini_values = [
            1 - compute_gini_coefficient_with_negative_rewards_shifted([
                episode[agent].sum() for agent in episode
                ]) for episode in rewards
            ]
        average_ginis.append(np.mean(gini_values))

    #breakpoint()
    model_names2plot = ['ppo', 'icm', 'im_reward', 'social_influence_visible', 'SVO_hetero_75', 'SVO_homog_15']
    model_labels = ['PPO', 'ICM', 'ICM Reward', 'Social Influence', 'SVO \nHeterogeneous', 'SVO \nHomogeneous']
    # take subset of average returns and standard errors based on model_names2plot
    print(average_returns, model_names)
    average_returns = [avg_return for i, avg_return in enumerate(average_returns) if model_names[i] in model_names2plot]
    standard_errors = [std_error for i, std_error in enumerate(standard_errors) if model_names[i] in model_names2plot]
    # take subset of average gini coefficients based on model_names2plot
    average_ginis = [avg_gini for i, avg_gini in enumerate(average_ginis) if model_names[i] in model_names2plot]
    # take subset of model names based on model_names2plot
    model_names = [model_name for model_name in model_names if model_name in model_names2plot]

    # Plotting using Seaborn
    sns.set_palette("colorblind")
    rotation = 0
    horizontal_alignment = 'center'
    fig, axs = plt.subplots(1, 2, figsize=(17, 10))
    # Subplot 1: Average total population return with standard error    
    sns.barplot(
        x=model_names,#formatted_model_names, 
        y=average_returns, 
        yerr=standard_errors, 
        capsize=0.1, 
        errorbar=None, 
        ax=axs[0],
        saturation=0.5
        )
    sns.despine()
    axs[0].set_ylabel('Average Return')
    axs[0].set_xlabel('Model')
    axs[0].set_title("Population Return")
    axs[0].set_xticklabels(model_labels, rotation=rotation, ha=horizontal_alignment)

    # Subplot 2: Average Gini coefficient
    sns.barplot(
        x=model_names,#formatted_model_names, 
        y=average_ginis, 
        errorbar=None, 
        ax=axs[1], 
        saturation=0.5
        )
    sns.despine()    
    axs[1].set_ylabel('Average Equity (1 - Gini Coefficient)')
    axs[1].set_xlabel('Model')
    axs[1].set_title("Population Equity")
    axs[1].set_xticklabels(model_labels, rotation=rotation, ha=horizontal_alignment)

    plt.tight_layout()
    fig_path = os.path.join(
        os.path.dirname(result_path), 
        os.path.basename(result_path).split('.')[0] + '_eval_metrics.png'
        )
    plt.savefig(fig_path)


def plot_eval_metrics2(result_path: str) -> None:        
    def compute_gini_coefficient_with_negative_rewards_shifted(rewards):
        # compute Gini coefficient handling negative values by shifting the rewards 
        # distribution to make all values positive
        shift_value = abs(min(rewards)) + 1e-5  # small epsilon to ensure no zero value
        shifted_rewards = [r + shift_value for r in rewards]        
        # compute Gini coefficient for shifted rewards
        sorted_rewards = sorted(shifted_rewards)
        n = len(sorted_rewards)
        numer = sum([(i + 1) * reward for i, reward in enumerate(sorted_rewards)])
        denom = n * sum(sorted_rewards)        
        if denom == 0:
            return 0
        else:
            return (2 * numer / denom) - (n + 1) / n

    # Load data from the pickle file
    with open(result_path, 'rb') as f:
        data = pd.read_pickle(f)    
    # extracting model names and their corresponding rewards data
    model_names = [list(item.keys())[0] for item in data]
    model_rewards = [list(item.values())[0] for item in data]
    
    # compute average total population return and standard error for each model
    average_returns = []
    standard_errors = []
    for rewards in model_rewards:
        # Calculate total return for each episode
        episode_totals = [sum([episode[agent].sum() for agent in episode]) for episode in rewards]        
        # Compute average and standard error
        average_returns.append(np.mean(episode_totals))
        standard_errors.append(np.std(episode_totals) / np.sqrt(len(episode_totals)))
    # Compute average Gini coefficient for each model
    average_ginis = []
    standard_errors_gini = []
    for rewards in model_rewards:
        # get the difference between 1 and the Gini coefficient 
        gini_values = [
            1 - compute_gini_coefficient_with_negative_rewards_shifted([
                episode[agent].sum() for agent in episode
                ]) for episode in rewards
            ]
        average_ginis.append(np.mean(gini_values))
        standard_errors_gini.append(np.std(gini_values) / np.sqrt(len(gini_values)))
        
    if 'clean' in result_path:
        model_names2plot = {
            'ppo': 'PPO',
            'im_reward': 'ICM Reward',
            'icm': 'ICM',        
            'social_influence_visible': 'Social Influence',
            'SVO_hetero_75': 'SVO \nHeterogeneous',
            'SVO_homog_30': 'SVO \nHomogeneous'
        }
    elif 'harvest' in result_path:
        model_names2plot = {
            'ppo': 'PPO',
            'im_reward': 'ICM Reward',
            'icm': 'ICM',        
            'social_influence_visible': 'Social Influence',
            'SVO_hetero_15': 'SVO \nHeterogeneous',
            'SVO_homog_30': 'SVO \nHomogeneous'
        }
    # take subset of average returns and standard errors based on model_names2plot
    average_returns_plot = []
    standard_errors_plot = []
    average_ginis_plot = []
    standard_errors_gini_plot = []    
    #for i, avg_return in enumerate(average_returns):
    for model_name in model_names2plot.keys():
        i = model_names.index(model_name)                
        average_returns_plot.append(average_returns[i])
        standard_errors_plot.append(standard_errors[i])
        average_ginis_plot.append(average_ginis[i])
        standard_errors_gini_plot.append(standard_errors_gini[i])
    model_names_plot = list(model_names2plot.keys())
    model_labels = list(model_names2plot.values())
    
    # Set plot aesthetics
    sns.set_palette("colorblind")
    rotation = 0
    horizontal_alignment = 'center'

    # Font size settings
    label_fontsize = 28
    title_fontsize = 40
    tick_fontsize = 14

    # Plot 1: Average total population return with standard error
    fig1 = plt.figure(figsize=(10, 6))
    sns.barplot(
        x=model_names_plot,
        y=average_returns_plot,
        yerr=standard_errors_plot,
        capsize=0.1,
        errorbar=None,
        saturation=0.5
    )
    sns.despine()
    plt.ylabel('Average Return', fontsize=label_fontsize)
    plt.xlabel('Model', fontsize=label_fontsize)
    plt.title("Population Return", fontsize=title_fontsize)
    ticks = plt.xticks()[0]
    plt.xticks(ticks=ticks, labels=model_labels, rotation=rotation, ha=horizontal_alignment, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    fig1_path = os.path.join(
        os.path.dirname(result_path),
        os.path.basename(result_path).split('.')[0] + '_eval_metrics_return.png'
    )
    fig1.savefig(fig1_path)
    
    # Plot 2: Average Gini coefficient
    fig2 = plt.figure(figsize=(10, 6))
    sns.barplot(
        x=model_names_plot,
        y=average_ginis_plot,
        yerr=standard_errors_gini_plot,
        errorbar=None,
        saturation=0.5
    )
    sns.despine()
    plt.ylabel('Equity (1 - Gini Coefficient)', fontsize=22)
    plt.xlabel('Model', fontsize=label_fontsize)
    plt.title("Population Equity", fontsize=title_fontsize)
    ticks = plt.xticks()[0]
    plt.xticks(ticks=ticks, labels=model_labels, rotation=rotation, ha=horizontal_alignment, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    fig2_path = os.path.join(
        os.path.dirname(result_path),
        os.path.basename(result_path).split('.')[0] + '_eval_metrics_gini.png'
    )
    fig2.savefig(fig2_path)


def plot_eval_metrics3(result_path: str) -> None:        
    def compute_gini_coefficient_with_negative_rewards_shifted(rewards):
        # compute Gini coefficient handling negative values by shifting the rewards 
        # distribution to make all values positive
        shift_value = abs(min(rewards)) + 1e-5  # small epsilon to ensure no zero value
        shifted_rewards = [r + shift_value for r in rewards]        
        # compute Gini coefficient for shifted rewards
        sorted_rewards = sorted(shifted_rewards)
        n = len(sorted_rewards)
        numer = sum([(i + 1) * reward for i, reward in enumerate(sorted_rewards)])
        denom = n * sum(sorted_rewards)        
        if denom == 0:
            return 0
        else:
            return (2 * numer / denom) - (n + 1) / n

    # Load data from the pickle file
    with open(result_path, 'rb') as f:
        data = pd.read_pickle(f)    
    # extracting model names and their corresponding rewards data
    model_names = [list(item.keys())[0] for item in data]
    model_rewards = [list(item.values())[0] for item in data]

    # compute average total population return and standard error for each model
    average_returns = []
    standard_errors = []
    for rewards in model_rewards:
        # Calculate total return for each episode
        episode_totals = [sum([episode[agent].sum() for agent in episode]) for episode in rewards]        
        # Compute average and standard error
        average_returns.append(np.mean(episode_totals))
        standard_errors.append(np.std(episode_totals) / np.sqrt(len(episode_totals)))
    # Compute average Gini coefficient for each model
    average_ginis = []
    for rewards in model_rewards:
        # get the difference between 1 and the Gini coefficient 
        gini_values = [
            1 - compute_gini_coefficient_with_negative_rewards_shifted([
                episode[agent].sum() for agent in episode
                ]) for episode in rewards
            ]
        average_ginis.append(np.mean(gini_values))

    # Create a dictionary
    data = {
        'Model': model_names,
        'Average Returns': average_returns,
        'Standard Errors': standard_errors,
        'Average Gini Coefficients': average_ginis
    }
    df = pd.DataFrame(data)
    # Filter rows based on 'Model'
    filtered_df = df[df['Model'].str.contains('SVO|svo', case=False)]
    # Create a new column for 'Heterogeneous' or 'Homogeneous'
    filtered_df['Type'] = ['Heterogeneous' if 'hetero' in model or 'heter' in model else 'Homogeneous' for model in filtered_df['Model']]
    # Create a new column to hold the degree (the last number in the 'Model')
    filtered_df['degree'] = filtered_df['Model'].str.extract(r'(\d+)$')[0].astype(int)
    #model_labels = ['PPO', 'ICM', 'ICM Reward', 'Social Influence', 'SVO \nHeterogeneous', 'SVO \nHomogeneous']

    # Sort the DataFrame by 'degree' and then by 'Type'
    sorted_df = filtered_df.sort_values(by=['degree', 'Type'])
    # Extract the sorted standard errors into a list
    sorted_standard_errors = sorted_df['Standard Errors'].tolist()

    # Set plot aesthetics
    sns.set_palette("colorblind")
    rotation = 0
    horizontal_alignment = 'center'

    # Font size settings
    label_fontsize = 28
    title_fontsize = 40
    tick_fontsize = 14

    # Plot 1: Average total population return with standard error
    fig1 = plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=filtered_df,
        x='degree',
        y='Average Returns',
        hue='Type',
        capsize=0.1,
        errorbar=None,
        saturation=0.5
    )
    sns.despine()
    # Add error bars
    for i, bar in enumerate(ax.patches):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        err = sorted_standard_errors[i]  # get the corresponding sorted error
        plt.errorbar(x, y, yerr=err, color='black', capsize=3, fmt='none')

    plt.ylabel('Population Return', fontsize=label_fontsize)
    plt.xlabel('Social Value Orientation (\u00B0)', fontsize=label_fontsize)
    plt.title("Return by Level of Altruism", fontsize=title_fontsize)
    plt.xticks(rotation=rotation, ha=horizontal_alignment, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=18)
    plt.tight_layout()
    fig1_path = os.path.join(
        os.path.dirname(result_path),
        os.path.basename(result_path).split('.')[0] + '_eval_metrics_svo_by_degree.png'
    )
    fig1.savefig(fig1_path)

    # Plot 2: Average Gini coefficient
    fig2 = plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=filtered_df,
        x='degree',
        y='Average Gini Coefficients',
        hue='Type',
        capsize=0.1,
        errorbar=None,
        saturation=0.5
    )
    sns.despine()
    plt.ylabel('Equity (1 - Gini Coefficient)', fontsize=22)
    plt.xlabel('Social Value Orientation (\u00B0)', fontsize=label_fontsize)
    plt.title("Equity by Level of Altruism", fontsize=36)
    plt.xticks(rotation=rotation, ha=horizontal_alignment, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.tight_layout()
    fig2_path = os.path.join(
        os.path.dirname(result_path),
        os.path.basename(result_path).split('.')[0] + '_eval_metrics_gini_by_degree.png'
    )
    fig2.savefig(fig2_path)


def plot_avg_agent_return_with_std_error(result_path: str) -> None:
    # Define helper function to compute average return
    def compute_avg_return(data):
        # Prepare data for computing
        avg_returns = {}
        for run in data:
            for model, episodes in run.items():
                if model not in avg_returns:
                    avg_returns[model] = {agent: [] for agent in episodes[0].keys()}
                for episode in episodes:
                    for agent, rewards in episode.items():
                        avg_returns[model][agent].append(np.sum(rewards))  # Sum of rewards for each episode
        # Calculate average return and standard error for each model
        for model, agents in avg_returns.items():
            agents = avg_returns[model]
            for agent, rewards in agents.items():
                avg_returns[model][agent] = (np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards)))  # Average return and standard error

        return avg_returns
    # Load pickle files
    data = pickle.load(open(result_path, "rb"))
    data = compute_avg_return(data)
    # Convert the data to a DataFrame format
    df = pd.DataFrame([
        {'model': model, 'agent': agent, 'avg_reward': values[0], 'std_dev': values[1]}
        for model, agents in data.items()
        for agent, values in agents.items()
    ])
    
    # Determine unique agent counts across models
    agent_counts = df.groupby('model')['agent'].nunique().unique()    
    sns.set_palette("colorblind")    
    # Create subplots for each unique agent count    
    fig, axes = plt.subplots(1, len(agent_counts), figsize=(15 * len(agent_counts), 10))
    # Ensure axes is a list for consistency in indexing
    if len(agent_counts) == 1:
        axes = [axes]    
    for i, count in enumerate(agent_counts):
        # Filter data for models with the current agent count
        models_with_count = df[df['model'].isin(df.groupby('model').filter(lambda x: x['agent'].nunique() == count)['model'].unique())]        
        # Plot on the current subplot
        ax = sns.barplot(x="agent", y="avg_reward", hue="model", data=models_with_count, ax=axes[i], capsize=.1)        
        # Adding error bars
        for bar, err in zip(ax.patches, models_with_count['std_dev']):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=err, color='black', capsize=3, fmt='none', zorder=3)

        axes[i].set_title(f"Average Return for Models with {count} Agents")
        axes[i].set_xlabel("Agent")
        axes[i].set_ylabel("Average Return per Episode")
        axes[i].legend(title="Model", prop={'size': 12}, title_fontsize='14')
    
    sns.despine()
    plt.tight_layout()
    # Save the plot
    folder_name = os.path.dirname(result_path)
    filename = os.path.basename(result_path).split('.pkl')[0]
    fig_save_path = os.path.join(folder_name, f"{filename}.png")
    plt.savefig(fig_save_path)
    print(f"Plot saved to {fig_save_path}")
    plt.close()


if __name__ == '__main__':
    cleanup_result_path = "results/cleanup_5agents.pkl"
    plot_avg_cumulative_rewards(cleanup_result_path)
    plot_eval_metrics(cleanup_result_path)
    # plot_avg_agent_return_with_std_error(cleanup_result_path)
        