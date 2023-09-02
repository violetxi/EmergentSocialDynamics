import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typeguard import typechecked

from social_rl.utils.utils import ensure_dir


# format model names
def format_model_name(name):
    parameters = name.split('-')
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
    return '-'.join(formatted_params)


@typechecked
def plot_avg_cumulative_rewards(result_path: str) -> None:
    """Plot cumulative rewards for each agent across differe
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
    
    formatted_model_names = [format_model_name(name) for name in model_names]

    # Plotting using Seaborn
    sns.set_palette("colorblind")
    rotation = 90
    horizontal_alignment = 'left'
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    # Subplot 1: Average total population return with standard error
    sns.barplot(
        x=formatted_model_names, 
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
    axs[0].set_xticklabels(formatted_model_names, rotation=rotation, ha=horizontal_alignment)

    # Subplot 2: Average Gini coefficient
    sns.barplot(
        x=formatted_model_names, 
        y=average_ginis, 
        errorbar=None, 
        ax=axs[1], 
        saturation=0.5
        )
    sns.despine()    
    axs[1].set_ylabel('Average Equity (1 - Gini Coefficient)')
    axs[1].set_xlabel('Model')
    axs[1].set_title("Population Equity")
    axs[1].set_xticklabels(formatted_model_names, rotation=rotation, ha=horizontal_alignment)

    plt.tight_layout()
    fig_path = os.path.join(
        os.path.dirname(result_path), 
        os.path.basename(result_path).split('.')[0] + '_eval_metrics.png'
        )
    plt.savefig(fig_path)



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
        