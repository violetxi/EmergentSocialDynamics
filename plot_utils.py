import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typeguard import typechecked

from social_rl.utils.utils import ensure_dir



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

        ax.set_title(f"Average Cumulative Rewards over Episodes for {model}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Average Cumulative Reward")
        ax.legend()
        sns.despine()
        # save the plot 
        fig_save_path = os.path.join(folder_name, f"{model}.png")
        plt.savefig(fig_save_path)
        print(f"Plot saved to {fig_save_path}")
        plt.close()


def plot_avg_return_with_std_error(result_path: str) -> None:
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
        axes[i].legend(title="Model")
    
    sns.despine()
    plt.tight_layout()
    # Save the plot
    folder_name = os.path.dirname(result_path)
    filename = os.path.basename(result_path).split('.pkl')[0]
    fig_save_path = os.path.join(folder_name, f"{filename}.png")
    plt.savefig(fig_save_path)
    print(f"Plot saved to {fig_save_path}")
    plt.close()


    # df = pd.DataFrame({
    #     'model': model,
    #     'agent': agent,
    #     'reward': rewards
    #     } for model_res in data 
    #     for model, agent_rewards_list in model_res.items() 
    #     for agent_rewards in agent_rewards_list 
    #     for agent, rewards in agent_rewards.items()
    #     )
    # df = df.explode('reward')

    # sns.set_palette("colorblind")
    # # Remove grid lines and plot the bar plot grouped by agent_id with standard error
    # plt.figure(figsize=(15, 10))
    # # bar plot with standard error
    # sns.barplot(x="agent", y="reward", hue="model", data=df, errorbar='se', capsize=.1)
    # plt.title("Average Return for Each Agent in Each Model")
    # plt.xlabel("Agent")
    # plt.ylabel("Average Return")
    # plt.legend(title="Model")
    # sns.despine()    


if __name__ == '__main__':
    cleanup_result_path = "log/cleanup.pkl"
    plot_avg_cumulative_rewards(cleanup_result_path)
    plot_avg_return_with_std_error(cleanup_result_path)
        