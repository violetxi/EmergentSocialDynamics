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
        result_folder, filename.split('.')[0]
        )
    ensure_dir(folder_name)        
    # Prepare data for plotting
    plot_data = {}        
    for run in data:
        for model, episodes in run.items():
            # TODO temporary fix for the bug in the data
            if len(episodes[0]['agent_0']) > 30:
                if model not in plot_data:
                    plot_data[model] = {agent: [] for agent in episodes[0].keys()}
                for episode in episodes:
                    for agent, rewards in episode.items():
                        plot_data[model][agent].append(np.cumsum(rewards))  # Cumulative rewards for each episode
    
    # Set the color palette to "Set2"
    sns.set_palette("Set2")
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
        
        plt.savefig(os.path.join(folder_name, f"{model}.png"))
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
    df = pd.DataFrame({
        'model': model,
        'agent': agent,
        'reward': rewards
        } for model_res in data 
        for model, agent_rewards_list in model_res.items() 
        for agent_rewards in agent_rewards_list 
        for agent, rewards in agent_rewards.items()
        )
    df = df.explode('reward')
    #breakpoint()
    # Set the color palette to "Set2"
    sns.set_palette("Set2")
    # Remove grid lines and plot the bar plot grouped by agent_id with standard error
    plt.figure(figsize=(15, 10))
    # bar plot with standard error
    sns.barplot(x="agent", y="reward", hue="model", data=df, errorbar='se', capsize=.1)
    plt.title("Average Return for Each Agent in Each Model")
    plt.xlabel("Agent")
    plt.ylabel("Average Return")
    plt.legend(title="Model")
    sns.despine()
    # Save the plot
    folder_name = os.path.dirname(result_path)
    filename = os.path.basename(result_path).split('.')[0]
    plt.savefig(os.path.join(folder_name, f"{filename}.png"))
    plt.close()


if __name__ == '__main__':
    spread_result_path = "results/mpe-simple_spread_v3.pkl"
    plot_avg_cumulative_rewards(spread_result_path)
    plot_avg_return_with_std_error(spread_result_path)
    