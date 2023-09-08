#%%
from plot_utils import (
    plot_avg_cumulative_rewards,
    plot_eval_metrics,    
)

#%%
cleanup_result_path = "results_poster/cleanup_5agents.pkl"
plot_avg_cumulative_rewards(cleanup_result_path)
plot_eval_metrics(cleanup_result_path)


# %%
cleanup_result_path = "results_ep-len_5000/cleanup_5agents.pkl"
plot_avg_cumulative_rewards(cleanup_result_path)
plot_eval_metrics(cleanup_result_path)
# %%
# results 10 episodes 2000 episode length
cleanup_result_path = "results_ep-len_2000/cleanup_5agents.pkl"
plot_avg_cumulative_rewards(cleanup_result_path)
plot_eval_metrics(cleanup_result_path)
# %%
