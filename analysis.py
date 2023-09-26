#%%
from plot_utils import (
    plot_avg_cumulative_rewards,
    plot_eval_metrics2,
    plot_eval_metrics3,
)

#%%
cleanup_result_path = "results/cleanup_5agents.pkl"
plot_avg_cumulative_rewards(cleanup_result_path)
plot_eval_metrics2(cleanup_result_path)
plot_eval_metrics3(cleanup_result_path)

# %%
# harvest_result_path = "results/harvest_5agents.pkl"
# plot_avg_cumulative_rewards(harvest_result_path)
# plot_eval_metrics2(harvest_result_path)
# plot_eval_metrics3(harvest_result_path)
# %%
