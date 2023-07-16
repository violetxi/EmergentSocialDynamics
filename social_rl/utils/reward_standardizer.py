import torch
from typeguard import typechecked


@typechecked
class RewardStandardizer:
    """
    Class for standardizing rewards using running mean and standard deviation.
    Used because we have large intrinsic rewards and small extrinsic rewards.
    https://arxiv.org/pdf/1806.08295.pdf
    """
    def __init__(self) -> None:
        self.mean = 0
        self.std = 1
        self.count = 0


    def update(
            self, 
            reward: torch.Tensor
            ) -> None:
        # Update running mean and standard deviation
        self.count += 1
        delta = reward - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = reward - self.mean
        # Welford's method for numerical stability
        self.std = torch.sqrt(
            (self.std**2 * (self.count - 1) + delta * delta2) / self.count
            )


    def standardize(
            self, 
            reward: torch.Tensor
            ) -> float:
        # Standardize reward
        standardized_reward = (reward - self.mean) / (self.std + 1e-8)  # add small constant to avoid division by zero
        return standardized_reward.item()

