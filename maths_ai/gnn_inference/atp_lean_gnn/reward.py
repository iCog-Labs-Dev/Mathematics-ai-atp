from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import Tensor


class RewardSource(ABC):
    @abstractmethod
    def get_reward(self, state: Any, action: int, next_state: Any) -> float:
        """Return the STV reward for a single (state, action, next_state) transition."""
        pass

    @abstractmethod
    def get_rewards_batch(
        self,
        actions: Tensor,
        tactic_targets: Tensor,
        success_mask: Tensor,
    ) -> Tensor:
        """Compute rewards for an entire batch. Returns [B] tensor of rewards."""
        pass


class MockRewardSource(RewardSource):
    def get_reward(self, state: Any, action: int, next_state: Any) -> float:
        return 1.0 if action == 0 else 0.0

    def get_rewards_batch(
        self,
        actions: Tensor,
        tactic_targets: Tensor,
        success_mask: Tensor,
    ) -> Tensor:
        # Returns 1.0 where actions == tactic_targets, 0.0 elsewhere
        return (actions == tactic_targets).float().to(device=actions.device)


class PLNRewardSource(RewardSource):
    def __init__(self) -> None:
        self.cache: dict[tuple[str, int, str], float] = {}

    def get_reward(self, state: Any, action: int, next_state: Any) -> float:
        key = (str(state), int(action), str(next_state))
        if key in self.cache:
            return self.cache[key]
        reward = 0.0
        self.cache[key] = reward
        return reward

    def get_rewards_batch(
        self,
        actions: Tensor,
        tactic_targets: Tensor,
        success_mask: Tensor,
    ) -> Tensor:
        return success_mask.float().to(device=success_mask.device)
