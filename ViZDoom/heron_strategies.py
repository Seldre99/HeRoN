"""
HERON Strategy Variants.

Implements 4 Helper/Reviewer intervention strategies:
1. HERON Initial - Assistance in first N turns only
2. HERON Final (delta=0.1) - Probabilistic, more help towards end
3. HERON Final (delta=0.2) - Probabilistic, even more help towards end
4. HERON Random - Random choice each turn

"""

import random
from enum import Enum
from typing import Tuple


class HERONStrategy(Enum):
    """Available HERON strategy types."""
    INITIAL = "initial"
    FINAL_01 = "final_01"
    FINAL_02 = "final_02"
    RANDOM = "random"


class HERONController:
    """
    Controls when to call Helper/Reviewer.
    Encapsulates the logic for each HERON strategy.
    """

    def __init__(self, strategy: HERONStrategy):
        """
        Initialize controller with a HERON strategy.

        Args:
            strategy: One of the 4 HERONStrategy variants
        """
        self.strategy = strategy
        self.current_turn = 0
        self.turns_in_episode = 0
        self.threshold = 0.0

    def reset_episode(self):
        """Reset counters for a new episode."""
        self.current_turn = 0
        self.turns_in_episode = 0
        self.threshold = 0.0

    def should_call_helper(self) -> bool:
        """
        Decide whether to call Helper based on current strategy.

        Returns:
            True if Helper should be consulted
        """
        if self.strategy == HERONStrategy.INITIAL:
            return self._initial_strategy()
        elif self.strategy == HERONStrategy.FINAL_01:
            return self._final_strategy(delta=0.1)
        elif self.strategy == HERONStrategy.FINAL_02:
            return self._final_strategy(delta=0.2)
        elif self.strategy == HERONStrategy.RANDOM:
            return self._random_strategy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def increment_turn(self):
        """Increment turn counter within episode."""
        self.current_turn += 1
        self.turns_in_episode += 1

    def _initial_strategy(self) -> bool:
        """
        HERON Initial: Concentrated assistance at episode start (~20%).

        - Helper called every 20 steps in first ~60 steps (3 calls)
        - Then every 30 steps until step 120 (2 more calls)
        - After step 120: no helper (autonomous agent)

        Returns:
            True if helper should be called in early phase
        """
        if self.current_turn < 60:
            return self.current_turn % 20 == 0
        elif self.current_turn <= 120:
            return (self.current_turn - 60) % 30 == 0
        else:
            return False

    def _final_strategy(self, delta: float) -> bool:
        """
        HERON Final: Probabilistic assistance increasing towards end.

        - Initial threshold: 1.0
        - Every 20 turns: threshold -= delta
        - Each turn: if random() > threshold, call Helper

        Args:
            delta: Threshold decrement per 20 turns
                   0.1 = slow decrease (help concentrated at end)
                   0.2 = fast decrease (help starts earlier)

        Returns:
            True if random() > threshold (becomes more likely over time)
        """
        if self.current_turn % 20 == 0:
            self.threshold -= delta

        threshold_clamped = max(0.0, min(1.0, self.threshold))
        return random.random() > threshold_clamped

    def _random_strategy(self) -> bool:
        """
        HERON Random: Independent random choice each turn.

        5% probability of calling Helper per turn.

        Returns:
            True with 5% probability
        """
        return random.random() < 0.05


class HERONTrainer:
    """
    Wrapper for integrating HERON strategies in training loop.
    Manages strategy selection and tracks statistics.
    """

    def __init__(self, strategy: HERONStrategy):
        """Initialize with a HERON strategy."""
        self.controller = HERONController(strategy)
        self.strategy = strategy
        self.helper_calls_per_episode = 0
        self.helper_calls_by_strategy = {
            HERONStrategy.INITIAL: 0,
            HERONStrategy.FINAL_01: 0,
            HERONStrategy.FINAL_02: 0,
            HERONStrategy.RANDOM: 0,
        }

    def reset_episode(self):
        """Reset controller for a new episode."""
        self.controller.reset_episode()
        self.helper_calls_per_episode = 0

    def should_call_helper(self) -> bool:
        """Ask controller if Helper should be called."""
        should_call = self.controller.should_call_helper()
        if should_call:
            self.helper_calls_per_episode += 1
            self.helper_calls_by_strategy[self.strategy] += 1
        self.controller.increment_turn()
        return should_call

    def get_stats(self) -> dict:
        """Return statistics for current strategy."""
        return {
            "strategy": self.strategy.value,
            "helper_calls_this_episode": self.helper_calls_per_episode,
            "total_calls_by_strategy": self.helper_calls_by_strategy[self.strategy],
        }


def parse_strategy(strategy_name: str) -> HERONStrategy:
    """
    Convert strategy name string to HERONStrategy enum.

    Args:
        strategy_name: 'initial', 'final_01', 'final_02', or 'random'

    Returns:
        Corresponding HERONStrategy enum

    Raises:
        ValueError if name is not recognized
    """
    strategy_map = {
        'initial': HERONStrategy.INITIAL,
        'final_01': HERONStrategy.FINAL_01,
        'final_02': HERONStrategy.FINAL_02,
        'random': HERONStrategy.RANDOM,
    }

    normalized = strategy_name.lower().strip()
    if normalized not in strategy_map:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {', '.join(strategy_map.keys())}"
        )

    return strategy_map[normalized]
