"""Tabular Q-learning trained by self-play, plus the frozen Q-table asset."""
from .agent import QLearningAgent, QTable, state_key

__all__ = ["QLearningAgent", "QTable", "state_key"]
