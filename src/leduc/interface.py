"""Agent-facing data only: never pass an RLCard environment to an agent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Action = Literal["call", "raise", "fold", "check"]


@dataclass(frozen=True)
class ActionEvent:
    seat: int
    round_index: int
    action: Action


@dataclass(frozen=True)
class Observation:
    hand_index: int
    seat: int
    small_blind_seat: int
    private_card: str
    public_card: str | None
    round_index: int
    contributions: tuple[int, int]
    pot: int
    to_call: int
    legal_actions: tuple[Action, ...]
    history: tuple[ActionEvent, ...]


@dataclass(frozen=True)
class HandResult:
    """Player-specific terminal feedback; folded opponent cards stay hidden."""

    hand_index: int
    seat: int
    private_card: str
    public_card: str | None
    opponent_revealed_card: str | None
    payoff_chips: float
    history: tuple[ActionEvent, ...]


class Agent(ABC):
    """Override lifecycle hooks as needed; select_action is the only required method.

    Match start resets opponent memory and seeds randomness. Hand start resets
    hand-local state only. Learned model parameters should survive both hooks.
    Feedback is delivered after every hand, including hands with no own actions.
    """

    def on_match_start(self, *, seed: int) -> None:
        pass

    def on_hand_start(self, *, hand_index: int, seat: int) -> None:
        pass

    @abstractmethod
    def select_action(self, observation: Observation) -> Action:
        raise NotImplementedError

    def on_hand_end(self, result: HandResult) -> None:
        pass

    def configuration(self) -> dict:
        """Public experiment settings; never return credentials."""
        return {"agent_class": type(self).__name__}

    def decision_metadata(self) -> dict:
        return {}

    def statistics(self) -> dict:
        return {}
