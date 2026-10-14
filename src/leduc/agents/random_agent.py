import random

from leduc.interface import Action, Agent, Observation


class RandomAgent(Agent):
    """Uniform legal-action baseline, implemented independently of RLCard's agent."""

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def on_match_start(self, *, seed: int) -> None:
        self._rng.seed(seed)

    def select_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise ValueError("Cannot choose an action from a terminal observation")
        return self._rng.choice(observation.legal_actions)
