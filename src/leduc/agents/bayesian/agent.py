"""Fixed Bayesian opponent model and belief-state search for RLCard Leduc.

No environment, hidden cards, or cross-hand opponent learning is used. Action
likelihoods are fitted offline from simulated styles and frozen during play.
"""
from functools import lru_cache

from leduc.interface import Agent, Observation
from .opponent_model import OpponentModel

CARDS = tuple(suit + rank for suit in ('S', 'H') for rank in ('J', 'Q', 'K'))


def legal_actions(chips, turn, raises):
    return tuple(a for a in ('call', 'raise', 'fold', 'check')
                 if not (a == 'raise' and raises >= 2)
                 and not (a == 'check' and chips[turn] < max(chips))
                 and not (a == 'call' and chips[turn] == max(chips)))


def normalize(weights):
    total = sum(weights)
    if total <= 0:
        raise ValueError('No possible opponent cards remain')
    return tuple(w / total for w in weights)


def advance(chips, turn, raises, quiet, action, round_index):
    chips = list(chips)
    if action == 'raise':
        chips[turn] = max(chips) + (2 if round_index == 0 else 4)
        raises += 1
        quiet = 1
    else:
        if action == 'call':
            chips[turn] = max(chips)
        quiet += 1
    return tuple(chips), 1 - turn, raises, quiet


class BayesianAgent(Agent):
    def __init__(self, model_path=None):
        self.model = OpponentModel.load(model_path)
        self.posterior = {}
        self.last_action_values = {}

    def configuration(self):
        return {"agent_class": type(self).__name__, "likelihood_model": dict(self.model.metadata),
                "smoothing_alpha": self.model.alpha}

    def on_match_start(self, *, seed):
        self.on_hand_start(hand_index=0, seat=0)

    def on_hand_start(self, *, hand_index, seat):
        self.posterior = {}
        self.last_action_values = {}

    def select_action(self, observation: Observation):
        own = observation.private_card
        seat = observation.seat
        candidates = tuple(c for c in CARDS if c != own)
        belief = normalize(tuple(1.0 for _ in candidates))
        small = observation.small_blind_seat
        chips = (1, 2) if small == 0 else (2, 1)
        turn, raises, quiet, rnd = small, 0, 0, 0
        # Reconstruct from scratch: each historical action contributes once.
        # A newly revealed public card is not used to explain preflop actions.
        for event in observation.history:
            if event.round_index != rnd:
                rnd = event.round_index
                raises = quiet = 0
            public = observation.public_card if rnd else None
            if event.seat != seat:
                legal = legal_actions(chips, turn, raises)
                belief = normalize(tuple(w * self.model.probabilities(c, public, legal, max(chips) - chips[turn], raises)[event.action]
                                         for c, w in zip(candidates, belief)))
            chips, turn, raises, quiet = advance(chips, turn, raises, quiet, event.action, rnd)
        if observation.round_index != rnd:
            raises = quiet = 0
        public = observation.public_card
        belief = normalize(tuple(0 if c == public else w for c, w in zip(candidates, belief)))
        self.posterior = dict(zip(candidates, belief))

        def showdown(chips, public, belief):
            def strength(c):
                return (c[1] == public[1], 'JQK'.index(c[1]))
            ours = strength(own)
            return sum(w * (chips[1-seat] if ours > strength(c) else
                            -chips[seat] if ours < strength(c) else
                            (chips[1-seat] - chips[seat]) / 2)
                       for c, w in zip(candidates, belief))

        @lru_cache(maxsize=None)
        def value(chips, turn, raises, quiet, rnd, public, belief):
            if quiet >= 2:
                if rnd == 1:
                    return showdown(chips, public, belief)
                total = 0.0
                # Four possible public cards for each opponent-card hypothesis.
                for card in CARDS:
                    if card == own:
                        continue
                    mass = sum(w for c, w in zip(candidates, belief) if c != card) / 4
                    if mass:
                        updated = normalize(tuple(w if c != card else 0 for c, w in zip(candidates, belief)))
                        total += mass * value(chips, turn, 0, 0, 1, card, updated)
                return total
            legal = legal_actions(chips, turn, raises)
            if turn == seat:
                return max(action_value(a, chips, turn, raises, quiet, rnd, public, belief) for a in legal)
            total = 0.0
            for action in legal:
                weighted = tuple(w * self.model.probabilities(c, public, legal, max(chips) - chips[turn], raises)[action]
                                 for c, w in zip(candidates, belief))
                mass = sum(weighted)
                if mass:
                    total += mass * action_value(action, chips, turn, raises, quiet,
                                                 rnd, public, normalize(weighted))
            return total

        def action_value(action, chips, turn, raises, quiet, rnd, public, belief):
            if action == 'fold':
                return -chips[seat] if turn == seat else chips[1-seat]
            next_state = advance(chips, turn, raises, quiet, action, rnd)
            return value(*next_state, rnd, public, belief)

        self.last_action_values = {
            action: action_value(action, observation.contributions, seat, raises, quiet,
                                 observation.round_index, public, belief)
            for action in observation.legal_actions
        }
        return max(self.last_action_values, key=self.last_action_values.get)
