"""Tabular Q-learning agent for RLCard Leduc, trained by self-play.

The agent starts with no strategic rules: every Q-value begins at 0. During
training it explores legal actions epsilon-greedily and backs up the net chips
won or lost at the end of each hand. After training, agents load a frozen Q-table and
act greedily. No additional learning occurs during matches or human play.

"""
from importlib.resources import files
import json
from pathlib import Path
import random

from leduc.interface import Action, Agent, HandResult, Observation

SCHEMA_VERSION = 1
ACTION_CODES = {'call': 'c', 'raise': 'r', 'fold': 'f', 'check': 'k'}


def state_key(observation: Observation) -> str:
    """
    All required information to pick next step
    """
    own = observation.private_card[1]
    public = observation.public_card[1] if observation.public_card else '-'
    position = 'sb' if observation.seat == observation.small_blind_seat else 'bb'
    rounds = ['', '']
    for event in observation.history:
        rounds[event.round_index] += ACTION_CODES[event.action]
    return f'{own}|{public}|{position}|{rounds[0]}/{rounds[1]}'


class QTable:

    def __init__(self, q_values=None, visits=None):
        self.q_values = {s: dict(a) for s, a in (q_values or {}).items()}
        self.visits = {s: dict(a) for s, a in (visits or {}).items()}

    def get(self, state, action):
        return self.q_values.get(state, {}).get(action, 0.0)

    def set(self, state, action, value):
        self.q_values.setdefault(state, {})[action] = value
        counts = self.visits.setdefault(state, {})
        counts[action] = counts.get(action, 0) + 1

    def __len__(self):
        return len(self.q_values)


class QLearningAgent(Agent):

    def __init__(self, model_path=None, *, table=None, training=False,
                 alpha=0.05, epsilon=0.0, gamma=1.0):
        self.metadata = {}
        if table is None:
            table = self._load(model_path)
        self.table = table
        self.training = training
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self._rng = random.Random(0)
        self._last = None
        self.last_q_values = {}

    # persistence
    def _load(self, model_path):
        if model_path is not None:
            source = Path(model_path)
        else:
            source = files('leduc.agents.reinforcement_learning').joinpath('models/qlearning-v1.json')
            if not source.is_file():
                return QTable()
        payload = json.loads(source.read_text(encoding='utf-8'))
        if payload.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported Q-learning model schema')
        self.metadata = payload.get('metadata', {})
        return QTable(payload['q_values'], payload.get('visits'))

    def save(self, path, metadata=None):
        payload = {'schema_version': SCHEMA_VERSION, 'metadata': metadata or {},
                   'q_values': self.table.q_values, 'visits': self.table.visits}
        Path(path).write_text(json.dumps(payload, sort_keys=True, indent=1) + '\n', encoding='utf-8')

    # Q-learning
    def q_value(self, state, action):
        """Q(state, action); 0.0 for anything never visited."""
        return self.table.get(state, action)

    def state_value(self, state, legal_actions):
        """max_a Q(state, a) over legal actions; 0.0 at a terminal state."""
        if not legal_actions:
            return 0.0
        return max(self.q_value(state, a) for a in legal_actions)

    def greedy_action(self, state, legal_actions):
        """Best action by Q-value, breaking ties uniformly at random."""
        best = self.state_value(state, legal_actions)
        return self._rng.choice([a for a in legal_actions if self.q_value(state, a) == best])

    def update(self, state, action, reward, next_state=None, next_legal=()):
        """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))."""
        current = self.q_value(state, action)
        target = reward
        if next_state is not None:
            target += self.discount * self.state_value(next_state, next_legal)
        self.table.set(state, action, current + self.alpha * (target - current))

    # agent
    def configuration(self):
        return {'agent_class': type(self).__name__, 'training': self.training,
                'alpha': self.alpha, 'epsilon': self.epsilon, 'gamma': self.discount,
                'states': len(self.table), 'model': dict(self.metadata)}

    def on_match_start(self, *, seed: int) -> None:
        self._rng.seed(seed)
        self._last = None

    def on_hand_start(self, *, hand_index: int, seat: int) -> None:
        self._last = None
        self.last_q_values = {}

    def select_action(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if not legal:
            raise ValueError('Cannot choose an action from a terminal observation')
        state = state_key(observation)
        if self.training and self._last is not None:
            # Previous decision led here with no chips settled yet (reward 0)
            self.update(*self._last, 0.0, state, legal)
        self.last_q_values = {a: self.q_value(state, a) for a in legal}
        if self._rng.random() < self.epsilon:
            action = self._rng.choice(legal)
        else:
            action = self.greedy_action(state, legal)
        self._last = (state, action)
        return action

    def on_hand_end(self, result: HandResult) -> None:
        if self.training and self._last is not None:
            # Terminal transition, reward is the hand's net chips
            self.update(*self._last, result.payoff_chips)
        self._last = None

    def decision_metadata(self):
        return {'q_values': dict(self.last_q_values)}
