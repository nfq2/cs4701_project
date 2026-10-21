"""Frozen empirical P(action | hypothetical card, public betting context).

Only the offline generator sees the acting simulated player's real card. At
inference, the Bayesian agent supplies a hypothetical card, never hidden state.
"""
from functools import lru_cache
from importlib.resources import files
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType

ACTIONS = ('call', 'raise', 'fold', 'check')
SCHEMA_VERSION = 1


def context_keys(card, public, legal, to_call, raises):
    # Suits do not affect strength in Leduc. Public rank determines round.
    base = (card[1], public[1] if public else '-', ','.join(sorted(legal)))
    return ('|'.join((*base, str(to_call), str(raises))), '|'.join(base), base[2])


class OpponentModel:
    """Read-only count tables with Laplace smoothing and hierarchical backoff."""

    def __init__(self, payload):
        if payload.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported opponent model schema')
        alpha = payload.get('alpha')
        if not isinstance(alpha, (float, int)) or not isfinite(alpha) or alpha <= 0:
            raise ValueError('Smoothing alpha must be positive and finite')
        self.alpha = float(alpha)
        self.metadata = MappingProxyType(payload.get('metadata', {}).copy())
        self.tables = tuple(MappingProxyType({k: tuple(v) for k, v in table.items()})
                            for table in payload['tables'])
        if len(self.tables) != 3 or any(
            len(v) != len(ACTIONS) or any(not isinstance(n, int) or n < 0 for n in v)
            for table in self.tables for v in table.values()
        ):
            raise ValueError('Invalid count tables')

    @classmethod
    def load(cls, path=None):
        source = Path(path) if path is not None else files('leduc.agents.bayesian').joinpath('models/bayesian-v1.json')
        return cls(json.loads(source.read_text(encoding='utf-8')))

    @lru_cache(maxsize=8192)
    def probabilities(self, card, public, legal, to_call, raises):
        keys = context_keys(card, public, legal, to_call, raises)
        counts = (0,) * len(ACTIONS)
        for table, key in zip(self.tables, keys):
            if key in table:
                counts = table[key]
                break
        total = sum(counts[ACTIONS.index(a)] + self.alpha for a in legal)
        return MappingProxyType({a: (counts[ACTIONS.index(a)] + self.alpha) / total
                                 for a in legal})
