from copy import deepcopy
import json

import pytest

from leduc.agents import BayesianAgent
from leduc.agents.bayesian.opponent_model import ACTIONS, OpponentModel
from leduc.agents.bayesian.train import evaluate, fit, generate_records


def example(action):
    return {'style': 'tight', 'action': action, 'observation': {
        'private_card': 'SK', 'public_card': None, 'legal_actions': ['call', 'raise', 'fold'],
        'to_call': 1, 'round_index': 0, 'history': []}}


def test_likelihoods_are_estimated_from_counts_and_smoothed():
    records = [example('raise')] * 6 + [example('call')] * 2
    model = OpponentModel(fit(records))
    probs = model.probabilities('HK', None, ('call', 'raise', 'fold'), 1, 0)
    assert probs['raise'] == pytest.approx(7 / 11)
    assert probs['call'] == pytest.approx(3 / 11)
    assert probs['fold'] == pytest.approx(1 / 11)
    # Fits actually respond to the data, rather than returning style formulas.
    other = OpponentModel(fit([example('fold')] * 8))
    assert other.probabilities('SK', None, ('call', 'raise', 'fold'), 1, 0)['fold'] == pytest.approx(9 / 11)
    with pytest.raises(TypeError):
        probs['raise'] = 0


def test_backoff_and_unseen_contexts_stay_normalized_and_legal():
    model = OpponentModel(fit([example('raise')] * 6 + [example('call')] * 2))
    # Unseen bet size -> card/public/legal counts; unseen card -> legal counts.
    for card, owed in [('HK', 99), ('SJ', 99)]:
        probs = model.probabilities(card, None, ('call', 'raise', 'fold'), owed, 1)
        assert probs['raise'] == pytest.approx(7 / 11)
    probs = model.probabilities('SQ', 'HJ', ('check', 'fold'), 0, 2)
    assert dict(probs) == {'check': .5, 'fold': .5}
    assert 'raise' not in probs


def test_generation_is_reproducible_and_game_splits_are_disjoint():
    train = list(generate_records(12, 123))
    assert train == list(generate_records(12, 123))
    heldout = list(generate_records(12, 456))
    assert {r['game_id'] for r in train}.isdisjoint(r['game_id'] for r in heldout)
    assert {r['style'] for r in train} == {
        'tight', 'aggressive', 'bluff-heavy', 'calling-heavy', 'balanced'}
    for row in train:
        assert row['action'] in row['observation']['legal_actions']
        assert 'opponent_card' not in row['observation']
    payload = fit(train)
    model = OpponentModel(payload)
    before = deepcopy(payload)
    metrics = evaluate(model, heldout)
    assert metrics['decisions'] == len(heldout)
    assert payload == before  # Scoring never fits on held-out records.


def test_saved_model_roundtrip_and_agent_freeze(tmp_path):
    payload = fit(list(generate_records(12, 123)), metadata={'training_seed': 123})
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(payload))
    agent = BayesianAgent(model_path=path)
    from test_bayesian import opening
    original = [dict(t) for t in agent.model.tables]
    agent.select_action(opening())
    agent.on_hand_start(hand_index=1, seat=0)
    agent.on_match_start(seed=5)
    assert [dict(t) for t in agent.model.tables] == original
    assert agent.configuration()['likelihood_model']['training_seed'] == 123


def test_bundled_model_probability_tables_and_provenance():
    model = OpponentModel.load()
    assert model.metadata['training_seed'] != model.metadata['heldout_seed']
    assert model.metadata['hands_per_style'] == 5000
    for table in model.tables:
        for key, counts in table.items():
            assert all(n >= 0 for n in counts)
            assert len(counts) == len(ACTIONS)
    for rank in 'JQK':
        for public in (None, 'SJ', 'HQ', 'SK'):
            for legal in (('call', 'raise', 'fold'), ('check', 'raise', 'fold'), ('check', 'fold')):
                probs = model.probabilities('H' + rank, public, legal, 2, 1)
                assert sum(probs.values()) == pytest.approx(1)
                assert all(0 < p <= 1 for p in probs.values())
