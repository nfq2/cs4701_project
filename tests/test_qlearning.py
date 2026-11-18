from dataclasses import replace
import io
import json

import pytest

from leduc.agents import AGENTS, QLearningAgent, RandomAgent
from leduc.agents.reinforcement_learning import QTable, state_key
from leduc.agents.reinforcement_learning.train import train
from leduc.interface import ActionEvent, HandResult, Observation
from leduc.runner import run_match


def opening():
    return Observation(0, 1, 0, 'SK', None, 0, (1, 2), 3, 0,
                       ('raise', 'fold', 'check'), ())


def result(payoff, history=()):
    return HandResult(0, 1, 'SK', None, None, payoff, history)


def test_state_key_ignores_suits_and_encodes_rounds():
    obs = replace(opening(), public_card='HJ', round_index=1,
                  history=(ActionEvent(0, 0, 'raise'), ActionEvent(1, 0, 'call'),
                           ActionEvent(0, 1, 'check')))
    assert state_key(obs) == 'K|J|bb|rc/k'
    assert state_key(replace(obs, private_card='HK', public_card='SJ')) == state_key(obs)
    assert state_key(replace(obs, seat=0)).split('|')[2] == 'sb'


def test_unseen_values_are_zero_and_greedy_is_legal():
    agent = QLearningAgent(table=QTable())
    obs = opening()
    assert agent.select_action(obs) in obs.legal_actions
    assert agent.last_q_values == {a: 0.0 for a in obs.legal_actions}


def test_terminal_update_uses_net_chips():
    agent = QLearningAgent(table=QTable(), training=True, alpha=0.5)
    agent.on_hand_start(hand_index=0, seat=1)
    action = agent.select_action(opening())
    agent.on_hand_end(result(4.0))
    assert agent.q_value(state_key(opening()), action) == pytest.approx(2.0)


def test_intermediate_update_bootstraps_from_next_decision():
    table = QTable({'K|-|bb|krr/': {'call': 6.0, 'fold': -2.0}})
    agent = QLearningAgent(table=table, training=True, alpha=1.0, epsilon=0.0)
    agent.on_hand_start(hand_index=0, seat=1)
    first = replace(opening(), history=(ActionEvent(0, 0, 'check'),),
                    legal_actions=('raise', 'check'))
    table.q_values[state_key(first)] = {'check': 0.0, 'raise': 1.0}
    assert agent.select_action(first) == 'raise'
    later = replace(first, history=(ActionEvent(0, 0, 'check'), ActionEvent(1, 0, 'raise'),
                                    ActionEvent(0, 0, 'raise')),
                    legal_actions=('call', 'fold'))
    agent.select_action(later)
    # alpha=1, gamma=1, reward 0: Q(first, raise) <- max Q(later, .)
    assert agent.q_value(state_key(first), 'raise') == pytest.approx(6.0)


def test_frozen_agent_never_updates():
    agent = QLearningAgent(table=QTable(), training=False, alpha=1.0)
    agent.select_action(opening())
    agent.on_hand_end(result(10.0))
    assert len(agent.table) == 0


def test_self_play_learns_and_saves(tmp_path):
    table, curve = train(hands=2000, chunk=500, alpha=0.1, gamma=1.0, epsilon_start=0.3,
                         epsilon_end=0.05, seed=1, eval_hands=200, eval_seed=2, verbose=False)
    assert len(table) > 100 and curve[-1]['hands'] == 2000
    path = tmp_path / 'q.json'
    QLearningAgent(table=table).save(path, {'note': 'test'})
    loaded = QLearningAgent(model_path=path)
    assert loaded.table.q_values == table.q_values
    assert loaded.configuration()['model'] == {'note': 'test'}


def test_registered_match_legal_and_reproducible():
    assert 'qlearning' in AGENTS
    runs = []
    for _ in range(2):
        log = io.StringIO()
        run_match((AGENTS['qlearning']['factory'](), RandomAgent()), hands=40, seed=42, log=log)
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        assert all(r['action'] in r['observation']['legal_actions']
                   for r in records if r['type'] == 'action')
        runs.append([r for r in records if r['type'] == 'hand_end'])
    assert runs[0] == runs[1]
