from dataclasses import replace
import io
import json

import pytest

from leduc.agents import BayesianAgent, RandomAgent
from leduc.interface import ActionEvent, Observation
from leduc.runner import run_match
from leduc.ui import HumanGame


def opening():
    return Observation(0, 1, 0, 'SK', None, 0, (1, 2), 3, 0,
                       ('raise', 'fold', 'check'), ())


def test_prior_card_exclusion_and_public_card():
    agent = BayesianAgent()
    obs = opening()
    agent.select_action(obs)
    assert 'SK' not in agent.posterior
    assert sum(agent.posterior.values()) == pytest.approx(1)
    assert all(w == pytest.approx(.2) for w in agent.posterior.values())
    obs = replace(obs, public_card='HJ', round_index=1, contributions=(2, 2),
                  pot=4, history=(ActionEvent(0, 0, 'call'), ActionEvent(1, 0, 'check')))
    assert agent.select_action(obs) in obs.legal_actions
    assert agent.posterior['HJ'] == 0
    assert sum(agent.posterior.values()) == pytest.approx(1)


def test_raise_updates_beliefs_once_and_resets():
    agent = BayesianAgent()
    obs = replace(opening(), contributions=(4, 2), pot=6, to_call=2,
                  legal_actions=('call', 'raise', 'fold'),
                  history=(ActionEvent(0, 0, 'raise'),))
    action = agent.select_action(obs)
    posterior = agent.posterior.copy()
    assert posterior['HK'] > posterior['SJ']
    assert agent.select_action(obs) == action
    assert agent.posterior == posterior
    assert agent.last_action_values['fold'] == -2
    agent.on_hand_start(hand_index=1, seat=1)
    assert agent.posterior == {}
    assert agent.last_action_values == {}


def test_match_legal_and_reproducible():
    summaries = []
    for _ in range(2):
        log = io.StringIO()
        run_match((BayesianAgent(), RandomAgent()), hands=40, seed=42, log=log)
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        assert all(r['action'] in r['observation']['legal_actions']
                   for r in records if r['type'] == 'action')
        ends = [r for r in records if r['type'] == 'hand_end']
        assert len(ends) == 40
        assert all(sum(r['payoffs_by_seat_chips']) == 0 for r in ends)
        summaries.append(ends)
    assert summaries[0] == summaries[1]


def test_dropdown_switch_and_restart():
    game = HumanGame(seed=123)
    view = game.change_opponent('bayesian')
    assert view['opponent'] == 'bayesian'
    original = game.agent
    for _ in range(5):
        while not game.view()['over']:
            legal = game.view()['legal_actions']
            game.act('check' if 'check' in legal else 'call')
        game.next_hand()
    view = game.restart()
    assert view['opponent'] == 'bayesian'
    assert game.agent is not original
    assert view['hand'] == 1


def test_search_betting_transitions_match_rlcard():
    from leduc.agents.bayesian_agent import advance, legal_actions
    log = io.StringIO()
    run_match((RandomAgent(), RandomAgent()), hands=100, seed=91, log=log)
    for line in log.getvalue().splitlines():
        record = json.loads(line)
        if record['type'] != 'action':
            continue
        obs = record['observation']
        small = obs['small_blind_seat']
        chips = (1, 2) if small == 0 else (2, 1)
        turn, raises, quiet, rnd = small, 0, 0, 0
        for event in obs['history']:
            if event['round_index'] != rnd:
                rnd = event['round_index']
                raises = quiet = 0
            chips, turn, raises, quiet = advance(chips, turn, raises, quiet,
                                                  event['action'], rnd)
        if obs['round_index'] != rnd:
            raises = quiet = 0
        assert chips == tuple(obs['contributions'])
        assert turn == obs['seat']
        assert set(legal_actions(chips, turn, raises)) == set(obs['legal_actions'])
