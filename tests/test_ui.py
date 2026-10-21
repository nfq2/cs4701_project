import io
import json

import pytest

from leduc.ui import HumanGame


def test_play_many_hands_privacy_accounting_and_logs():
    log = io.StringIO()
    game = HumanGame(seed=123, log=log)
    total = 0
    for hand in range(50):
        view = game.view()
        assert view['hand'] == hand + 1
        while not view['over']:
            assert view['opponent_card'] is None
            assert 'seed' not in view
            assert game.seat == game.human_seat
            action = 'check' if 'check' in view['legal_actions'] else 'call'
            view = game.act(action)
        assert (view['opponent_card'] is not None) == view['showdown']
        assert view['legal_actions'] == []
        total += view['payoff']
        assert view['total_chips'] == total
        assert view['completed'] == hand + 1
        before = game.view()
        with pytest.raises(ValueError):
            game.act('call')
        assert game.view() == before
        game.next_hand()
    events = [json.loads(line) for line in log.getvalue().splitlines()]
    assert any(e['type'] == 'action' and e['actor'] == 'human' for e in events)
    assert any(e['type'] == 'action' and e['actor'] == 'random' for e in events)
    assert sum(e['type'] == 'hand_end' for e in events) >= 50


def test_illegal_actions_and_midhand_deal_do_not_change_state():
    game = HumanGame(seed=123)
    while game.view()['over']:
        game.next_hand()
    before = game.view()
    for action in ('cheat', None, 1, [], {}):
        with pytest.raises(ValueError):
            game.act(action)
        assert game.view() == before
    with pytest.raises(ValueError):
        game.next_hand()
    assert game.view() == before


def test_fold_hides_opponent_and_settles_once():
    game = HumanGame(seed=123)
    while game.view()['over']:
        game.next_hand()
    result = game.act('fold')
    assert result['over']
    assert result['payoff'] < 0
    assert result['opponent_card'] is None
    assert not result['showdown']
    assert game.view() == result
    assert game.view() == result


def test_session_seed_replays_human_actions():
    games = [HumanGame(seed=456), HumanGame(seed=456)]
    for _ in range(20):
        assert games[0].view() == games[1].view()
        if games[0].view()['over']:
            for game in games:
                game.next_hand()
        else:
            action = games[0].view()['legal_actions'][0]
            for game in games:
                game.act(action)



def test_opponent_selection_validates_choices_without_changing_game():
    game = HumanGame(seed=123)
    before = game.view()
    assert {o['id'] for o in before['opponents']} == {'random', 'bayesian', 'llm', 'llm-memory'}
    assert game.change_opponent('random') == before
    for invalid in ('missing', None, [], {}):
        with pytest.raises(ValueError):
            game.change_opponent(invalid)
        assert game.view() == before



def test_restart_resets_score_and_hand_and_keeps_opponent(monkeypatch):
    monkeypatch.setattr("leduc.ui.secrets.randbits", lambda bits: 123)
    log = io.StringIO()
    game = HumanGame(seed=123, log=log)
    game.act('fold')
    revision = game.revision
    view = game.restart()
    assert view['hand'] == 1
    assert view['completed'] == 0
    assert view['total_chips'] == 0
    assert view['opponent'] == 'random'
    assert view['revision'] > revision
    assert view['opponent_card'] is None
    # Restart is also available during an unfinished hand.
    revision = view['revision']
    assert game.restart()['revision'] > revision
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert sum(r['type'] == 'session_end' and r['reason'] == 'restart' for r in records) == 2


def test_llm_switch_and_restart_preserve_configuration_and_reset_memory(monkeypatch):
    from leduc.agents import LLMAgent, MemoryLLMAgent, AGENTS
    made = []
    def factory(name):
        if name in ('llm', 'llm-memory'):
            cls = MemoryLLMAgent if name == 'llm-memory' else LLMAgent
            agent = cls(model='test-model', provider='groq', max_calls=9,
                        reasoning_effort='low', completion=lambda payload: {
                            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                                'action': 'check' if 'check' in json.loads(payload['messages'][1]['content'])['legal_actions'] else 'call'
                            })}}]})
        else:
            agent = AGENTS[name]['factory']()
        made.append(agent)
        return agent
    monkeypatch.setattr('leduc.ui.secrets.randbits', lambda bits: 123)
    log = io.StringIO()
    game = HumanGame(seed=123, log=log, agent_factory=factory)
    game.change_opponent('llm-memory')
    original = game.agent
    game.act('fold')
    assert len(original._completed_hands) == 1
    revision = game.revision
    game.restart()
    assert game.agent is not original
    assert game.agent._completed_hands == []
    assert game.agent.configuration()['model'] == 'test-model'
    assert game.agent.max_calls == 9
    assert game.revision > revision
    game.change_opponent('llm')
    assert type(game.agent) is LLMAgent
    game.change_opponent('bayesian')
    assert game.view()['opponent'] == 'bayesian'
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert any(r.get('agent_configuration', {}).get('memory') == 'full_visible_history' for r in records)
    assert any(r.get('reason') == 'opponent_changed' for r in records)


def test_unavailable_llm_does_not_destroy_current_session(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    game = HumanGame(seed=123)
    before = game.view()
    with pytest.raises(ValueError, match='GROQ_API_KEY'):
        game.change_opponent('llm')
    assert game.view() == before
