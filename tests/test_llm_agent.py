from dataclasses import replace
import io
import json

import pytest

from leduc.agents import LLMAgent, RandomAgent
from leduc.agents.llm_agent import APIError, openai_completion
from leduc.interface import Observation
from leduc.runner import run_match
from leduc.ui import HumanGame

OBS = Observation(42, 0, 0, 'SJ', None, 0, (1, 2), 3, 1, ('call', 'raise', 'fold'), ())


def response(action='call'):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'action': action})}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 8}}


def test_prompt_privacy_schema_and_usage():
    requests = []
    def complete(payload):
        requests.append(payload)
        return response()
    agent = LLMAgent(model='test-model', completion=complete)
    assert agent.select_action(OBS) == 'call'
    visible = json.loads(requests[0]['messages'][1]['content'])
    assert 'hand_index' not in visible
    assert set(visible) == set(OBS.__dataclass_fields__) - {'hand_index'}
    assert visible['private_card'] == 'SJ'
    schema = requests[0]['response_format']['json_schema']['schema']
    assert schema['properties']['action']['enum'] == list(OBS.legal_actions)
    assert agent.statistics() == dict(api_calls=1, fallbacks=0, prompt_tokens=100, completion_tokens=8)
    assert 'api_key' not in json.dumps(agent.configuration())


@pytest.mark.parametrize('bad', [None, {}, {'choices': []}, response('check'), response([]),
    {'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]},
    {'choices': [{'finish_reason': 'stop', 'message': {'content': 'raise'}}]},
    {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"action":"call","extra":1}'}}]}])
def test_bad_responses_fallback_legally(bad):
    agent = LLMAgent(model='test', completion=lambda _: bad)
    assert agent.select_action(OBS) in OBS.legal_actions
    assert agent.decision_metadata()['fallback']
    assert agent.statistics()['fallbacks'] == 1


def test_failure_call_cap_and_match_reset():
    def fail(_):
        raise APIError('connection_or_timeout')
    agent = LLMAgent(model='test', completion=fail, max_calls=1)
    agent.on_match_start(seed=5)
    first = agent.select_action(OBS)
    assert agent.decision_metadata()['failure_reason'] == 'connection_or_timeout'
    agent.select_action(OBS)
    assert agent.decision_metadata()['failure_reason'] == 'call_limit'
    assert agent.calls == 1
    agent.on_match_start(seed=5)
    assert agent.select_action(OBS) == first
    with pytest.raises(ValueError, match='terminal'):
        agent.select_action(replace(OBS, legal_actions=()))
    assert agent.calls == 1


def test_transport_payload_and_sanitized_failures(monkeypatch):
    from urllib.error import HTTPError
    import leduc.agents.llm_agent as module
    def reject(request, timeout):
        assert request.full_url == 'https://api.openai.com/v1/chat/completions'
        assert request.get_header('Authorization') == 'Bearer fake-key'
        assert json.loads(request.data) == {'model': 'test'}
        assert timeout == 2
        raise HTTPError(request.full_url, 401, 'sensitive provider body', {}, None)
    monkeypatch.setattr(module, 'urlopen', reject)
    with pytest.raises(APIError, match='^http_401$'):
        openai_completion({'model': 'test'}, api_key='fake-key', timeout=2)


def test_configuration_errors(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        LLMAgent(model='test')
    for kwargs in ({'model': ''}, {'model': 'test', 'max_calls': -1},
                   {'model': 'test', 'timeout': float('nan')}, {'model': 'test', 'max_completion_tokens': 0}):
        with pytest.raises(ValueError):
            LLMAgent(completion=lambda _: response(), **kwargs)


def legal_completion(payload):
    obs = json.loads(payload['messages'][1]['content'])
    return response('check' if 'check' in obs['legal_actions'] else 'call')


def test_match_and_human_integration():
    agent = LLMAgent(model='test', completion=legal_completion, max_calls=1000)
    log = io.StringIO()
    summary = run_match((agent, RandomAgent()), hands=20, log=log)
    assert summary['agents']['agent_a']['fallbacks'] == 0
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert records[0]['agent_configurations'][0]['model'] == 'test'
    assert any(r.get('agent_metadata', {}).get('prompt_tokens') == 100 for r in records)
    log = io.StringIO()
    game = HumanGame(seed=5, agent=agent, opponent='llm', log=log)
    for _ in range(10):
        while not game.view()['over']:
            assert game.view()['opponent_card'] is None
            legal = game.view()['legal_actions']
            game.act('check' if 'check' in legal else 'call')
        assert game.view()['opponent_name'] == 'LLM without memory'
        game.next_hand()
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert any(r.get('actor') == 'llm' for r in records)
    assert agent.fallbacks == 0


def test_groq_provider_uses_own_key_endpoint_and_reasoning(monkeypatch):
    import leduc.agents.llm_agent as module
    monkeypatch.setenv('OPENAI_API_KEY', 'wrong-provider-key')
    monkeypatch.setenv('GROQ_API_KEY', 'fake-groq-key')
    def complete(request, timeout):
        assert request.full_url == 'https://api.groq.com/openai/v1/chat/completions'
        assert request.get_header('Authorization') == 'Bearer fake-groq-key'
        assert request.get_header('User-agent') == 'leduc-agents/0.1'
        payload = json.loads(request.data)
        assert payload['reasoning_effort'] == 'low'
        assert payload['max_completion_tokens'] == 2048
        assert payload['response_format']['json_schema']['strict'] is True
        return io.StringIO(json.dumps(response()))
    monkeypatch.setattr(module, 'urlopen', complete)
    agent = LLMAgent(model='openai/gpt-oss-120b', provider='groq',
                     reasoning_effort='low', max_completion_tokens=2048)
    assert agent.select_action(OBS) == 'call'
    assert agent.configuration()['provider'] == 'groq'
    assert 'fake-groq-key' not in json.dumps(agent.configuration())
    monkeypatch.delenv('GROQ_API_KEY')
    with pytest.raises(ValueError, match='GROQ_API_KEY'):
        LLMAgent(model='openai/gpt-oss-120b', provider='groq')


def test_full_memory_privacy_reset_and_no_memory_comparison():
    from leduc.agents import MemoryLLMAgent
    from leduc.interface import ActionEvent, HandResult
    requests = []
    def complete(payload):
        requests.append(json.loads(payload['messages'][1]['content']))
        return legal_completion(payload)
    memory = MemoryLLMAgent(model='test', completion=complete)
    plain = LLMAgent(model='test', completion=complete)
    folded = HandResult(0, 0, 'SJ', None, None, -1, (ActionEvent(0, 0, 'fold'),))
    showdown = HandResult(1, 1, 'HQ', 'SQ', 'SK', 4,
                          (ActionEvent(1, 1, 'call'),))
    for agent in (memory, plain):
        agent.on_match_start(seed=1)
        agent.on_hand_start(hand_index=0, seat=0)
        agent.select_action(OBS)
        agent.on_hand_end(folded)
        agent.on_hand_start(hand_index=1, seat=1)
        agent.on_hand_end(showdown)  # Retain outcomes even without an own decision.
        agent.on_hand_start(hand_index=2, seat=0)
        agent.select_action(OBS)
    remembered = requests[1]['previous_hands']
    assert len(remembered) == 2
    assert remembered[0]['result']['opponent_revealed_card'] is None
    assert remembered[1]['result']['opponent_revealed_card'] == 'SK'
    assert remembered[1]['result']['seat'] == 1
    assert remembered[0]['observations_at_own_turns'][0]['private_card'] == 'SJ'
    assert 'hand_index' not in json.dumps(remembered)
    assert 'previous_hands' not in requests[-1]
    assert memory.decision_metadata()['memory_hands'] == 2
    memory.on_match_start(seed=2)
    memory.select_action(OBS)
    assert requests[-1]['previous_hands'] == []


def test_memory_matches_use_independent_deals():
    from leduc.agents import MemoryLLMAgent
    agent = MemoryLLMAgent(model='test', completion=legal_completion)
    log = io.StringIO()
    run_match((agent, RandomAgent()), hands=10, log=log)
    events = [json.loads(line) for line in log.getvalue().splitlines()]
    assert events[0]['seat_policy'] == 'independent_deals_alternating_seats'
    starts = [e for e in events if e['type'] == 'hand_start']
    assert len({e['deal_seed'] for e in starts}) == 10
    assert all(e['pair_index'] is None for e in starts)
    assert starts[0]['seat_to_agent'] == list(reversed(starts[1]['seat_to_agent']))
    with pytest.raises(ValueError, match='independent'):
        run_match((agent, RandomAgent()), hands=2, deal_policy='paired')
