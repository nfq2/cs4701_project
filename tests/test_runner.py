from dataclasses import FrozenInstanceError, asdict
import io
import json

import pytest

from leduc.agents import RandomAgent
from leduc.interface import Agent
from leduc.runner import run_match


class RecordingAgent(Agent):
    def __init__(self, fold=False):
        self.fold = fold
        self.observations = []
        self.results = []
        self.starts = []

    def on_match_start(self, *, seed):
        self.observations.clear()
        self.results.clear()
        self.starts.clear()

    def on_hand_start(self, *, hand_index, seat):
        self.starts.append((hand_index, seat))

    def select_action(self, observation):
        self.observations.append(observation)
        if self.fold:
            return "fold"
        return "check" if "check" in observation.legal_actions else "call"

    def on_hand_end(self, result):
        self.results.append(result)


def records(seed=7, hands=20):
    log = io.StringIO()
    run_match((RandomAgent(), RandomAgent()), seed=seed, hands=hands, log=log)
    data = [json.loads(line) for line in log.getvalue().splitlines()]
    for row in data:
        row.pop("decision_seconds", None)
        if row["type"] == "match_summary":
            for stats in row["agents"].values():
                stats.pop("mean_decision_seconds")
    return data


def test_thousand_hands_legal_zero_sum_and_balanced():
    data = records(hands=1000)
    ends = [r for r in data if r["type"] == "hand_end"]
    assert len(ends) == 1000
    assert all(sum(r["payoffs_by_seat_chips"]) == 0 for r in ends)
    starts = [r for r in data if r["type"] == "hand_start"]
    for first, second in zip(starts[::2], starts[1::2]):
        assert first["deal_seed"] == second["deal_seed"]
        assert first["small_blind_seat"] == second["small_blind_seat"]
        assert first["seat_to_agent"] == list(reversed(second["seat_to_agent"]))
    for row in data:
        if row["type"] == "action":
            assert row["action"] in row["observation"]["legal_actions"]
    assert sum(a["total_chips"] for a in data[-1]["agents"].values()) == 0


def test_reproducibility():
    assert records() == records()
    assert records(7) != records(8)


def test_fold_privacy_payoff_units_and_lifecycle():
    agents = (RecordingAgent(fold=True), RecordingAgent(fold=True))
    run_match(agents, hands=10)
    for agent in agents:
        assert len(agent.starts) == len(agent.results) == 10
        assert {s for _, s in agent.starts} == {0, 1}
        for result in agent.results:
            assert result.opponent_revealed_card is None
            assert abs(result.payoff_chips) == 1  # one chip, not 0.5 big blinds
        for obs in agent.observations:
            assert obs.public_card is None
            assert obs.pot == 3
            assert obs.to_call == 1
            assert "opponent_revealed_card" not in asdict(obs)
            assert "hand_cards" not in asdict(obs)
            with pytest.raises(FrozenInstanceError):
                obs.private_card = "SK"


def test_showdown_feedback_and_history_snapshots():
    agents = (RecordingAgent(), RecordingAgent())
    run_match(agents, hands=10)
    for i, agent in enumerate(agents):
        for result, other in zip(agent.results, agents[1-i].results):
            assert result.opponent_revealed_card == other.private_card
            assert result.public_card is not None
            assert {e.round_index for e in result.history} == {0, 1}
    assert agents[0].observations[0].history != agents[0].results[0].history


def test_invalid_action_is_rejected():
    class BadAgent(Agent):
        def select_action(self, observation):
            return "not-an-action"

    with pytest.raises(ValueError, match="illegal action"):
        run_match((BadAgent(), BadAgent()), hands=2)


@pytest.mark.parametrize("hands", [0, -2, 3])
def test_invalid_hand_counts(hands):
    with pytest.raises(ValueError, match="positive even"):
        run_match((RandomAgent(), RandomAgent()), hands=hands)
