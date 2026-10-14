"""Seeded matches, paired seat assignments, and JSONL experiment records."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
from time import perf_counter
from typing import TextIO

import rlcard

from leduc.agents import RandomAgent
from leduc.interface import ActionEvent, Agent, HandResult, Observation

BIG_BLIND_CHIPS = 2


def make_observation(state, *, hand_index, seat, small_blind_seat, history):
    raw = state["raw_obs"]
    contributions = tuple(int(value) for value in raw["all_chips"])
    return Observation(
        hand_index=hand_index,
        seat=seat,
        small_blind_seat=small_blind_seat,
        private_card=raw["hand"],
        public_card=raw["public_card"],
        round_index=int(raw["public_card"] is not None),
        contributions=contributions,
        pot=sum(contributions),
        to_call=max(contributions) - contributions[seat],
        legal_actions=tuple(state["raw_legal_actions"]),
        history=tuple(history),
    )


def run_match(
    agents: tuple[Agent, Agent],
    *,
    hands: int = 1000,
    seed: int = 0,
    names: tuple[str, str] = ("agent_a", "agent_b"),
    log: TextIO | None = None,
) -> dict:
    """Run independent hands, preserving agent memory within a match.

    Consecutive hands replay an initial deal with swapped agent seats. Thus each
    agent gets both blind positions and private cards in every pair. Pair outcomes
    are correlated; use independent matches for later confidence intervals.
    """
    if hands <= 0 or hands % 2:
        raise ValueError("hands must be a positive even number for seat-balanced pairs")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if len(agents) != 2 or agents[0] is agents[1]:
        raise ValueError("Provide two separate agent instances")
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError("Provide two distinct agent names")

    def emit(record):
        if log is not None:
            log.write(json.dumps(record, sort_keys=True) + "\n")

    rng = random.Random(seed)
    agent_seeds = [rng.getrandbits(32) for _ in agents]
    for agent, agent_seed in zip(agents, agent_seeds):
        agent.on_match_start(seed=agent_seed)
    env = rlcard.make("leduc-holdem", config={"seed": seed % (2**32)})
    emit({"type": "match_start", "schema_version": 1, "seed": seed,
          "hands": hands, "agents": names, "agent_seeds": agent_seeds,
          "rlcard_version": "1.2.0", "payoff_unit": "chips",
          "small_blind": 1, "big_blind": BIG_BLIND_CHIPS,
          "seat_policy": "paired_initial_deals_swapped_agents"})
    totals = [0.0, 0.0]
    decisions = [0, 0]
    seconds = [0.0, 0.0]
    for hand_index in range(hands):
        if hand_index % 2 == 0:
            deal_seed = rng.getrandbits(32)
        env.seed(deal_seed)
        state, seat = env.reset()
        small_blind_seat = seat
        seat_to_agent = (0, 1) if hand_index % 2 == 0 else (1, 0)
        history = []
        for player_seat, agent_index in enumerate(seat_to_agent):
            agents[agent_index].on_hand_start(hand_index=hand_index, seat=player_seat)
        emit({"type": "hand_start", "hand_index": hand_index,
              "pair_index": hand_index // 2, "deal_seed": deal_seed,
              "seat_to_agent": seat_to_agent, "small_blind_seat": small_blind_seat})
        while not env.is_over():
            observation = make_observation(
                state, hand_index=hand_index, seat=seat,
                small_blind_seat=small_blind_seat, history=history,
            )
            agent_index = seat_to_agent[seat]
            start = perf_counter()
            action = agents[agent_index].select_action(observation)
            elapsed = perf_counter() - start
            # RLCard's integer decoder silently replaces illegal actions. Reject
            # them ourselves and use raw actions so agent bugs cannot be hidden.
            if action not in observation.legal_actions:
                raise ValueError(f"{names[agent_index]} returned illegal action {action!r}; "
                                 f"legal actions: {observation.legal_actions}")
            decisions[agent_index] += 1
            seconds[agent_index] += elapsed
            emit({"type": "action", "agent_index": agent_index,
                  "observation": asdict(observation), "action": action,
                  "decision_seconds": elapsed})
            history.append(ActionEvent(seat, observation.round_index, action))
            state, seat = env.step(action, raw_action=True)

        # RLCard reports big-blind units; the project's metric is actual chips.
        payoffs = [float(value) * BIG_BLIND_CHIPS for value in env.get_payoffs()]
        showdown = history[-1].action != "fold"
        terminal = [env.get_state(s)["raw_obs"] for s in (0, 1)]
        for player_seat, agent_index in enumerate(seat_to_agent):
            totals[agent_index] += payoffs[player_seat]
            agents[agent_index].on_hand_end(HandResult(
                hand_index=hand_index, seat=player_seat,
                private_card=terminal[player_seat]["hand"],
                public_card=terminal[player_seat]["public_card"],
                opponent_revealed_card=terminal[1-player_seat]["hand"] if showdown else None,
                payoff_chips=payoffs[player_seat], history=tuple(history),
            ))
        emit({"type": "hand_end", "hand_index": hand_index,
              "payoffs_by_seat_chips": payoffs, "showdown": showdown,
              "public_card": terminal[0]["public_card"],
              "revealed_cards_by_seat": [s["hand"] for s in terminal] if showdown else None,
              "history": [asdict(event) for event in history]})
    summary = {"type": "match_summary", "seed": seed, "hands": hands,
               "agents": {name: {
                   "total_chips": totals[i], "chips_per_hand": totals[i] / hands,
                   "decisions": decisions[i],
                   "mean_decision_seconds": seconds[i] / decisions[i] if decisions[i] else 0.0,
               } for i, name in enumerate(names)}}
    emit(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=1000, help="Positive even number")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        help="New log path (default: unique file in results/)")
    args = parser.parse_args()
    if args.hands <= 0 or args.hands % 2 or args.seed < 0:
        parser.error("--hands must be positive and even; --seed must be nonnegative")
    if args.output is None:
        Path("results").mkdir(parents=True, exist_ok=True)
        log_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir="results",
            prefix=f"random-seed-{args.seed}-", suffix=".jsonl", delete=False,
        )
        output = Path(log_file.name)
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_file = output.open("x", encoding="utf-8")
        except FileExistsError:
            parser.error(f"Output already exists: {output}. Choose a new --output "
                         "path or omit --output for an automatic filename.")
    with log_file as log:
        summary = run_match((RandomAgent(), RandomAgent()), hands=args.hands,
                            seed=args.seed, names=("random_a", "random_b"), log=log)
    print(json.dumps(summary, indent=2))
    print(f"Log: {output}")


if __name__ == "__main__":
    main()
