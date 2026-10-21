# CS 4701: Leduc Hold'em agents

Compare seven agents: random, heuristic, fixed Bayesian, adaptive Bayesian,
supervised learning, Q-learning, and LLM. The shared foundation, random baseline, and fixed Bayesian agent are implemented. See [the team work plan](docs/work-plan.md) for the rest.

## Setup

Python 3.10+ is required (tested locally with Python 3.12).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m leduc.runner --hands 1000 --seed 42 --output results/random-42.jsonl
```

After installation, `leduc-match` is equivalent. Omit `--output` to automatically
create a unique log in `results/` on every run. Explicit output paths must be new
files; existing logs are preserved. Generated results are ignored by Git.
`requirements-dev.lock` records the dependencies used for foundation verification;
optionally install it before the editable package to reproduce those versions.

## Play in your browser

```sh
source .venv/bin/activate
python -m leduc.ui
```

Open **http://127.0.0.1:8000**. Keep the terminal running; press Ctrl+C to stop.
If that port is busy, use `python -m leduc.ui --port 8001` and open that port.
No additional dependencies or frontend build are required. After reinstalling
the editable package, `leduc-play` is also available as a shortcut.

Play against the random agent using Fold, Check, Call, and Raise. The interface
shows your private card, the shared card, the pot, legal actions, action history,
and cumulative net chips. Opponent cards are sent to the browser only at showdown,
and stay hidden after folds. Click **Deal next hand** after each result. **How to
play** explains the rules. This is a local single-session UI: multiple tabs share
the same game, and restarting the server starts a fresh score.

Human sessions use independent deals, alternate player seats, and save decisions
and outcomes to unique `results/human-*.jsonl` files. These research logs contain
both players' observations and are not served to the browser. Add `--seed 123`
to reproduce a session with the same human choices. Unlike the tournament runner,
the human interface does not repeat deals in pairs.

## Fixed game rules

Use **RLCard 1.2.0's `leduc-holdem`**: two players, six cards (two each of J, Q, K),
one private card each, one public card, and two betting rounds. Small/big blinds
are 1/2 chips; raises are 2 chips before the public card and 4 afterward, with at
most two raises per round. A matching pair beats a high card; equal ranks split
the pot. This is RLCard's blind-based variant, with fresh contributions each hand
and no persistent bankroll or stack limit. Alternate decks/stacks are deferred.

The adapter converts RLCard's big-blind-normalized payoff to **net chips**.
See the [RLCard documentation](https://rlcard.org/games.html#leduc-hold-em) and the
pinned implementation for environment details.

## Adding an agent

Subclass `Agent` in `src/leduc/interface.py` and implement
`select_action(observation)`, returning one of its legal action strings:
`call`, `raise`, `fold`, or `check`. The runner rejects illegal actions.

Immutable observations contain the acting player's card and public information:
public card, seat, small-blind seat, round (0 or 1), contributions by seat, pot,
amount to call, legal actions, and betting history. No environment object,
opponent card, deal seed, or future card is exposed. Seat indexes and agent
indexes differ: assignments swap each hand.

Lifecycle hooks:

- `on_match_start(seed=...)`: reset opponent memory and seed a local RNG.
- `on_hand_start(hand_index=..., seat=...)`: clear hand-local state, preserving
  opponent memory and learned model parameters.
- `on_hand_end(result)`: receive own card, public card, complete public action
  history, own chip payoff, and the opponent card **only at showdown**. Training
  agents can retain their own observations to form learning trajectories.

Use separate agent instances, even for self-play:

```python
from leduc.agents import RandomAgent
from leduc.runner import run_match

summary = run_match(
    (RandomAgent(), RandomAgent()),
    names=("random_a", "random_b"), hands=1000, seed=42,
)
```

## Match protocol and logging

Each consecutive pair reuses the initial-deal seed with agents swapped across
seats. Both agents get both private cards and blind positions within each pair;
action trajectories can differ. Use a positive even number of hands. Agent RNGs
are independent of the environment RNG. Identical match seeds and deterministic
agents reproduce gameplay; measured times vary. Adaptive agents must not exploit
hand numbering or repeated-deal structure to infer hidden cards. Final adaptation
experiments should use an independent-deal protocol to avoid this correlation.

JSONL records contain match metadata, hand/deal seeds, seat assignments, each
decision's observation/action/latency, hand outcomes, and the summary. These are
**offline research logs** containing observations from both players; never feed
the combined log to an agent or human opponent during play.

Summaries report total chips, chips per hand, and mean decision time per agent.
Confidence intervals and multi-match tournaments belong to the next phase; use
independent match seeds, not correlated paired hands, as replicates.

## Layout

```text
src/leduc/interface.py           Observations, feedback, agent contract
src/leduc/agents/random_agent.py Uniform legal-action baseline
src/leduc/runner.py              RLCard adapter, paired matches, CLI, JSONL
src/leduc/ui.py                  Local human-versus-agent server
src/leduc/web/                   Browser interface (HTML, CSS, JavaScript)
tests/test_runner.py             Integration and information-boundary checks
tests/test_ui.py                 Human gameplay, scoring, and card privacy
docs/work-plan.md                Balanced ownership and sequential milestones
```

## Fixed Bayesian agent

Select **Bayesian agent** in the browser to start playing against it immediately.
Changing opponents starts a fresh session. Restart creates a new agent instance;
card beliefs also reset every hand. No opponent tendencies are learned across hands.

The agent enumerates the five other physical cards, conditions on the public
card, and applies Bayes' rule to the opponent's observed actions. It reconstructs
beliefs from the public history each turn so no action is counted twice. Its
fixed action-likelihood model uses smoothed hand-strength weights, allowing
bluffing and weak decisions; these are explicit assumptions, not trained values.

It searches the remaining betting tree with the two-raise cap, marginalizes over
future public cards and opponent responses, and maximizes expected net chips.
Future decisions use updated card distributions, not knowledge of a sampled
opponent card. The model follows RLCard 1.2.0's fixed blind/raise rules.
Performance depends on the assumed opponent policy; this is not an equilibrium
solver or the adaptive Bayesian agent. Likelihood weights live in
`src/leduc/agents/bayesian_agent.py`.
