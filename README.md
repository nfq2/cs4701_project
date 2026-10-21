# CS 4701: Leduc Hold'em agents

Compare five agent families: random, fixed Bayesian, supervised learning,
Q-learning, and LLM. The shared foundation, random baseline, fixed Bayesian agent,
and LLM variants (without memory and with full visible history) are implemented.
See [the team work plan](docs/work-plan.md) for the rest.

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

By default, play against the random agent using Fold, Check, Call, and Raise.
Use `--opponent llm --llm-model MODEL_ID` to play against the LLM (setup below). The interface
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

## LLM agent

### Choose an opponent in the browser

```sh
source .venv/bin/activate
source .env
python -m leduc.ui
```

Open http://127.0.0.1:8000. Choose **Bayesian agent**, **Random agent**, **LLM without memory**, or
**LLM with full memory** from the agent dropdown. Selecting a different opponent
starts a fresh session; **Restart session** resets the current opponent. This discards the
current hand and resets the score, memory, and per-session request counter.
The browser UI defaults to Groq GPT-OSS 120B, low reasoning effort, and a 2,048
completion-token cap. You can override these with the LLM command-line options.
The initial opponent is random, so starting the server makes no model calls.
Both LLM choices use the same configured model and limits.

- `llm`: sees its own card, public information, and all current-hand betting
  actions. No previous-hand information is sent.
- `llm-memory`: also receives every completed hand in the current session in
  chronological order: its own observations, actions by both players, its card,
  public card, net chips won/lost, and the opponent card only if shown at showdown.
  It sees the same kind of information a human opponent could remember. It never
  receives folded opponent cards, the deck, or future cards. Each past hand
  identifies its seat because positions can change.

Full history is resent without summaries or truncation; it resets between
sessions/matches. Long sessions can hit provider context/token limits. Those
errors are logged and use the existing random fallback, as do exhausted call
budgets. Full memory is not unlimited API capacity.

For automated comparisons, use `--agent-a llm --agent-b llm-memory` with your
provider/model options. Any match involving a memory agent automatically uses
independent deals with alternating seats, avoiding hidden-card clues from the
runner's repeated paired deals. Use `--deal-policy independent` for other
matchups when comparing them with memory-agent results.

### Groq hosted model (current tested setup)

Use `openai/gpt-oss-120b` hosted by Groq; no local model download or GPU is needed.
Set `GROQ_API_KEY` in your shell, or load the ignored local `.env` if configured.
Groq offers a limited free tier; check your account limits before large runs.
The OpenAI provider remains available separately and is the match CLI default;
the browser UI defaults to Groq.

```sh
source .venv/bin/activate
source .env
python -m leduc.ui --opponent llm --llm-provider groq --llm-model openai/gpt-oss-120b --llm-reasoning-effort low --llm-max-tokens 2048
```

Open http://127.0.0.1:8000. For an automated smoke test:

```sh
python -m leduc.runner --agent-a llm --agent-b random --llm-provider groq --llm-model openai/gpt-oss-120b --llm-reasoning-effort low --llm-max-tokens 2048 --hands 2 --seed 42 --llm-max-calls 8
```

The higher token cap allows room for reasoning before the final action. Only the
final action is parsed; reasoning content is not logged or passed to later turns.
The live two-hand test on October 21 returned five valid model decisions and no
fallbacks. This checks connectivity and action handling, not strategic strength.
Rate-limit failures still use the explicitly logged random fallback; allow time
between free-tier runs and check fallback counts before interpreting results.

### Adapter and OpenAI setup

The hosted model receives the fixed Leduc rules plus a JSON description of its
current observation. Structured output restricts the response to legal actions;
the adapter also validates the response locally. No prior model conversations, hidden
opponent cards, deal seeds, or hand numbers are sent. The memory variant includes
only its player-visible records from previous hands. Model and prompt stay fixed within
a run. Choose a model supporting OpenAI Chat Completions with strict JSON schema
output; see the [API reference](https://developers.openai.com/api/reference/resources/chat).

Set `OPENAI_API_KEY` in your shell using your usual secret-management method.
The key is read only by the Python server, never sent to the browser or logs.
No extra Python dependencies are needed. Replace `MODEL_ID` below with your
chosen model ID (prefer a versioned snapshot for final experiments):

```sh
source .venv/bin/activate
python -m leduc.runner --agent-a llm --agent-b random --llm-model MODEL_ID --hands 10 --seed 42 --llm-max-calls 50
python -m leduc.ui --opponent llm --llm-model MODEL_ID --llm-max-calls 100
```

Each LLM instance allows 100 API requests per match/session by default, with
no retries, a 20-second network timeout, and a 256 completion-token cap per
request. Configure these with `--llm-max-calls`, `--llm-timeout`, and
`--llm-max-tokens`. Reasoning models may need a larger completion-token cap.
These are request/token limits, not a dollar budget. Calls can incur charges;
usage returned by successful API responses is logged, but timed-out requests
may still be billed. Separate LLM agents have separate limits.

Malformed/illegal responses, refusals, API errors, and exhausted call budgets
fall back to a seeded uniform random legal action. Logs record every fallback
and its reason; match summaries include API calls, fallbacks, and reported token
usage. Report fallback rates with chip results: a run with fallbacks measures a
mixed LLM/random policy. Check a small run before conducting large experiments.
Session logs include cumulative API statistics at each hand end. The browser
waits while the model responds; credentials and API diagnostics stay server-side.

The match/session metadata records the model, full prompt, prompt hash, and
limits. Fixed seeds reproduce deals and fallback choices; hosted model outputs
are not guaranteed to be reproducible. Tests use a simulated completion backend
and do not spend API credits.

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
agents reproduce gameplay; measured times vary. Agents must not exploit
hand numbering or repeated-deal structure to infer hidden cards. LLM prompts
omit hand numbering. Matches with the full-memory variant use independent deals
instead of repeated pairs; other matches can opt in with `--deal-policy independent`.

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
src/leduc/agents/llm_agent.py    LLM prompt, API adapter, validation and limits
src/leduc/agents/factory.py      Shared CLI agent selection
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
fixed action-likelihood model is fitted from simulated tight, aggressive,
bluff-heavy, calling-heavy, and balanced opponents. These are synthetic policies,
not human data. The deployed agent reads frozen action counts; it does not call
the style policies or fit anything during play.

It searches the remaining betting tree with the two-raise cap, marginalizes over
future public cards and opponent responses, and maximizes expected net chips.
Future decisions use updated card distributions, not knowledge of a sampled
opponent card. The model follows RLCard 1.2.0's fixed blind/raise rules.
Performance depends on the assumed opponent policy; this is not an equilibrium
solver or the adaptive Bayesian agent. The bundled snapshot lives in
`src/leduc/agents/bayesian/models/bayesian-v1.json` and is included in installed packages.

### Reproducing Bayesian likelihood estimation

```sh
python -m leduc.agents.bayesian.train --hands-per-style 5000 --heldout-hands-per-style 1000 --seed 4701 --heldout-seed 94701 --data results/bayesian-training-rerun.jsonl --output results/bayesian-model-rerun.json
```

This generates 25,000 independent training hands (5,000 per focal style),
alternates focal seats, and rotates opponents among the five styles and random.
Each record contains the focal player's own card, public context, chosen action,
and game ID. These are offline research records and are never exposed to live
players. Existing datasets are preserved; use a new `--data` path for each run.

The model estimates `P(action | private rank, public rank, legal actions,
amount to call, raises this round)` from counts with Laplace smoothing (alpha 1).
Sparse/unseen contexts back off to card/public/legal, then legal-action counts,
then uniform legal actions. Suits are equivalent for hand strength. Counts pool
decisions from equal hands per style, so styles producing longer hands contribute
more decisions; this is a fixed empirical mixture, not online style inference.
The model omits richer history and position features; it is an approximation.

Training uses seed 4701; held-out scoring uses independent games with seed 94701.
The snapshot records generator settings, style parameters, decision counts, and
held-out metrics. Log loss measures action-probability quality; it is separate
from chip payoff. Neither generated style parameters nor the snapshot are tuned
on final human evaluation. Synthetic policies remain assumptions, and performance
against these styles does not establish performance against humans.

`BayesianAgent(model_path="...")` can load another frozen snapshot. Change the
bundled snapshot only through an explicit offline training run, and freeze it
before final evaluation. Model metadata is included in match/session logs.
