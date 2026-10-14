# Team work plan

## Shared foundation — chunks 1–3 implemented

1. Project setup: package, pinned RLCard version, setup instructions.
2. Shared contract: immutable player observations, legal actions, lifecycle hooks,
   and terminal feedback that hides folded cards.
3. Random baseline and runner: seeded paired deals, swapped seats, action
   validation, action/payoff logs, timing, and 1,000-hand integration checks.

All three people should run the quickstart and review the interface before
branching. Assign people by preference. Agree on interface changes together;
each track should own its modules to reduce merge conflicts.

## Three approximately balanced tracks

Effort points are estimates, not hours. Each track totals **13**. Bayesian and
learning work have more algorithm uncertainty; the LLM/UI track also owns shared
evaluation tooling and human-study integration. Revisit workload after the first
runnable agent from each track.

| Owner | Sequential small deliverables | Points |
| --- | --- | --- |
| A: Bayesian and opponent modeling | A1. Heuristic agent and configurable opponent styles | 2 |
| A | A2. Card enumeration, priors, posterior updates | 2 |
| A | A3. Fixed Bayesian action-value estimator and playable agent | 3 |
| A | A4. Adaptive style beliefs, cross-hand memory, match resets | 3 |
| A | A5. Adaptation experiments, diagnostics, analysis | 3 |
| B: Learning agents | B1. CFR reference adapter and labeled-game generation | 2 |
| B | B2. Game-level splits, features, supervised baseline | 3 |
| B | B3. Q-learning updates, state encoding, initial self-play | 3 |
| B | B4. Checkpoint saving/loading and training runs | 2 |
| B | B5. Training/data-size curves and analysis | 3 |
| C: LLM, evaluation, human interface | C1. Round-robin harness, independent seeds, payoff tables, match-level confidence intervals | 3 |
| C | C2. LLM prompt/API adapter, output validation, failures, cost limits | 3 |
| C | C3. Minimal human-play UI using the shared observation contract | 3 |
| C | C4. Human-session logging, integration, study setup | 2 |
| C | C5. Final tournament runs, latency/cost figures, analysis | 2 |

Each deliverable should be a separate reviewable commit or PR. Each owner writes
their experiment's report section. **All three** recruit participants, run human
sessions, review one another's work, and assemble the presentation. These shared
tasks are not left to C.

## Dependencies and handoffs

- A1, B1, and C1 start immediately in parallel.
- A2 → A3 → A4 → A5. A1 supplies controlled opponent styles. Start with a simple,
  documented action-value approximation before improving it.
- B1 → B2. B3 can start independently of the CFR dataset. B2/B3 → B4 → B5.
  Start with tabular Q-learning; add a neural network only if justified.
- C1 → C2 → C3 → C4. The UI can start with random opponents before the LLM is
  ready. C5 waits for the fixed agents and B's CFR adapter.
- B hands C a CFR adapter conforming to the shared agent interface and a dataset
  schema. Split by complete games; keep duplicate paired deals in the same split
  if generating data through the paired runner.
- A hands B/C opponent-style presets. Hold out some style parameters from
  development for the final adaptation experiment.
- C adds an agent-factory registry and evaluation configuration so A/B can
  register agents without editing tournament internals.
- A owns independent-hand adaptation experiments and reset rules. Coordinate
  runner changes with C: the foundation's paired deals are correlated, so final
  adaptation experiments need an explicit independent-deal protocol.

## Integration checkpoints

1. **First playable versions:** heuristic, supervised baseline, and LLM each
   complete a short match; C's harness evaluates random versus random.
2. **All agent families:** fixed Bayesian, Q-learning, adaptive Bayesian, and
   human UI work end-to-end. Check hidden-information boundaries and logging.
3. **Freeze:** select checkpoints/prompts/hyperparameters and final seed lists;
   agree on hand counts and LLM budget before viewing final outcomes.
4. **Final experiments:** each owner runs their experiments through common
   tooling; C coordinates the fixed-agent tournament. Everyone runs human sessions.
5. **Write-up:** combine results, limitations, reproducible commands, and figures.

## Scope controls

Use standard RLCard Leduc for the core project. Do not build a new poker engine.
Defer expanded decks, persistent stacks, elaborate frontend styling, and deep RL
until basic comparisons are complete. If B's training work grows, move checkpoint
tooling to C; if C's UI grows, keep it to a simple local interface and share human
session work. Rebalance from actual progress, not just these point estimates.
