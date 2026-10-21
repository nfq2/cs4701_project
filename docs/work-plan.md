# Team work plan

## Current scope

Compare five agents: random, fixed Bayesian, supervised learning, Q-learning,
and LLM. Per the October 21 status report, heuristic and adaptive Bayesian
agents are removed to allow deeper evaluation and hyperparameter tuning.
RLCard's pretrained CFR agent is an external reference, not a sixth implemented
agent. Some older sections in the shared status report still describe the former
scope; this plan follows the explicit scope reduction.

## Implemented foundation

- Pinned RLCard Leduc environment, immutable player observations, legal-action
  validation, terminal feedback with folded opponent cards hidden.
- Fixed Bayesian agent with per-hand card beliefs and a fixed opponent model.
- Random baseline, paired deals with swapped seats, JSONL decisions/payoffs,
  timing, and integration checks.
- Local human-play UI, independent deals, alternating seats, and session logs.
- LLM prompt and OpenAI API adapter, structured action output, local validation,
  timeouts, request/token limits, seeded random fallback, usage/failure logs,
  and selection through match/UI command-line options. Offline tests pass;
  Groq GPT-OSS 120B passed live smoke tests for both memory variants; deeper
  model/prompt evaluation remains to be done.

## Tracks and handoffs

| Owner | Sequential deliverables |
| --- | --- |
| A: Bayesian | Card enumeration and priors; fixed opponent action likelihoods; posterior updates; expected chip-value estimates; playable fixed Bayesian agent; calibration and sensitivity analysis |
| B: Learning | CFR adapter and labeled data; game-level train/validation/test splits; supervised baseline; tabular Q-learning and self-play; checkpoints; learning-rate/discount-factor tuning; training/data-size curves |
| C: LLM and evaluation | Preliminary live model/prompt tests; round-robin harness and independent match seeds; payoff tables and match-level confidence intervals; latency/token/fallback analysis; human-study integration |

Each agent implements the shared `Agent` contract. Register completed agents in
`agents/factory.py` and the CLI choices so matches and human play use the same
implementations. Keep credentials out of agent configuration and metadata.

B supplies C with the CFR adapter and dataset schema. Keep complete games and
both halves of duplicate paired deals in the same dataset split. A documents
its fixed opponent model and likelihood-estimation data. All three review one
another's work, recruit participants, run human sessions, and write their own
experiment sections. Rebalance work based on actual progress.

## Evaluation checkpoints

1. Smoke-test each completed agent against random and verify information
   boundaries. Test the LLM with a small explicit request limit first.
2. Tune Bayesian assumptions, supervised hyperparameters, and Q-learning
   learning rate/discount factor on development seeds. Compare preliminary LLM
   models/prompts and their chip results, latency, token use, and fallback rates.
3. Freeze model snapshots, prompts, checkpoints, hyperparameters, and final seed
   lists before examining final results.
4. Run the five-agent round robin and CFR-reference comparisons. Use independent
   match seeds as replicates for confidence intervals; paired hands are correlated.
   Target about 20 matches of 1,000 hands where practical, with a separately
   reported smaller LLM sample and explicit API budget.
5. Run human-versus-agent sessions with outside participants, accounting for
   differences in poker experience and sample sizes.
6. Report chips per hand, match-level confidence intervals, decision latency,
   learning curves, hyperparameter sensitivity, and LLM usage/fallback rates.
   Clearly distinguish fallback actions from successful model decisions.

## Scope controls

Use standard RLCard Leduc with its existing deck and blinds. Defer expanded
decks, persistent stacks, elaborate frontend styling, and deep RL. The adaptive Bayesian agent remains out of scope. The LLM now has two
comparison variants: current-hand-only and full player-visible session history.
Use independent deals for memory comparisons and report each variant separately. Hosted-model
outputs may vary even with fixed game seeds; keep full experiment configuration
in logs and report that limitation.
