"""Generate independent simulated games, fit likelihoods, and score held-out games.

Style policies below are synthetic data generators, not the deployed Bayesian
likelihood model and not additional selectable research agents.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import rlcard

from leduc.agents.bayesian.opponent_model import ACTIONS, SCHEMA_VERSION, OpponentModel, context_keys
from leduc.interface import ActionEvent
from leduc.runner import make_observation

# (raise tendency, fold tendency, call tendency, weak-hand bluff tendency)
STYLES = {
    'tight': (.7, 2.5, .7, .02),
    'aggressive': (3., .3, .8, .35),
    'bluff-heavy': (1.2, .6, .8, 1.8),
    'calling-heavy': (.35, .15, 4., .03),
    'balanced': (1.2, 1., 1.2, .15),
}


def style_action(style, observation, rng):
    raise_scale, fold_scale, call_scale, bluff = STYLES[style]
    rank = 'JQK'.index(observation.private_card[1])
    strength = (rank + 1) / 4 if observation.public_card is None else (
        .95 if observation.private_card[1] == observation.public_card[1] else (rank + 1) / 6)
    # Pressure makes folding weak hands more likely; every action has support.
    pressure = 1 + observation.to_call / (2 if observation.round_index == 0 else 4)
    weights = {
        'raise': .03 + raise_scale * strength ** 2 + bluff * (1 - strength),
        'fold': .03 + fold_scale * (1 - strength) ** 2 * pressure,
        'call': .03 + call_scale * (.2 + strength),
        'check': 1.,
    }
    return rng.choices(observation.legal_actions,
                       weights=[weights[a] for a in observation.legal_actions])[0]


def generate_records(hands_per_style, seed):
    """All decisions from a focal style, against rotating styles and random.

    Each style gets equal hands, alternate seats, and independent deals. Games
    are never reused across training and held-out streams. Only focal decisions
    become examples; the label never uses the other player's private card.
    """
    rng = random.Random(seed)
    env = rlcard.make('leduc-holdem', config={'seed': seed % 2**32})
    opponents = (*STYLES, 'random')
    for style in STYLES:
        for hand in range(hands_per_style):
            deal_seed = rng.getrandbits(32)
            env.seed(deal_seed)
            state, seat = env.reset()
            small = seat
            focal = hand % 2
            other_style = opponents[hand % len(opponents)]
            history = []
            while not env.is_over():
                obs = make_observation(state, hand_index=hand, seat=seat,
                                       small_blind_seat=small, history=history)
                acting_style = style if seat == focal else other_style
                action = (rng.choice(obs.legal_actions) if acting_style == 'random'
                          else style_action(acting_style, obs, rng))
                if seat == focal:
                    yield {'game_id': f'{seed}:{style}:{hand}', 'style': style,
                           'deal_seed': deal_seed, 'opponent_style': other_style,
                           'observation': asdict(obs), 'action': action}
                history.append(ActionEvent(seat, obs.round_index, action))
                state, seat = env.step(action, raw_action=True)


def features(record):
    obs = record['observation']
    raises = sum(e['action'] == 'raise' and e['round_index'] == obs['round_index']
                 for e in obs['history'])
    return (obs['private_card'], obs['public_card'], tuple(obs['legal_actions']),
            obs['to_call'], raises)


def fit(records, metadata=None, alpha=1.):
    tables = [{}, {}, {}]
    decisions = Counter()
    for record in records:
        card, public, legal, to_call, raises = features(record)
        if record['action'] not in legal:
            raise ValueError('Training record contains illegal action')
        for table, key in zip(tables, context_keys(card, public, legal, to_call, raises)):
            counts = table.setdefault(key, [0] * len(ACTIONS))
            counts[ACTIONS.index(record['action'])] += 1
        decisions[record['style']] += 1
    if not decisions:
        raise ValueError('No training decisions')
    return {'schema_version': SCHEMA_VERSION, 'alpha': alpha,
            'metadata': {**(metadata or {}), 'decisions_by_style': dict(decisions),
                         'mixture': 'Pooled decisions from equal hands per style; no online style inference',
                         'features': ['private_rank', 'public_rank', 'legal_actions', 'to_call', 'raises'],
                         'backoff': ['full context', 'card/public/legal', 'legal', 'uniform'],
                         'styles': {k: list(v) for k, v in STYLES.items()}},
            'tables': tables}


def evaluate(model, records):
    loss = uniform_loss = 0.
    count = correct = 0
    levels = Counter()
    for record in records:
        args = features(record)
        probs = model.probabilities(*args)
        loss -= math.log(probs[record['action']])
        uniform_loss += math.log(len(probs))
        correct += max(probs, key=probs.get) == record['action']
        count += 1
        keys = context_keys(*args)
        level = next((i for i, (table, key) in enumerate(zip(model.tables, keys)) if key in table), 3)
        levels[str(level)] += 1
    return {'decisions': count, 'negative_log_likelihood': loss / count,
            'uniform_negative_log_likelihood': uniform_loss / count,
            'action_accuracy': correct / count, 'backoff_levels': dict(levels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands-per-style', type=int, default=5000)
    parser.add_argument('--heldout-hands-per-style', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=4701)
    parser.add_argument('--heldout-seed', type=int, default=94701)
    parser.add_argument('--output', type=Path, default=Path('src/leduc/agents/bayesian/models/bayesian-v1.json'))
    parser.add_argument('--data', type=Path, default=Path('results/bayesian-training.jsonl'))
    args = parser.parse_args()
    if min(args.hands_per_style, args.heldout_hands_per_style) <= 0 or min(args.seed, args.heldout_seed) < 0:
        parser.error('Hand counts must be positive and seeds nonnegative')
    if args.seed == args.heldout_seed:
        parser.error('Training and held-out seeds must differ')
    args.data.parent.mkdir(parents=True, exist_ok=True)
    # Preserve old datasets; choose a new --data path for another run.
    with args.data.open('x', encoding='utf-8') as log:
        def recorded():
            for row in generate_records(args.hands_per_style, args.seed):
                log.write(json.dumps(row, sort_keys=True) + '\n')
                yield row
        payload = fit(recorded(), metadata={
            'training_seed': args.seed, 'heldout_seed': args.heldout_seed,
            'hands_per_style': args.hands_per_style,
            'heldout_hands_per_style': args.heldout_hands_per_style,
            'rlcard_version': '1.2.0', 'deal_policy': 'independent',
            'generator_version': 1,
        })
    model = OpponentModel(payload)
    metrics = evaluate(model, generate_records(args.heldout_hands_per_style, args.heldout_seed))
    payload['metadata']['heldout_metrics'] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print(f'Frozen model: {args.output}; training decisions: {args.data}')


if __name__ == '__main__':
    main()
