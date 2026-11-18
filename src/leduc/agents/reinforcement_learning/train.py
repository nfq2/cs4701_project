"""Train the Q-learning agent by self-play and freeze the learned Q-table.

Two QLearningAgent instances share one Q-table and play each other through the
standard runner, so both seats' experience updates the same estimates. 

Exploration decays linearly from --epsilon-start to --epsilon-end. 
"""
import argparse
import json
from pathlib import Path
import random

from leduc.agents.random_agent import RandomAgent
from leduc.agents.reinforcement_learning.agent import QLearningAgent, QTable
from leduc.runner import run_match

DEFAULT_OUTPUT = Path('src/leduc/agents/reinforcement_learning/models/qlearning-v1.json')


def evaluate(table, opponent, hands, seed):
    """Greedy play from current table"""
    frozen = QLearningAgent(table=table, training=False, epsilon=0.0)
    summary = run_match((frozen, opponent), hands=hands, seed=seed, names=('qlearning', 'opponent'))
    return summary['agents']['qlearning']['chips_per_hand']


def train(*, hands, chunk, alpha, gamma, epsilon_start, epsilon_end, seed,
          eval_hands, eval_seed, verbose=True):
    if hands <= 0 or chunk <= 0 or chunk % 2 or hands % chunk:
        raise ValueError('hands must be a positive multiple of an even chunk size')
    table = QTable()
    learners = tuple(QLearningAgent(table=table, training=True, alpha=alpha, gamma=gamma)
                     for _ in range(2))
    rng = random.Random(seed)
    curve = []
    chunks = hands // chunk
    for i in range(chunks):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * i / max(chunks - 1, 1)
        for learner in learners:
            learner.epsilon = epsilon
        run_match(learners, hands=chunk, seed=rng.getrandbits(31),
                  names=('self_a', 'self_b'), deal_policy='independent')
        if (i + 1) % max(chunks // 20, 1) == 0 or i + 1 == chunks:
            score = evaluate(table, RandomAgent(), eval_hands, eval_seed)
            curve.append({'hands': (i + 1) * chunk, 'epsilon': round(epsilon, 4),
                          'states': len(table), 'vs_random_chips_per_hand': score})
            if verbose:
                print(f"{(i + 1) * chunk:>8} hands  eps={epsilon:.3f}  states={len(table):>4}  "
                      f"vs random: {score:+.3f} chips/hand", flush=True)
    return table, curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands', type=int, default=200_000, help='Self-play training hands')
    parser.add_argument('--chunk', type=int, default=2_000, help='Hands per epsilon step')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--epsilon-start', type=float, default=0.3)
    parser.add_argument('--epsilon-end', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=4701)
    parser.add_argument('--eval-hands', type=int, default=2_000)
    parser.add_argument('--eval-seed', type=int, default=94701)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0 < args.alpha <= 1 or not 0 <= args.gamma <= 1:
        parser.error('alpha must be in (0, 1] and gamma in [0, 1]')
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error('Require 0 <= epsilon-end <= epsilon-start <= 1')
    if args.seed == args.eval_seed:
        parser.error('Training and evaluation seeds must differ')

    table, curve = train(hands=args.hands, chunk=args.chunk, alpha=args.alpha, gamma=args.gamma,
                         epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                         seed=args.seed, eval_hands=args.eval_hands, eval_seed=args.eval_seed)
    
    final = {'vs_random_chips_per_hand': curve[-1]['vs_random_chips_per_hand']}

    metadata = {'training_hands': args.hands, 'chunk': args.chunk, 'alpha': args.alpha,
                'gamma': args.gamma, 'epsilon_start': args.epsilon_start,
                'epsilon_end': args.epsilon_end, 'training_seed': args.seed,
                'eval_seed': args.eval_seed, 'eval_hands': args.eval_hands,
                'opponent': 'self-play (shared Q-table)', 'reward': 'net chips at hand end',
                'state': 'own rank | public rank | position | betting sequence per round',
                'rlcard_version': '1.2.0', 'learning_curve': curve, 'final_evaluation': final}
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    QLearningAgent(table=table).save(args.output, metadata)
    print(json.dumps(final, indent=2))
    print(f'Frozen Q-table ({len(table)} states): {args.output}')


if __name__ == '__main__':
    main()
