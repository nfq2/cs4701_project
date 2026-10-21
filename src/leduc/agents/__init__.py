from .random_agent import RandomAgent
from .bayesian_agent import BayesianAgent

# Register additional agents here once they are implemented.
AGENTS = {
    "bayesian": {
        "name": "Bayesian agent",
        "description": "Updates beliefs about your card from your actions and chooses the highest expected chip payoff under a fixed opponent model. Beliefs reset every hand.",
        "factory": BayesianAgent,
    },
    "random": {
        "name": "Random agent",
        "description": "Chooses equally among legal actions. Does not learn your strategy.",
        "factory": RandomAgent,
    },
}

__all__ = ["RandomAgent", "BayesianAgent", "AGENTS"]
