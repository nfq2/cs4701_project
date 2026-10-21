from .random_agent import RandomAgent

# Register additional agents here once they are implemented.
AGENTS = {
    "random": {
        "name": "Random agent",
        "description": "Chooses equally among legal actions. Does not learn your strategy.",
        "factory": RandomAgent,
    },
}

__all__ = ["RandomAgent", "AGENTS"]
