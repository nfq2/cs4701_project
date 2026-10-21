from .random_agent import RandomAgent
from .bayesian_agent import BayesianAgent
from .llm_agent import LLMAgent, MemoryLLMAgent

# Register additional agents here once they are implemented.
AGENTS = {
    "bayesian": {
        "name": "Bayesian agent",
        "description": "Updates beliefs about your card from your actions and chooses the highest expected chip payoff under a fixed opponent model. Beliefs reset every hand.",
        "factory": BayesianAgent,
    },
    "llm": {
        "name": "LLM without memory",
        "description": "Uses GPT-OSS 120B on Groq to choose from legal actions using its own card and the current hand’s public history. Does not remember previous hands.",
        "factory": lambda: LLMAgent(model="openai/gpt-oss-120b", provider="groq",
                                     reasoning_effort="low", max_completion_tokens=2048),
    },
    "llm-memory": {
        "name": "LLM with full memory",
        "description": "Uses the same LLM plus every completed hand’s visible actions and outcomes in this session. Opponent cards are remembered only when revealed at showdown. Memory resets on restart or opponent change.",
        "factory": lambda: MemoryLLMAgent(model="openai/gpt-oss-120b", provider="groq",
                                           reasoning_effort="low", max_completion_tokens=2048),
    },
    "random": {
        "name": "Random agent",
        "description": "Chooses equally among legal actions. Does not learn your strategy.",
        "factory": RandomAgent,
    },
}

__all__ = ["RandomAgent", "BayesianAgent", "LLMAgent", "MemoryLLMAgent", "AGENTS"]
