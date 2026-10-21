"""Shared command-line selection for matches and human play."""
from .llm_agent import LLMAgent, MemoryLLMAgent
from . import AGENTS


def add_agent_options(parser):
    parser.add_argument("--llm-provider", choices=("openai", "groq"), default="openai")
    parser.add_argument("--llm-model", help="Provider model ID supporting structured JSON schema output")
    parser.add_argument("--llm-reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--llm-max-calls", type=int, default=100, help="Per LLM agent per match/session")
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--llm-timeout", type=float, default=20)


def create_agent(name, args):
    if name in ("llm", "llm-memory"):
        agent_class = MemoryLLMAgent if name == "llm-memory" else LLMAgent
        return agent_class(model=args.llm_model, provider=args.llm_provider,
                        reasoning_effort=args.llm_reasoning_effort, max_calls=args.llm_max_calls,
                        max_completion_tokens=args.llm_max_tokens, timeout=args.llm_timeout)
    if name in AGENTS:
        return AGENTS[name]["factory"]()
    raise ValueError(f"Unknown agent: {name}")
