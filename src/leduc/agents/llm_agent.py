"""Stateless Leduc policy with bounded hosted Chat Completions requests."""

from dataclasses import asdict
import hashlib
import json
import math
import os
import random
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from leduc.interface import Action, Agent, HandResult, Observation

PROVIDERS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
}

PROMPT = """Play two-player RLCard Leduc Hold'em to maximize expected net chips.
The deck is SJ, HJ, SQ, HQ, SK, HK: two of each rank, J < Q < K. Suits
have no effect. Each player has one private card. One shared public card is
revealed after the first betting round. A pair with the public card beats a
high card; otherwise higher private rank wins; equal ranks split the pot.
Blinds are 1 and 2 chips. Raises add 2 chips in round_index 0 and 4 in
round_index 1, with at most two raises per round. A raise also pays any
outstanding amount to call. There is no stack limit or persistent bankroll.
Contributions and pot are cumulative chips for this hand. Seats are 0 and 1.
You see only your private card and public information for the current hand.
Choose exactly one of legal_actions. Return only a JSON object with one field,
"action", containing that exact lowercase action. Do not include explanations.
"""


class APIError(Exception):
    """Sanitized provider failure, safe to put in experiment logs."""


def openai_completion(payload: dict, *, api_key: str, timeout: float, provider: str = "openai") -> dict:
    request = Request(
        PROVIDERS[provider][0],
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "User-Agent": "leduc-agents/0.1"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise APIError(f"http_{exc.code}") from None
    except (URLError, TimeoutError, OSError):
        raise APIError("connection_or_timeout") from None
    except (ValueError, UnicodeError):
        raise APIError("invalid_response_json") from None


class LLMAgent(Agent):
    """No cross-hand memory or hidden-deal identifiers enter the prompt.

    One request per decision, no retries. Failures use seeded uniform legal
    actions and are explicitly recorded. Limits apply per match/UI session.
    A supplied completion callable allows offline tests without credentials.
    """

    def __init__(self, *, model: str, max_calls: int = 100,
                 max_completion_tokens: int = 256, timeout: float = 20,
                 completion=None, provider: str = "openai", reasoning_effort: str | None = None):
        if not model or not model.strip():
            raise ValueError("Specify --llm-model (a model supporting JSON schema output)")
        if max_calls < 0 or max_completion_tokens <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LLM call limit must be nonnegative; token limit and timeout must be positive")
        if provider not in PROVIDERS:
            raise ValueError("Provider must be openai or groq")
        if reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("Reasoning effort must be low, medium, or high")
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.model = model
        self.max_calls = max_calls
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        key_name = PROVIDERS[provider][1]
        self._api_key = os.environ.get(key_name, "")
        if completion is None and not self._api_key:
            raise ValueError(f"Set {key_name} in your shell before using the LLM agent")
        self._completion = completion or self._request
        self.on_match_start(seed=0)

    def _request(self, payload):
        return openai_completion(payload, api_key=self._api_key, timeout=self.timeout, provider=self.provider)

    def on_match_start(self, *, seed):
        self._rng = random.Random(seed)
        self.calls = self.fallbacks = self.prompt_tokens = self.completion_tokens = 0
        self._last = {}

    def system_prompt(self):
        return PROMPT

    def visible_information(self, observation):
        visible = asdict(observation)
        del visible["hand_index"]
        return visible

    def configuration(self):
        return {"agent_class": type(self).__name__, "provider": self.provider, "reasoning_effort": self.reasoning_effort,
                "model": self.model, "prompt_sha256": hashlib.sha256(self.system_prompt().encode()).hexdigest(),
                "prompt": self.system_prompt(), "memory": "none", "max_calls": self.max_calls,
                "max_completion_tokens": self.max_completion_tokens,
                "timeout_seconds": self.timeout, "fallback": "uniform_legal", "retries": 0}

    def statistics(self):
        return {"api_calls": self.calls, "fallbacks": self.fallbacks,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}

    def decision_metadata(self):
        return dict(self._last)

    def select_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise ValueError("Cannot choose an action from a terminal observation")
        self._last = {"fallback": False}
        if self.calls >= self.max_calls:
            return self._fallback(observation, "call_limit")
        visible = self.visible_information(observation)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt()},
                         {"role": "user", "content": json.dumps(visible, sort_keys=True)}],
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "poker_action", "strict": True,
                "schema": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": list(observation.legal_actions)}},
                    "required": ["action"], "additionalProperties": False}}},
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        self.calls += 1
        try:
            response = self._completion(payload)
        except APIError as exc:
            return self._fallback(observation, str(exc))
        try:
            usage = response.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key, 0)
                if isinstance(value, int) and value >= 0:
                    setattr(self, key, getattr(self, key) + value)
                    self._last[key] = value
            choice = response["choices"][0]
            if choice.get("finish_reason") != "stop":
                return self._fallback(observation, "incomplete_or_refused")
            result = json.loads(choice["message"]["content"])
            if not isinstance(result, dict) or set(result) != {"action"}:
                raise ValueError("Invalid action object")
            action = result["action"]
            if action not in observation.legal_actions:
                raise ValueError("Illegal action")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return self._fallback(observation, "invalid_action_response")
        return action

    def _fallback(self, observation, reason):
        self.fallbacks += 1
        self._last.update(fallback=True, failure_reason=reason)
        return self._rng.choice(observation.legal_actions)


class MemoryLLMAgent(LLMAgent):
    """Full player-visible completed-hand history, reset per match/session.

    No summaries, sliding windows, or model conversations are retained. All
    completed hands are sent on every decision, subject to provider limits.
    """

    requires_independent_deals = True

    def on_match_start(self, *, seed):
        super().on_match_start(seed=seed)
        self._completed_hands = []
        self._hand_observations = []

    def on_hand_start(self, *, hand_index, seat):
        self._hand_observations = []

    def on_hand_end(self, result: HandResult):
        visible_result = asdict(result)
        del visible_result["hand_index"]
        self._completed_hands.append({
            "observations_at_own_turns": self._hand_observations,
            "result": visible_result,
        })
        self._hand_observations = []

    def system_prompt(self):
        return PROMPT + """
You also receive previous_hands in chronological order for this match.
Each result is from your perspective: seat identifies your seat in that hand,
payoff_chips is your net gain or loss, and opponent_revealed_card is null when
not revealed. Seats can change between hands; use each hand's seat to identify
your actions and the opponent's. Null cards remain unknown. Use the full visible
history to identify opponent tendencies and improve your current decision.
Only select an action from the current hand's legal_actions.
"""

    def select_action(self, observation):
        if observation.legal_actions:
            self._hand_observations.append(super().visible_information(observation))
        return super().select_action(observation)

    def visible_information(self, observation):
        return {**super().visible_information(observation),
                "previous_hands": self._completed_hands}

    def decision_metadata(self):
        return {**super().decision_metadata(), "memory_hands": len(self._completed_hands)}

    def configuration(self):
        return {**super().configuration(), "memory": "full_visible_history"}
