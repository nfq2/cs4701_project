"""Local, single-player browser interface backed by the shared RLCard adapter."""

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import tempfile
from time import perf_counter
from urllib.parse import urlparse

import rlcard

from leduc.agents import RandomAgent
from leduc.interface import ActionEvent, HandResult
from leduc.runner import BIG_BLIND_CHIPS, make_observation


class HumanGame:
    """One human session. Browser state never contains hidden opponent cards."""

    def __init__(self, seed=None, log=None):
        self.seed = secrets.randbits(32) if seed is None else seed
        self.env = rlcard.make("leduc-holdem", config={"seed": self.seed})
        self.agent = RandomAgent()
        self.agent.on_match_start(seed=self.seed ^ 0xA5A5A5A5)
        self.log = log
        self.hand_index = -1
        self.revision = 0
        self.total = 0.0
        self.completed = 0
        self.last_payoff = None
        self.emit({"type": "session_start", "seed": self.seed,
                   "opponent": "random", "payoff_unit": "chips",
                   "deal_policy": "independent_hands_alternating_seats"})
        self.next_hand()

    def emit(self, record):
        if self.log:
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()

    def next_hand(self):
        if self.hand_index >= 0 and not self.env.is_over():
            raise ValueError("Finish the current hand before dealing again.")
        self.hand_index += 1
        self.human_seat = self.hand_index % 2
        self.bot_seat = 1 - self.human_seat
        self.history = []
        self.last_payoff = None
        self.state, self.seat = self.env.reset()
        self.small_blind_seat = self.seat
        self.agent.on_hand_start(hand_index=self.hand_index, seat=self.bot_seat)
        self.emit({"type": "hand_start", "hand_index": self.hand_index,
                   "human_seat": self.human_seat,
                   "small_blind_seat": self.small_blind_seat})
        self.advance_bot()
        self.revision += 1
        return self.view()

    def observation(self):
        return make_observation(self.state, hand_index=self.hand_index,
                                seat=self.seat, small_blind_seat=self.small_blind_seat,
                                history=self.history)

    def step(self, action, elapsed):
        observation = self.observation()
        if action not in observation.legal_actions:
            raise ValueError("That action is not legal right now.")
        self.emit({"type": "action", "actor": "human" if self.seat == self.human_seat else "random",
                   "observation": asdict(observation), "action": action,
                   "decision_seconds": elapsed})
        self.history.append(ActionEvent(self.seat, observation.round_index, action))
        self.state, self.seat = self.env.step(action, raw_action=True)

    def advance_bot(self):
        while not self.env.is_over() and self.seat == self.bot_seat:
            start = perf_counter()
            action = self.agent.select_action(self.observation())
            self.step(action, perf_counter() - start)
        if self.env.is_over():
            payoffs = [float(x) * BIG_BLIND_CHIPS for x in self.env.get_payoffs()]
            self.last_payoff = payoffs[self.human_seat]
            self.total += self.last_payoff
            self.completed += 1
            human = self.env.get_state(self.human_seat)["raw_obs"]
            bot = self.env.get_state(self.bot_seat)["raw_obs"]
            showdown = self.history[-1].action != "fold"
            self.agent.on_hand_end(HandResult(
                hand_index=self.hand_index, seat=self.bot_seat,
                private_card=bot["hand"], public_card=bot["public_card"],
                opponent_revealed_card=human["hand"] if showdown else None,
                payoff_chips=payoffs[self.bot_seat], history=tuple(self.history),
            ))
            self.emit({"type": "hand_end", "hand_index": self.hand_index,
                       "human_payoff_chips": self.last_payoff,
                       "human_total_chips": self.total, "showdown": showdown})
        self.turn_started = perf_counter()

    def act(self, action):
        if self.env.is_over() or self.seat != self.human_seat:
            raise ValueError("It is not your turn.")
        self.step(action, perf_counter() - self.turn_started)
        self.advance_bot()
        self.revision += 1
        return self.view()

    def view(self):
        raw = self.env.get_state(self.human_seat)["raw_obs"]
        over = self.env.is_over()
        showdown = over and self.history[-1].action != "fold"
        opponent_card = self.env.get_state(self.bot_seat)["raw_obs"]["hand"] if showdown else None
        contributions = [int(v) for v in raw["all_chips"]]
        return {
            "revision": self.revision, "hand": self.hand_index + 1,
            "completed": self.completed, "total_chips": self.total,
            "payoff": self.last_payoff, "over": over, "showdown": showdown,
            "human_card": raw["hand"], "opponent_card": opponent_card,
            "public_card": raw["public_card"], "pot": sum(contributions),
            "human_contribution": contributions[self.human_seat],
            "opponent_contribution": contributions[self.bot_seat],
            "human_blind": "Small blind" if self.human_seat == self.small_blind_seat else "Big blind",
            "round": 2 if raw["public_card"] else 1,
            "to_call": max(contributions) - contributions[self.human_seat],
            "raise_by": 4 if raw["public_card"] else 2,
            "legal_actions": list(self.state["raw_legal_actions"]) if not over else [],
            "history": [{"actor": "You" if e.seat == self.human_seat else "Random agent",
                         "round": e.round_index + 1, "action": e.action} for e in self.history],
        }


WEB_ROOT = Path(__file__).with_name("web")


def make_handler(game):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, data):
            self.send(status, json.dumps(data).encode())

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                self.send_json(200, game.view())
                return
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path not in files:
                self.send_json(404, {"error": "Not found"})
                return
            name, content_type = files[path]
            self.send(200, (WEB_ROOT / name).read_bytes(), content_type)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.send_json(403, {"error": "Cross-origin requests are not allowed."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size.")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected JSON.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                if data.get("revision") != game.revision:
                    self.send_json(409, {"error": "The hand changed. Refreshing the table."})
                    return
                if self.path == "/api/action":
                    state = game.act(data.get("action"))
                elif self.path == "/api/next":
                    state = game.next_hand()
                else:
                    self.send_json(404, {"error": "Not found"})
                    return
                self.send_json(200, state)
            except (ValueError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, help="Optional reproducible session seed")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or (args.seed is not None and not 0 <= args.seed < 2**32):
        parser.error("Port must be 1–65535; seed must be between 0 and 2^32 - 1.")
    Path("results").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir="results",
                                     prefix="human-", suffix=".jsonl", delete=False) as log:
        game = HumanGame(seed=args.seed, log=log)
        try:
            server = HTTPServer(("127.0.0.1", args.port), make_handler(game))
        except OSError as exc:
            parser.exit(1, f"Cannot start server: {exc}. Try a different --port.\n")
        print(f"Play at http://127.0.0.1:{args.port}", flush=True)
        print(f"Session log: {log.name}\nPress Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
