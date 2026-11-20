"""Play Figgie in the browser against agents from this repo.

    python scripts/play.py                                  # top three by tournament profit
    python scripts/play.py --opponents fig.AdaptiveMM PPO-v2 Bayesian --tick 0.3

You sit at seat 0. The agents get the same Obs as in the tournament and every order
goes through Game.apply, so legality and settlement are the engine's own. Your orders
are queued and applied on your turn in the next tick (several per turn are allowed,
like the multi-order bot ports). The page is re-read from disk on every load, and
--reload restarts the server when a .py file under figgie/ or scripts/ changes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from figgie.agents.registry import AGENTS, make  # noqa: E402
from figgie.engine import (ASK, BID, CANCEL_ASK, CANCEL_BID, HIT, LIFT, NOOP, SUITS,  # noqa: E402
                           Config, Game, Obs)

PAGE = Path(__file__).with_name("play_web") / "index.html"
PPO = {"PPO-v1": "checkpoints/ppo_v1.pt", "PPO-v2": "checkpoints/ppo_v2.pt",
       "PPO-v2-long": "checkpoints/ppo_v2_long.pt", "PPO-noBC": "checkpoints/ppo_scratch.pt"}
KINDS = {"bid": BID, "ask": ASK, "cancel_bid": CANCEL_BID, "cancel_ask": CANCEL_ASK, "lift": LIFT, "hit": HIT}


def build(name, seed):
    """Same construction as scripts/tournament.py: PPO nets play deterministically."""
    if name in PPO:
        from figgie.rl.policy import PolicyAgent, load
        return PolicyAgent(load(ROOT / PPO[name]).net, deterministic=True, name=name)
    return make(name, seed)


class Session:
    def __init__(self, opponents, tick_s, start_paused):
        self.opponents, self.tick_s = opponents, tick_s
        self.names = ["You"] + opponents
        self.lock = threading.Lock()
        self.paused = start_paused
        self.steps = 0          # ticks requested while paused
        self.scores = [0.0] * len(self.names)
        self.history = []       # profits per finished game
        self.new_game()
        threading.Thread(target=self.loop, daemon=True).start()

    def new_game(self):
        seed = random.randrange(1 << 30)
        self.game = Game(Config(n_players=len(self.names)), seed)
        self.agents = [None] + [build(n, seed + i) for i, n in enumerate(self.opponents, 1)]
        self.obs = [Obs(self.game, i) for i in range(self.game.n)]
        for i in range(1, self.game.n):
            self.agents[i].reset(self.obs[i])
        self.order_rng = random.Random(seed * 7919 + 1)
        self.pending = []       # human orders waiting for their turn
        self.log = []           # human order outcomes
        self.next_tick = 0
        self.profits = None

    def run_tick(self):
        g = self.game
        g.tick = t = self.next_tick
        order = list(range(g.n))
        self.order_rng.shuffle(order)
        for p in order:
            if p == 0:
                acts, self.pending = self.pending, []
            else:
                acts = self.agents[p].act(self.obs[p])
                if not acts:
                    continue
                if isinstance(acts, tuple):
                    acts = (acts,)
            for act in acts:
                kind = act[0]
                n0 = len(g.trades)
                ok = g.apply(p, kind, act[1] if len(act) > 1 else 0, act[2] if len(act) > 2 else 0)
                if p == 0:
                    what = {v: k for k, v in KINDS.items()}[kind] + " " + SUITS[act[1]] + (f" @{act[2]}" if kind in (BID, ASK) else "")
                    self.log.append(dict(tick=t, text=what, ok=ok, traded=len(g.trades) > n0))
        self.next_tick += 1
        if self.next_tick >= g.cfg.ticks:
            self.profits = g.settle()
            self.history.append(self.profits)
            self.scores = [a + b for a, b in zip(self.scores, self.profits)]

    def loop(self):
        while True:
            time.sleep(self.tick_s if not self.paused else 0.02)
            with self.lock:
                if self.profits is not None:
                    continue
                if self.paused:
                    if not self.steps:
                        continue
                    self.steps -= 1
                self.run_tick()

    def state(self):
        g = self.game
        done = self.profits is not None
        book = []
        for s in range(4):
            bb, ba = g.best_bid(s), g.best_ask(s)
            book.append(dict(bid=bb and dict(price=bb[0], who=bb[1]), ask=ba and dict(price=ba[0], who=ba[1]),
                             my_bid=g.bids[0][s], my_ask=g.asks[0][s], last=g.last_price[s]))
        return dict(
            names=self.names, tick=self.next_tick, ticks=g.cfg.ticks, paused=self.paused, tick_s=self.tick_s,
            hand=g.hands[0], cash=g.cash[0], book=book, pending=len(self.pending),
            counts=[sum(h) for h in g.hands],
            trades=[dict(tick=t.tick, suit=t.suit, price=t.price, buyer=t.buyer, seller=t.seller,
                         agg=t.buyer if t.buyer_aggressed else t.seller) for t in g.trades[-60:]],
            n_trades=len(g.trades), log=self.log[-12:], done=done,
            reveal=done and dict(hands=g.hands, hands0=g.hands0, goal=g.goal, sizes=g.sizes,
                                 profits=self.profits, winners=g.winners()),
            scores=self.scores, games=len(self.history),
        )

    def command(self, d):
        c = d.get("cmd")
        with self.lock:
            if c in KINDS:
                self.pending.append((KINDS[c], int(d["suit"]), int(d.get("price", 0))))
            elif c == "cancel_all":
                self.pending += [(k, s, 0) for s in range(4) for k in (CANCEL_BID, CANCEL_ASK)]
            elif c == "pause":
                self.paused = not self.paused
            elif c == "step":
                self.paused, self.steps = True, self.steps + int(d.get("n", 1))
            elif c == "speed":
                self.tick_s = max(0.02, float(d["tick_s"]))
            elif c == "finish":   # fast-forward to settlement (you no-op)
                while self.profits is None:
                    self.run_tick()
            elif c == "new":
                self.new_game()
            else:
                return False
        return True


def make_handler(sess):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype="application/json", code=200):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/state":
                with sess.lock:
                    self.send(json.dumps(sess.state()))
            elif self.path in ("/", "/index.html"):
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send("not found", "text/plain", 404)

        def do_POST(self):
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
            ok = sess.command(d)
            self.send(json.dumps(dict(ok=ok)), code=200 if ok else 400)
    return H


def watch_and_reexec():
    files = list((ROOT / "figgie").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    stamp = {f: f.stat().st_mtime for f in files}
    while True:
        time.sleep(0.5)
        if any(f.exists() and f.stat().st_mtime != m for f, m in stamp.items()):
            print("change detected, restarting", flush=True)
            os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponents", nargs="+", default=["fig.AdaptiveMM", "fig.BayesianMM", "Bayesian"],
                    help=f"agent names: {', '.join(list(AGENTS) + list(PPO))}")
    ap.add_argument("--tick", type=float, default=0.5, help="seconds per tick")
    ap.add_argument("--paused", action="store_true", help="start paused (step mode)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8766)))
    ap.add_argument("--reload", action="store_true", help="restart when a .py file changes")
    a = ap.parse_args()
    bad = [n for n in a.opponents if n not in AGENTS and n not in PPO]
    if bad or len(a.opponents) not in (3, 4):
        ap.error(f"need 3 or 4 known opponents, unknown: {bad}")
    sess = Session(a.opponents, a.tick, a.paused)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sess))
    if a.reload:
        threading.Thread(target=watch_and_reexec, daemon=True).start()
    print(f"Figgie vs {', '.join(a.opponents)}  ->  http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
