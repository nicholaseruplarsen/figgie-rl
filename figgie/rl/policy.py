"""Actor-critic policies over the CS224R action space (§6.1).

Every policy is MLP encoder -> memory core -> masked categorical actor over 257
actions + scalar critic. The core is one of

  none         no memory: the actor sees only the current encoding
  mingru       minGRU (Feng et al. 2024): h = (1-z) h + z W x, gates from x only
  gru          GRUCell, the CS224R architecture (default, matches old checkpoints)
  transformer  causal 2-layer transformer, each tick attending to the last `window`
               ticks of its episode at every layer (cached keys and values)

State is a flat [B, S] tensor for every core so the trainer can store and reset
it uniformly. `step` is one tick, `unroll` replays a chunk from a stored state
and gives exactly the same outputs as repeated `step` calls.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..actions import N_ACTIONS, decode, legal_mask
from .features import FLOW_DIM, OBS_DIM, make_featurizer

H = 128
ARCHS = ("none", "mingru", "gru", "transformer")


class Block(nn.Module):
    """Pre-norm transformer block whose queries are the new ticks and whose keys
    and values also cover cached earlier ticks."""

    def __init__(self, h, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.q, self.kv, self.o = nn.Linear(h, h), nn.Linear(h, 2 * h), nn.Linear(h, h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.ReLU(), nn.Linear(2 * h, h))

    def forward(self, x, full, mask):
        B, T, H = x.shape
        n, d = self.heads, H // self.heads
        q = self.q(self.ln1(x)).view(B, T, n, d).transpose(1, 2)
        k, v = self.kv(self.ln1(full)).view(B, -1, 2, n, d).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        x = x + self.o(a.transpose(1, 2).reshape(B, T, H))
        return x + self.ff(self.ln2(x))


class ActorCritic(nn.Module):
    def __init__(self, arch="gru", obs_dim=OBS_DIM, hidden=H, n_actions=N_ACTIONS, window=128, layers=2, heads=4):
        super().__init__()
        assert arch in ARCHS, arch
        self.arch, self.hidden, self.window = arch, hidden, window
        self.config = dict(arch=arch, obs_dim=obs_dim, hidden=hidden, window=window, layers=layers, heads=heads)
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        if arch == "gru":
            self.gru = nn.GRUCell(hidden, hidden)
        elif arch == "mingru":
            self.mz = nn.Linear(hidden, hidden)
            self.mh = nn.Linear(hidden, hidden)
        elif arch == "transformer":
            self.blocks = nn.ModuleList(Block(hidden, heads) for _ in range(layers))
            self.tf_norm = nn.LayerNorm(hidden)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)

    # ---- state layout
    @property
    def state_dim(self):
        if self.arch == "none":
            return 1
        if self.arch == "transformer":
            # every layer's inputs for the last window-1 ticks (a KV cache), plus their valid flags
            return (len(self.blocks) * self.hidden + 1) * (self.window - 1)
        return self.hidden

    def init_state(self, b, device=None):
        return torch.zeros(b, self.state_dim, device=device)

    def _tf(self, e, s, done):
        """Sliding-window causal attention with cached layer inputs.

        e [B, T, H] new encodings, s [B, S] state, done [B, T] (1 = episode starts at that tick).
        Each query attends to the last `window` ticks of its own episode, at every layer.
        """
        B, T, H = e.shape
        C, K, L = self.window - 1, self.window, len(self.blocks)
        cache = s[:, :L * C * H].view(B, L, C, H)
        valid = torch.cat([s[:, L * C * H:] > 0.5, torch.ones(B, T, dtype=torch.bool, device=e.device)], 1)
        ep_new = done.cumsum(1)
        ep = torch.cat([torch.zeros(B, C, device=e.device), ep_new], 1)
        pos = torch.arange(C + T, device=e.device)
        qpos = C + torch.arange(T, device=e.device)
        band = (pos[None] <= qpos[:, None]) & (pos[None] > qpos[:, None] - K)          # [T, C+T]
        mask = band[None] & valid[:, None] & (ep[:, None] == ep_new[:, :, None])      # [B, T, C+T]
        x, new_cache = e, []
        for l, blk in enumerate(self.blocks):
            full = torch.cat([cache[:, l], x], 1)
            new_cache.append(full[:, -C:])
            x = blk(x, full, mask)
        keep = valid[:, -C:] & (ep[:, -C:] == ep_new[:, -1:])
        ns = torch.cat([torch.stack(new_cache, 1).reshape(B, -1), keep.float()], 1)
        return self.tf_norm(x), ns

    def core_step(self, e, s):
        if self.arch == "none":
            return e, s
        if self.arch == "gru":
            h = self.gru(e, s)
            return h, h
        if self.arch == "mingru":
            z = torch.sigmoid(self.mz(e))
            h = (1 - z) * s + z * self.mh(e)
            return h, h
        z, ns = self._tf(e.unsqueeze(1), s, torch.zeros(len(e), 1, device=e.device))
        return z[:, 0], ns

    def step(self, x, s, mask):
        z, s = self.core_step(self.enc(x), s)
        self.last_z = z
        logits = self.actor(z).masked_fill(~mask, -1e9)
        return logits, self.critic(z).squeeze(-1), s

    def unroll(self, x, s0, mask, done):
        """x: [T, B, D]; done[t] marks that step t starts a new episode (state reset)."""
        e = self.enc(x)
        T, B = x.shape[:2]
        if self.arch == "none":
            z = e
        elif self.arch == "gru":
            z = self._gru_unroll(e, s0, done)
        elif self.arch == "mingru":
            zs, h = [], s0
            for t in range(T):
                h = h * (1 - done[t]).unsqueeze(-1)
                g = torch.sigmoid(self.mz(e[t]))
                h = (1 - g) * h + g * self.mh(e[t])
                zs.append(h)
            z = torch.stack(zs)
        else:
            z = self._tf_unroll(e, s0, done)
        logits = self.actor(z).masked_fill(~mask, -1e9)
        return logits, self.critic(z).squeeze(-1), None

    def _gru_unroll(self, e, h0, done):
        # Fused GRU kernel over the spans between episode resets (same weights
        # as the GRUCell, so it matches step() exactly).
        g = self.gru
        cuts = [0] + [t for t in range(1, e.shape[0]) if bool(done[t].any())] + [e.shape[0]]
        hs, h = [], h0
        for a, b in zip(cuts[:-1], cuts[1:]):
            h = h * (1 - done[a]).unsqueeze(-1)
            out, hn = torch._VF.gru(e[a:b], h.unsqueeze(0), [g.weight_ih, g.weight_hh, g.bias_ih, g.bias_hh],
                                     True, 1, 0.0, self.training, False, False)
            hs.append(out)
            h = hn[0]
        return torch.cat(hs)

    def _tf_unroll(self, e, s0, done):
        z, _ = self._tf(e.transpose(0, 1), s0 * (1 - done[0]).unsqueeze(-1), done.transpose(0, 1))
        return z.transpose(0, 1)


def ActorCriticRecurrent(**kw):
    """The CS224R GRU policy (kept for old call sites and checkpoints)."""
    return ActorCritic("gru", **kw)


class PolicyAgent:
    """Plays a trained network on the shared engine."""

    def __init__(self, net, deterministic=True, name="PPO", seed=None):
        self.net, self.det, self.name = net, deterministic, name
        self.gen = torch.Generator().manual_seed(seed or 0)
        self.flow = net.config["obs_dim"] == FLOW_DIM

    def reset(self, obs):
        self.feat = make_featurizer(self.flow)(obs)
        self.h = self.net.init_state(1)

    @torch.no_grad()
    def act(self, obs):
        x = torch.from_numpy(self.feat(obs)).unsqueeze(0)
        m = torch.from_numpy(legal_mask(obs)).unsqueeze(0)
        logits, _, self.h = self.net.step(x, self.h, m)
        if self.det:
            a = int(logits.argmax(-1))
        else:
            a = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=self.gen))
        return decode(a)


def save(net, path, **extra):
    torch.save(dict(config=net.config, state_dict=net.state_dict(), **extra), path)


def load_net(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "config" in sd:
        net = ActorCritic(**sd["config"])
        net.load_state_dict(sd["state_dict"])
    else:                                   # old CS224R checkpoints: a bare GRU state_dict
        net = ActorCriticRecurrent()
        net.load_state_dict(sd)
    return net.eval()


def load(path, **kw):
    return PolicyAgent(load_net(path), **kw)
