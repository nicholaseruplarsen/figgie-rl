"""Actor-critic policies over the CS224R action space (§6.1).

Every policy is MLP encoder -> memory core -> masked categorical actor over 257
actions + scalar critic. The core is one of

  none         no memory: the actor sees only the current encoding
  mingru       minGRU (Feng et al. 2024): h = (1-z) h + z W x, gates from x only
  gru          GRUCell, the CS224R architecture (default, matches old checkpoints)
  transformer  causal 2-layer transformer over the last `window` encodings

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
            layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.0, batch_first=True,
                                               norm_first=True)
            self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
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
            return (self.window - 1) * (self.hidden + 1)     # last window-1 encodings + valid flags
        return self.hidden

    def init_state(self, b, device=None):
        return torch.zeros(b, self.state_dim, device=device)

    def _split(self, s):
        k, h = self.window - 1, self.hidden
        return s[:, :k * h].view(-1, k, h), s[:, k * h:]

    # ---- cores
    def _tf_windows(self, tok, valid):
        """tok [N, K, H], valid [N, K] bool (last token always valid) -> output at last token [N, H]."""
        out = self.tf(tok, src_key_padding_mask=~valid)
        return self.tf_norm(out[:, -1])

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
        tok, val = self._split(s)
        tok = torch.cat([tok, e.unsqueeze(1)], 1)
        val = torch.cat([val > 0.5, torch.ones_like(val[:, :1], dtype=torch.bool)], 1)
        z = self._tf_windows(tok, val)
        ns = torch.cat([tok[:, 1:].reshape(len(e), -1), val[:, 1:].float()], 1)
        return z, ns

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
        T, B, Hd = e.shape
        K = self.window
        s0 = s0 * (1 - done[0]).unsqueeze(-1)
        tok0, val0 = self._split(s0)                                   # [B, K-1, H], [B, K-1]
        seq = torch.cat([tok0, e.transpose(0, 1)], 1)                  # [B, K-1+T, H]
        val = torch.cat([val0 > 0.5, torch.ones(B, T, dtype=torch.bool, device=e.device)], 1)
        ep = torch.cat([torch.zeros(B, K - 1, device=e.device), done.transpose(0, 1).cumsum(1)], 1)
        idx = torch.arange(T, device=e.device).unsqueeze(1) + torch.arange(K, device=e.device)  # [T, K]
        win, wv, wep = seq[:, idx], val[:, idx], ep[:, idx]           # [B, T, K, ...]
        wv = wv & (wep == wep[..., -1:])
        z = self._tf_windows(win.reshape(B * T, K, Hd), wv.reshape(B * T, K))
        return z.view(B, T, Hd).transpose(0, 1)


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


def save(net, path):
    torch.save(dict(config=net.config, state_dict=net.state_dict()), path)


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
