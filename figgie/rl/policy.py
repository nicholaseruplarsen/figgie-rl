"""Recurrent actor-critic (CS224R §6.1): 2-layer MLP encoder -> GRU cell (128)
-> masked categorical actor over 257 actions + scalar critic."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..actions import N_ACTIONS, decode, legal_mask
from .features import OBS_DIM, Featurizer

H = 128


class ActorCriticRecurrent(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=H, n_actions=N_ACTIONS):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRUCell(hidden, hidden)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)

    def step(self, x, h, mask):
        h = self.gru(self.enc(x), h)
        logits = self.actor(h).masked_fill(~mask, -1e9)
        return logits, self.critic(h).squeeze(-1), h

    def unroll(self, x, h0, mask, done):
        """x: [T, B, D]; done[t] marks that step t starts a new episode."""
        # Run the fused GRU kernel over the spans between episode resets
        # (same weights as the GRUCell, so it matches step() exactly).
        e = self.enc(x)
        g = self.gru
        cuts = [0] + [t for t in range(1, x.shape[0]) if bool(done[t].any())] + [x.shape[0]]
        hs, h = [], h0
        for a, b in zip(cuts[:-1], cuts[1:]):
            h = h * (1 - done[a]).unsqueeze(-1)
            out, hn = torch._VF.gru(e[a:b], h.unsqueeze(0), [g.weight_ih, g.weight_hh, g.bias_ih, g.bias_hh],
                                     True, 1, 0.0, self.training, False, False)
            hs.append(out)
            h = hn[0]
        hs = torch.cat(hs)
        logits = self.actor(hs).masked_fill(~mask, -1e9)
        return logits, self.critic(hs).squeeze(-1), h


class PolicyAgent:
    """Plays a trained network on the shared engine."""

    def __init__(self, net, deterministic=True, name="PPO", seed=None):
        self.net, self.det, self.name = net, deterministic, name
        self.gen = torch.Generator().manual_seed(seed or 0)

    def reset(self, obs):
        self.feat = Featurizer(obs)
        self.h = torch.zeros(1, H)

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


def load(path, **kw):
    net = ActorCriticRecurrent()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return PolicyAgent(net, **kw)
