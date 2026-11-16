"""step() and unroll() agree for every memory core, across episode resets."""
import pytest
import torch

from figgie.rl.features import FLOW_DIM
from figgie.rl.policy import ARCHS, ActorCritic, load_net


@pytest.mark.parametrize("arch", ARCHS)
def test_step_matches_unroll(arch):
    torch.manual_seed(0)
    net = ActorCritic(arch, obs_dim=FLOW_DIM, window=8).eval()
    T, B = 21, 3
    x = torch.randn(T, B, FLOW_DIM)
    mask = torch.ones(T, B, 257, dtype=torch.bool)
    done = torch.zeros(T, B)
    done[0] = 1
    done[7, 0] = done[15, 2] = 1
    s = torch.randn(B, net.state_dim).abs()          # garbage state, wiped by done[0]
    s0 = s.clone()
    with torch.no_grad():
        step_logits = []
        for t in range(T):
            s = s * (1 - done[t]).unsqueeze(-1)
            lg, _, s = net.step(x[t], s, mask[t])
            step_logits.append(lg)
        step_logits = torch.stack(step_logits)
        # replay in two chunks, starting the second from the state after step 10
        s = s0.clone()
        for t in range(10):
            s = s * (1 - done[t]).unsqueeze(-1)
            _, _, s = net.step(x[t], s, mask[t])
        a, _, _ = net.unroll(x[:10], s0, mask[:10], done[:10])
        b, _, _ = net.unroll(x[10:], s * (1 - done[10]).unsqueeze(-1), mask[10:], done[10:])
    torch.testing.assert_close(torch.cat([a, b]), step_logits, atol=1e-5, rtol=1e-4)


def test_old_checkpoint_loads():
    net = load_net("checkpoints/ppo_v2.pt")
    assert net.arch == "gru" and net.config["obs_dim"] == 69
