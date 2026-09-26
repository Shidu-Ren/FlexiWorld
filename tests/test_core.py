from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flexiworld.data.splits import window_split
from flexiworld.data.windows import sample_windows
from flexiworld.models.action import ARPrimitiveActor, VarKActionEncoder
from flexiworld.planning.history import PrimitiveTailLedger
from flexiworld.protocol import expert_history
from flexiworld.utils import RawZeroActionProcessor
from flexiworld.data.sampler import HomogeneousDistributedBatchSampler


@pytest.mark.parametrize("size", [11, 20, 100, 137])
def test_split_matches_upstream_torch_semantics(size):
    actual = window_split(size, 42)
    expected = torch.utils.data.random_split(
        range(size), [0.9, 1 - 0.9], generator=torch.Generator().manual_seed(42)
    )
    for a, b in zip(actual, expected):
        assert a.tolist() == b.indices
    assert not set(actual[0]) & set(actual[1])


def test_window_boundaries_and_exact_spans():
    for span in (35, 55, 75):
        w = sample_windows(
            np.random.default_rng(0),
            np.array([span + 1, span + 1]),
            np.array([0, span + 1]),
            2,
            n_blocks=span // 5,
            total_span=span,
            exclude_uniform_schedule=True,
        )
        assert np.all(w.k.sum(1) == span)
        assert np.all((w.k >= 1) & (w.k <= 10))
        assert not np.any((w.k == 5).all(1))
        assert not w.prev_mask.any()
        assert np.all(w.lat_idx[:, -1] == (w.ep + 1) * (span + 1) - 1)


def test_encoder_ignores_padding():
    torch.manual_seed(0)
    model = VarKActionEncoder(emb_dim=16, d_tok=16, depth=1, heads=2, dim_head=8).eval()
    a = torch.randn(2, 20)
    b = a.clone()
    b[:, 6:] = 1000
    torch.testing.assert_close(model(a, k=3), model(b, k=3), rtol=0, atol=0)


def make_actor():
    return ARPrimitiveActor(
        embed_dim=16,
        action_emb_dim=16,
        d_model=16,
        depth=1,
        heads=2,
        dim_head=8,
        mlp_dim=32,
        use_k_token=False,
    )


def test_actor_prefix_and_student_forcing_backward():
    actor = make_actor().eval()
    z, intent, previous = (torch.randn(2, 16) for _ in range(3))
    short = actor.decode(z, intent, previous, 3)
    long = actor.decode(z, intent, previous, 5)
    torch.testing.assert_close(short, long[:, :3], rtol=0, atol=0)
    actor.train()
    result = actor.nll(
        z, intent, previous, torch.randn(2, 10, 2), torch.tensor([3, 7]), student_p=0.5
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_arcem_zero_noise_recovers_actor_and_unit_noise():
    from flexiworld.planning.arcem import ActorPathCEMSolver

    actor = make_actor().eval()
    solver = ActorPathCEMSolver(
        model=SimpleNamespace(intent_actor=actor), num_samples=4, topk=2, n_steps=3
    )
    z, intent, previous = (torch.randn(2, 16) for _ in range(3))
    expected = actor.decode(z, intent, previous, 5)
    actual, _ = solver._actor_action_from_noise(z, intent, previous, torch.zeros(2, 10))
    torch.testing.assert_close(actual.reshape(2, 5, 2), expected)
    noisy, _ = solver._actor_action_from_noise(z, intent, previous, torch.ones(2, 10))
    torch.testing.assert_close(noisy[:, :2] - actual[:, :2], torch.full((2, 2), 0.2))


def test_history_padding_and_replan_remapping():
    processor = RawZeroActionProcessor(np.array([2.0, 4.0]), np.array([2.0, 2.0]))

    class Dataset:
        def load_chunk(self, episodes, starts, stops):
            return [{"action": np.array([[4.0, 6.0]]), "step_idx": np.array([0])}]

    history = expert_history(Dataset(), [0, 1], [0, 1], processor)
    np.testing.assert_allclose(history[0], np.tile([-1, -2], (5, 1)))
    np.testing.assert_allclose(history[1, -1], [1, 1])
    info = {
        "goal": torch.tensor([[1.0], [2.0]]),
        "action": torch.zeros(2, 1, 2),
        "_expert_previous_block": torch.tensor(history),
    }
    ledger = PrimitiveTailLedger("carry5")
    torch.testing.assert_close(
        ledger.override(info, is_replan=False, primitive_dim=2), torch.tensor(history)
    )
    plans = torch.randn(2, 25, 2)
    ledger.remember(info, plans)
    active = {"goal": info["goal"][[1]], "action": info["action"][[1]]}
    torch.testing.assert_close(
        ledger.override(active, is_replan=True, primitive_dim=2), plans[1:, -5:]
    )


def test_distributed_sampler_keeps_spans_homogeneous():
    ranks = [list(HomogeneousDistributedBatchSampler([16, 16], 2, 2, r, 0)) for r in range(2)]
    assert len(ranks[0]) == 8
    for a, b in zip(*ranks):
        assert not set(a) & set(b)
        assert len({index // 16 for index in a + b}) == 1
