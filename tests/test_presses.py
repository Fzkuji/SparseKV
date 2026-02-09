# SPDX-License-Identifier: Apache-2.0

"""Tests for AdaSparseKV presses."""

import torch
import pytest


def _make_dummy_inputs(batch_size=2, num_heads=4, seq_len=256, head_dim=64):
    """Create dummy inputs for press testing."""
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim)
    attentions = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
    return keys, values, hidden_states, attentions


class MockModule:
    """Mock attention module."""
    layer_idx = 0
    head_dim = 64


class TestBlockDropoutPress:
    def test_training_mode_drops_blocks(self):
        from adasparse.presses.block_dropout_press import BlockDropoutPress

        press = BlockDropoutPress(block_size=32, drop_ratio=0.5, protect_start=4, protect_recent=32, training=True)
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        new_keys, new_values = press.compress(MockModule(), hidden_states, keys, values, attentions, {})

        # Should have fewer tokens
        assert new_keys.shape[2] < keys.shape[2]
        assert new_values.shape[2] < values.shape[2]
        assert new_keys.shape[2] == new_values.shape[2]

    def test_eval_mode_passthrough(self):
        from adasparse.presses.block_dropout_press import BlockDropoutPress

        press = BlockDropoutPress(training=False)
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        new_keys, new_values = press.compress(MockModule(), hidden_states, keys, values, attentions, {})

        assert new_keys.shape == keys.shape

    def test_zero_drop_ratio(self):
        from adasparse.presses.block_dropout_press import BlockDropoutPress

        press = BlockDropoutPress(drop_ratio=0.0, training=True)
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        new_keys, new_values = press.compress(MockModule(), hidden_states, keys, values, attentions, {})

        assert new_keys.shape == keys.shape

    def test_protects_sink_and_recent(self):
        from adasparse.presses.block_dropout_press import BlockDropoutPress

        press = BlockDropoutPress(
            block_size=16, drop_ratio=0.9, protect_start=4, protect_recent=16, training=True
        )
        keys, values, hidden_states, attentions = _make_dummy_inputs(seq_len=128)

        new_keys, _ = press.compress(MockModule(), hidden_states, keys, values, attentions, {})

        # Should always have at least protect_start + protect_recent tokens
        assert new_keys.shape[2] >= 20  # 4 + 16


class TestAdaptiveBlockDropoutPress:
    def test_with_attentions(self):
        from adasparse.presses.adaptive_dropout_press import AdaptiveBlockDropoutPress

        press = AdaptiveBlockDropoutPress(
            block_size=32, base_drop_ratio=0.3, training=True
        )
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        new_keys, new_values = press.compress(MockModule(), hidden_states, keys, values, attentions, {})

        assert new_keys.shape[2] <= keys.shape[2]

    def test_without_attentions_fallback(self):
        from adasparse.presses.adaptive_dropout_press import AdaptiveBlockDropoutPress

        press = AdaptiveBlockDropoutPress(
            block_size=32, base_drop_ratio=0.3, training=True
        )
        keys, values, hidden_states, _ = _make_dummy_inputs()

        new_keys, new_values = press.compress(MockModule(), hidden_states, keys, values, None, {})

        assert new_keys.shape[2] <= keys.shape[2]


class TestSoftThresholdPress:
    def test_training_soft_mask(self):
        from adasparse.presses.soft_threshold_press import SoftThresholdPress

        press = SoftThresholdPress(threshold=0.01, temperature=10.0, training=True)
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        new_keys, new_values = press.compress(MockModule(), hidden_states, keys, values, attentions, {})

        # Soft mask doesn't change shape, but modifies values
        assert new_keys.shape == keys.shape
        assert new_values.shape == values.shape

    def test_temperature_annealing(self):
        from adasparse.presses.soft_threshold_press import SoftThresholdPress

        press = SoftThresholdPress()
        press.anneal_temperature(step=50, total_steps=100, start_temp=1.0, end_temp=50.0)
        assert press.temperature > 1.0


class TestSparseRegPress:
    def test_entropy_reg(self):
        from adasparse.presses.sparse_reg_press import SparseRegPress

        press = SparseRegPress(reg_type="entropy", reg_weight=0.01, training=True)
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        # Should not modify KV cache
        new_keys, new_values = press.compress(MockModule(), hidden_states, keys, values, attentions, {})
        assert new_keys.shape == keys.shape

        # Should accumulate reg loss
        loss = press.get_reg_loss()
        assert loss.item() > 0

    def test_l1_reg(self):
        from adasparse.presses.sparse_reg_press import SparseRegPress

        press = SparseRegPress(reg_type="l1", reg_weight=0.01, training=True)
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        press.compress(MockModule(), hidden_states, keys, values, attentions, {})
        loss = press.get_reg_loss()
        assert loss.item() > 0

    def test_reset(self):
        from adasparse.presses.sparse_reg_press import SparseRegPress

        press = SparseRegPress(reg_type="entropy", training=True)
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        press.compress(MockModule(), hidden_states, keys, values, attentions, {})
        press.reset()
        loss = press.get_reg_loss()
        assert loss.item() == 0


class TestVariableBlockPress:
    def test_drops_blocks(self):
        from adasparse.presses.variable_block_press import VariableBlockDropoutPress

        press = VariableBlockDropoutPress(
            min_block_size=16, max_block_size=64, drop_ratio=0.5, training=True
        )
        keys, values, hidden_states, attentions = _make_dummy_inputs()

        new_keys, _ = press.compress(MockModule(), hidden_states, keys, values, attentions, {})
        assert new_keys.shape[2] < keys.shape[2]


class TestCurriculum:
    def test_linear(self):
        from adasparse.training.curriculum import LinearCurriculum

        c = LinearCurriculum(start_ratio=0.0, end_ratio=0.5, warmup_steps=100)
        assert c.get_ratio(0, 1000) == 0.0
        assert c.get_ratio(50, 1000) == 0.0  # still in warmup
        assert 0.0 < c.get_ratio(500, 1000) < 0.5
        assert abs(c.get_ratio(1000, 1000) - 0.5) < 0.01

    def test_step(self):
        from adasparse.training.curriculum import StepCurriculum

        c = StepCurriculum(start_ratio=0.0, end_ratio=0.5, warmup_steps=0, n_stages=5)
        r1 = c.get_ratio(0, 1000)
        r2 = c.get_ratio(999, 1000)
        assert r2 > r1

    def test_cosine(self):
        from adasparse.training.curriculum import CosineCurriculum

        c = CosineCurriculum(start_ratio=0.0, end_ratio=0.5, warmup_steps=0)
        assert c.get_ratio(0, 1000) == 0.0
        mid = c.get_ratio(500, 1000)
        assert 0.2 < mid < 0.3  # cosine midpoint ≈ 0.25


class TestSparsityMetrics:
    def test_compute_sparsity_metrics(self):
        from adasparse.evaluation.sparsity import compute_sparsity_metrics

        attention = torch.softmax(torch.randn(2, 4, 64, 64), dim=-1)
        metrics = compute_sparsity_metrics(attention, block_size=16)

        assert "effective_support" in metrics
        assert "top_30_coverage" in metrics
        assert "block_sparsity" in metrics
        assert metrics["effective_support"] > 0

    def test_sparse_attention_has_lower_support(self):
        from adasparse.evaluation.sparsity import effective_support

        # Uniform attention
        uniform = torch.ones(1, 1, 32, 32) / 32
        # Sparse attention (most mass on first token)
        sparse = torch.zeros(1, 1, 32, 32)
        sparse[..., 0] = 1.0

        assert effective_support(sparse) < effective_support(uniform)
