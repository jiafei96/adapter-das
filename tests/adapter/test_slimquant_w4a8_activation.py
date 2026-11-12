# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""fused_experts_w4a8 激活面契约（CPU 单测）。

Kimi-K3 的路由专家是 SituGLU，不是 silu：上游 SituAndMul 为
``beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)``。
本文件的参考实现直接按该式写，不引用 sglang 的 kernel；同时锁定
未支持的激活名必须显式报错，而不是静默退回 silu。
"""

import pytest
import torch

from hcu_adapter.slimquant import _apply_gated_activation


def _situ_ref(gate, up, beta, linear_beta=None):
    g = gate.float()
    u = up.float()
    g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (g * u).to(gate.dtype)


def test_situ_matches_upstream_formula():
    torch.manual_seed(0)
    gate = torch.randn(48, 64, dtype=torch.bfloat16) * 2.0
    up = torch.randn(48, 64, dtype=torch.bfloat16) * 2.0
    for beta, linear_beta in ((4.0, 25.0), (1.0, None), (2.5, 8.0)):
        out = _apply_gated_activation(gate, up, "situ", beta, linear_beta)
        ref = _situ_ref(gate, up, beta, linear_beta)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_situ_differs_from_silu():
    """A silu fallback would pass shape/dtype checks but be numerically wrong."""
    torch.manual_seed(1)
    gate = torch.randn(32, 64, dtype=torch.bfloat16) * 3.0
    up = torch.randn(32, 64, dtype=torch.bfloat16) * 3.0
    situ = _apply_gated_activation(gate, up, "situ", 4.0, 25.0)
    silu = _apply_gated_activation(gate, up, "silu", 4.0, 25.0)
    assert not torch.allclose(situ.float(), silu.float(), atol=1e-3)


def test_situ_without_linear_beta_leaves_up_uncapped():
    gate = torch.zeros(4, 8, dtype=torch.bfloat16)
    up = torch.full((4, 8), 100.0, dtype=torch.bfloat16)
    out = _apply_gated_activation(gate, up, "situ", 4.0, None)
    # gate term is 0 at gate==0 -> product is 0, and up must not be clipped
    assert torch.equal(out, torch.zeros_like(out))
    up2 = torch.full((4, 8), 100.0, dtype=torch.bfloat16)
    gate2 = torch.full((4, 8), 10.0, dtype=torch.bfloat16)
    capped = _apply_gated_activation(gate2, up2, "situ", 4.0, 25.0)
    uncapped = _apply_gated_activation(gate2, up2, "situ", 4.0, None)
    assert capped.abs().max() < uncapped.abs().max()


def test_silu_path_unchanged():
    torch.manual_seed(2)
    gate = torch.randn(16, 64, dtype=torch.bfloat16)
    up = torch.randn(16, 64, dtype=torch.bfloat16)
    out = _apply_gated_activation(gate, up, "silu", None, None)
    ref = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(out, ref)


def test_unknown_activation_is_loud():
    gate = torch.zeros(4, 8, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="gelu"):
        _apply_gated_activation(gate, gate, "gelu", None, None)


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - standalone runner
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)
