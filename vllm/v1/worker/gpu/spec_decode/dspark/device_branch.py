# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable GPU predicate for H1W, shared with the native Draft FULL graph."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _predict_branch_kernel(
    metadata,
    previous_hit,
    run_backbone,
    FORCE_PREDICTION: tl.constexpr,
):
    fresh = tl.load(metadata) != 0
    old = tl.load(previous_hit)
    prediction = (
        old if FORCE_PREDICTION < 0 else tl.full((), FORCE_PREDICTION != 0, tl.int1)
    )
    skip = prediction & fresh
    tl.store(run_backbone, ~skip)
    tl.store(metadata + 4, prediction.to(tl.int32))
    tl.store(metadata + 5, skip.to(tl.int32))
    tl.store(previous_hit, fresh)


class DSparkDeviceBranch:
    def __init__(
        self,
        device: torch.device,
        pattern: list[bool] | None = None,
        shadow_run: bool = False,
    ):
        if not str(torch.__version__).startswith("2.14.") or not all(
            hasattr(torch.cuda.CUDAGraph, name)
            for name in (
                "begin_capture_to_if_node",
                "end_capture_to_conditional_node",
            )
        ):
            raise RuntimeError(
                "DSpark device branching requires the pinned Torch 2.14 "
                "CUDAGraph conditional-node API"
            )
        self.previous_hit = torch.zeros((), dtype=torch.bool, device=device)
        self.run_backbone = torch.ones((), dtype=torch.bool, device=device)
        # Shadow execution refreshes native state but does not publish its output.
        self.execute_backbone = (
            torch.ones_like(self.run_backbone) if shadow_run else self.run_backbone
        )
        self.pattern = tuple(pattern) if pattern is not None else None

    def reset(self) -> None:
        self.previous_hit.zero_()
        self.run_backbone.fill_(True)

    def predict(self, metadata: torch.Tensor, round_id: int) -> None:
        force = (
            -1
            if self.pattern is None
            else int(self.pattern[round_id % len(self.pattern)])
        )
        _predict_branch_kernel[(1,)](
            metadata, self.previous_hit, self.run_backbone, force, num_warps=1
        )


@triton.jit
def _valid_confidence_kernel(
    values, valid, output, N: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    is_valid = tl.load(valid)
    p = tl.load(values + i, mask=(i < N) & is_valid, other=1.0)
    tl.store(output + i, p, mask=i < N)


def valid_confidences(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Publish a neutral estimate without reading skipped, stale head output."""
    output = torch.empty_like(values)
    _valid_confidence_kernel[(1,)](
        values,
        valid,
        output,
        values.numel(),
        triton.next_power_of_2(values.numel()),
        num_warps=1,
    )
    return output
