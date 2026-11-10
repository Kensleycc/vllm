# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2 GPU-accelerated n-gram speculator.

These tests target the Triton proposer in
``vllm.v1.worker.gpu.spec_decode.ngram.speculator`` and complement the CPU
``NgramProposer`` tests in ``test_ngram.py``. The GPU speculator follows a
slightly different policy than the CPU one: when multiple n-gram matches of
the same length exist, the GPU kernel picks the right-most (most recent)
match inside the active context, whereas the CPU implementation returns the
left-most. The expectations below reflect the GPU behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
from vllm.v1.worker.gpu.states import RequestState


@pytest.mark.parametrize("k", [15, 31])
@pytest.mark.parametrize("min_n", [3, 6, 8, 12])
@pytest.mark.parametrize(
    "case",
    ["miss", "nearest", "descending", "overlap", "boundary", "inactive", "random"],
)
def test_dspark_copy_wide_lookup_matches_exact_known_context(k, min_n, case):
    """The wide adapter keeps G0's longest/rightmost/known-only semantics."""
    import random

    from vllm.v1.worker.gpu.spec_decode.dspark.copy_wide import GPUCopyLookup

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a real NVIDIA CUDA environment.")
    rng = random.Random(20261109)
    values = {
        "miss": list(range(49)),
        "nearest": list(range(16))
        + [99, 98]
        + list(range(16))
        + [77, 76]
        + list(range(16)),
        "descending": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9],
        "overlap": [7] * 49,
        "boundary": list(range(16)) * 4,
        "inactive": list(range(16)) * 3,
        "random": [rng.randrange(7) for _ in range(64)],
    }[case]
    if case == "random":
        values[-12:] = values[9:21]
    active = case != "inactive"
    mapping = torch.tensor([1], dtype=torch.int64, device=DEVICE)
    token_ids = torch.full((2, 64), -999, dtype=torch.int32, device=DEVICE)
    token_ids[1, : len(values)] = torch.tensor(values, dtype=torch.int32, device=DEVICE)
    lengths = torch.tensor([0, len(values)], dtype=torch.int32, device=DEVICE)
    sampled = torch.tensor([int(active)], dtype=torch.int32, device=DEVICE)
    last = torch.tensor([[-7], [values[-1]]], dtype=torch.int64, device=DEVICE)
    lookup = GPUCopyLookup(64, min_n, 16, k, DEVICE)
    drafts, metadata = lookup.propose(token_ids, lengths, mapping, sampled, last)
    match = None
    if active:
        for n in range(min(16, len(values)), min_n - 1, -1):
            for position in range(len(values) - n - 1, -1, -1):
                if values[position : position + n] == values[-n:]:
                    match = (n, position)
                    break
            if match is not None:
                break
    expected = [values[-1]] * k
    if match is None:
        info = [0, 0, 0, len(values)]
    else:
        n, position = match
        continuation = values[position + n :][:k]
        expected[: len(continuation)] = continuation
        info = [1, n, len(values) - position - n, len(values)]
    assert drafts.cpu().tolist() == [expected]
    assert metadata.cpu().tolist() == [info]
    # The same allocated scan buffers must not retain the previous request's hit.
    token_ids[1] = torch.arange(64, device=DEVICE, dtype=torch.int32)
    lengths[1] = 64
    sampled.fill_(1)
    last[1] = 63
    drafts, metadata = lookup.propose(token_ids, lengths, mapping, sampled, last)
    assert metadata.cpu().tolist() == [[0, 0, 0, 64]]
    assert drafts.cpu().tolist() == [[63] * k]


@pytest.mark.parametrize("rule", ["latest", "two_hits"])
def test_dspark_copy_wide_flags_use_distinct_completed_rounds(rule):
    """Re-reading a stale hit does not fabricate a second hit or leak a slot."""
    from vllm.v1.worker.gpu.spec_decode.dspark.copy_wide import (
        CopyFlag,
        CopyWidthPolicy,
    )

    policy = CopyWidthPolicy(rule)
    assert policy.decide(1) == (False, None, None)
    policy.observe(CopyFlag(0, True, 8))
    assert policy.decide(2) == (rule == "latest", 0, 1)
    policy.observe(CopyFlag(0, True, 8))
    assert len(policy.flags) == 1
    policy.observe(CopyFlag(1, True, 12))
    assert policy.decide(3) == (True, 1, 1)
    policy.observe(CopyFlag(2, False, 0))
    assert policy.decide(4) == (False, 2, 1)
    reset = CopyWidthPolicy(rule)
    assert reset.decide(1) == (False, None, None)


@pytest.mark.parametrize("pattern", [[True], [True, False, True]])
def test_dspark_copy_wide_diagnostic_schedule_keeps_twenty_transitions(pattern):
    from vllm.v1.worker.gpu.spec_decode.dspark.copy_wide import CopyWidthPolicy

    policy = CopyWidthPolicy("latest", pattern)
    assert [policy.decide(r)[0] for r in range(1, 21)] == [
        pattern[(r - 1) % len(pattern)] for r in range(1, 21)
    ]


if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "CUDA required for NgramGPUSpeculator tests",
        allow_module_level=True,
    )

DEVICE = torch.device(current_platform.device_type)


def _make_vllm_config(
    min_n: int,
    max_n: int,
    k: int,
    max_num_seqs: int = 8,
    max_model_len: int = 64,
    method: str = "ngram_gpu",
) -> VllmConfig:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    speculative_config = SpeculativeConfig(
        method=method,
        prompt_lookup_min=min_n,
        prompt_lookup_max=max_n,
        num_speculative_tokens=k,
    )
    return VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        speculative_config=speculative_config,
    )


def _make_request_state(cfg: VllmConfig) -> RequestState:
    assert cfg.speculative_config is not None
    return RequestState(
        max_num_reqs=cfg.scheduler_config.max_num_seqs,
        max_model_len=cfg.model_config.max_model_len,
        max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens,
        num_speculative_steps=cfg.speculative_config.num_speculative_tokens,
        vocab_size=cfg.model_config.get_vocab_size(),
        device=DEVICE,
        use_dense_all_token_ids=True,
    )


def _make_speculator(
    min_n: int,
    max_n: int,
    k: int,
    max_num_seqs: int = 8,
    max_model_len: int = 32,
) -> NgramGPUSpeculator:
    cfg = _make_vllm_config(
        min_n=min_n,
        max_n=max_n,
        k=k,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    return NgramGPUSpeculator(cfg, DEVICE, _make_request_state(cfg))


def _propose(
    spec: NgramGPUSpeculator,
    rows: list[list[int]],
    seq_lens: list[int] | None = None,
    num_sampled: list[int] | None = None,
    last_sampled: list[int] | None = None,
    slots: list[int] | None = None,
) -> list[list[int]]:
    """Place each batch row at a request slot and run propose().

    Returns the drafts as python lists in batch order.
    """
    B = len(rows)
    if seq_lens is None:
        seq_lens = [len(r) for r in rows]
    if num_sampled is None:
        num_sampled = [1] * B
    if last_sampled is None:
        last_sampled = [0] * B
    if slots is None:
        slots = list(range(B))

    max_num_reqs = spec.max_num_reqs
    all_token_ids = spec.req_states.all_token_ids.gpu
    total_len = spec.req_states.total_len.gpu
    all_token_ids.zero_()
    total_len.zero_()
    last_sampled_t = torch.zeros((max_num_reqs, 1), dtype=torch.int64, device=DEVICE)
    for row, slot, seq_len, last in zip(rows, slots, seq_lens, last_sampled):
        if row:
            all_token_ids[slot, : len(row)] = torch.tensor(
                row, dtype=torch.int32, device=DEVICE
            )
        total_len[slot] = seq_len
        last_sampled_t[slot, 0] = last

    idx_mapping = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    input_batch = SimpleNamespace(num_reqs=B, idx_mapping=idx_mapping)

    drafts = spec.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device=DEVICE),
        aux_hidden_states=None,
        num_sampled=torch.tensor(num_sampled, dtype=torch.int32, device=DEVICE),
        num_rejected=torch.zeros(B, dtype=torch.int32, device=DEVICE),
        last_sampled=last_sampled_t,
        next_prefill_tokens=torch.zeros(B, dtype=torch.int32, device=DEVICE),
        temperature=torch.zeros(B, dtype=torch.float32, device=DEVICE),
        seeds=torch.zeros(B, dtype=torch.int64, device=DEVICE),
        dp_sync_state=None,
    )
    return drafts.cpu().tolist()


@pytest.mark.parametrize("max_model_len", [32, 300])
def test_no_match_clears_previous_proposal(max_model_len):
    spec = _make_speculator(min_n=2, max_n=2, k=2, max_model_len=max_model_len)
    row = [0] * (max_model_len - 5) + [1, 2, 3, 1, 2]
    assert _propose(spec, [row]) == [[3, 1]]
    drafts = _propose(spec, [[1, 2, 3, 4, 5]], last_sampled=[42])
    assert drafts == [[42, 42]]


def test_no_4gram_match_only():
    """No 4-gram match in [1,2,3,4,1,2,3] → 0 valid drafts."""
    spec = _make_speculator(min_n=4, max_n=4, k=2)
    drafts = _propose(spec, [[1, 2, 3, 4, 1, 2, 3]], last_sampled=[7])
    assert drafts == [[7, 7]]


def test_falls_back_to_3gram_when_4gram_missing():
    """No 4-gram match but a 3-gram match exists → propose [4, 1]."""
    spec = _make_speculator(min_n=3, max_n=4, k=2)
    drafts = _propose(spec, [[1, 2, 3, 4, 1, 2, 3]])
    assert drafts == [[4, 1]]


def test_prefers_longer_ngram():
    """Prefer a 4-gram match over a more recent 3-gram match."""
    spec = _make_speculator(min_n=3, max_n=4, k=2)
    drafts = _propose(spec, [[1, 2, 3, 4, 50, 51, 2, 3, 4, 60, 61, 1, 2, 3, 4]])
    assert drafts == [[50, 51]]


def test_picks_longest_match_among_2_3_4_grams():
    """Prefer a 3-gram match over a more recent 2-gram match."""
    spec = _make_speculator(min_n=2, max_n=4, k=2)
    drafts = _propose(spec, [[2, 3, 4, 50, 51, 3, 4, 60, 61, 1, 2, 3, 4]])
    assert drafts == [[50, 51]]


@pytest.mark.parametrize("max_model_len", [32, 128, 257, 1025])
def test_picks_rightmost_when_multiple_matches(max_model_len):
    """Pick the last valid match across blocks, ignoring trailing tokens."""
    spec = _make_speculator(min_n=3, max_n=3, k=2, max_model_len=max_model_len)
    padding = [0] * (spec.block_l - 5) if spec.n_blocks > 1 else []
    row = [1, 2, 3, 100] + padding + [1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]
    drafts = _propose(spec, [row + [1, 2, 3, 999, 1, 2, 3]], seq_lens=[len(row)])
    assert drafts == [[300, 1]]


def test_short_context_yields_zero_valid():
    """The only length-2 window overlaps the suffix itself → no match."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts = _propose(spec, [[5, 6]], last_sampled=[99])
    assert drafts == [[99, 99]]


def test_zero_sampled_disables_proposal():
    """num_sampled==0 disables proposals for that request regardless of match."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts = _propose(spec, [[1, 2, 3, 1, 2]], num_sampled=[0], last_sampled=[77])
    assert drafts == [[77, 77]]


def test_tail_falls_back_when_few_tokens_after_match():
    """Fewer than k tokens after the match → the tail falls back.

    Tokens: [1, 2, 1, 2] (seq_len=4). Suffix (1, 2) matches at position 0
    (the match at position 2 is the suffix itself and is excluded). With
    k=3, only 2 slots map to tokens inside the context.
    """
    spec = _make_speculator(min_n=2, max_n=2, k=3)
    drafts = _propose(spec, [[1, 2, 1, 2]], last_sampled=[55])
    assert drafts == [[1, 2, 55]]


def test_multibatch_mixed():
    """Mixed batch: row 0 matches, row 1 has no match."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts = _propose(
        spec,
        [[1, 2, 3, 1, 2], [4, 5, 6]],
        last_sampled=[10, 20],
    )
    assert drafts[0] == [3, 1]
    assert drafts[1] == [20, 20]


def test_multibatch_independent_choice_of_n():
    """Each row independently picks its longest matched n."""
    spec = _make_speculator(min_n=2, max_n=3, k=2)
    drafts = _propose(
        spec,
        [
            [9, 1, 2, 3, 8, 1, 2, 3],  # 3-gram (1,2,3) at idx 1 → [8, 1]
            [7, 1, 2, 9, 1, 2],  # 2-gram (1,2) at idx 1 → [9, 1]
        ],
    )
    assert drafts[0] == [8, 1]
    assert drafts[1] == [9, 1]


def test_min_n_eq_1():
    """min_n=max_n=1 — single-token n-grams always match if context > 1."""
    spec = _make_speculator(min_n=1, max_n=1, k=2)
    drafts = _propose(spec, [[1, 2, 3, 4, 1]])
    assert drafts == [[2, 3]]


def test_noncontiguous_idx_mapping():
    """propose() reads token rows in place via idx_mapping (non-contiguous)."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts = _propose(
        spec,
        [[7, 8, 9, 7, 8], [1, 2, 3, 1, 2]],
        slots=[3, 0],
    )
    assert drafts == [[9, 7], [3, 1]]


def test_copy_width_lag1_ignores_newer_and_missing_flags():
    from vllm.v1.worker.gpu.spec_decode.dspark.copy_wide import (
        CopyFlag,
        CopyWidthPolicy,
    )

    policy = CopyWidthPolicy("latest", lag1=True)
    policy.observe(CopyFlag(0, True, 8))
    policy.observe(CopyFlag(1, False, 0))
    assert policy.decide(1) == (False, None, None)
    assert policy.decide(2) == (True, 0, 1)
    assert policy.decide(3) == (False, 1, 1)
    policy.observe(CopyFlag(3, True, 8))
    assert policy.decide(4) == (False, None, None)


@pytest.mark.parametrize(
    "old,fresh", [(False, False), (False, True), (True, False), (True, True)]
)
def test_branch_predicate_and_copy_publication(old, fresh):
    from vllm.v1.worker.gpu.spec_decode.dspark.copy_wide import _publish_drafts_kernel
    from vllm.v1.worker.gpu.spec_decode.dspark.device_branch import (
        DSparkDeviceBranch,
        valid_confidences,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a real NVIDIA CUDA environment.")
    device = torch.device("cuda")
    branch = DSparkDeviceBranch(device)
    branch.previous_hit.fill_(old)
    metadata = torch.zeros((1, 6), dtype=torch.int32, device=device)
    metadata[0, 0].fill_(fresh)
    branch.predict(metadata, 1)
    native = torch.arange(7, device=device, dtype=torch.int64)[None]
    copy = torch.arange(31, device=device, dtype=torch.int64)[None] + 100
    narrow = torch.empty_like(native)
    wide = torch.empty_like(copy)
    last = torch.full((1, 1), 99, device=device, dtype=torch.int64)
    _publish_drafts_kernel[(1,)](
        native,
        copy,
        metadata,
        narrow,
        wide,
        last,
        branch.run_backbone,
        True,
        31,
        7,
        32,
        num_warps=1,
    )
    skipped = old and fresh
    assert branch.run_backbone.item() == (not skipped)
    assert metadata[0, 4:].tolist() == [int(old), int(skipped)]
    assert narrow.tolist() == (copy[:, :7] if skipped else native).tolist()
    expected = copy if fresh else torch.cat([native, last.expand(1, 24)], dim=1)
    assert torch.equal(wide, expected)
    confidence = torch.full((1, 7), float("nan") if skipped else 0.25, device=device)
    sanitized = valid_confidences(confidence, branch.run_backbone)
    assert sanitized.tolist() == [[1.0 if skipped else 0.25] * 7]
    branch.reset()
    assert not branch.previous_hit.item() and branch.run_backbone.item()


def test_construction_validates_speculative_config():
    spec = _make_speculator(min_n=2, max_n=3, k=2)
    assert spec.min_n == 2
    assert spec.max_n == 3
    assert spec.num_speculative_steps == 2
    # No-op hooks must not raise.
    spec.init_cudagraph_manager(None)
    spec.capture()
