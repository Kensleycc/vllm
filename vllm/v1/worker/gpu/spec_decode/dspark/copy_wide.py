# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-width/GPU-source DSpark copy rounds; no extra host synchronization."""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (
    _ngram_finalize_kernel,
    _ngram_scan_kernel,
)

if TYPE_CHECKING:
    import numpy as np

    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


@dataclass(frozen=True)
class CopyFlag:
    round: int
    hit: bool
    match_n: int


class CopyWidthPolicy:
    """Only completed flags enter this request-local policy."""

    def __init__(
        self, rule: str, pattern: list[bool] | None = None, lag1: bool = False
    ):
        if rule not in ("latest", "two_hits"):
            raise ValueError("Unknown copy-width rule")
        if pattern is not None and not pattern:
            raise ValueError("A diagnostic width pattern must be nonempty")
        self.rule = rule
        self.lag1 = lag1
        self.pattern = tuple(pattern) if pattern is not None else None
        self.flags: deque[CopyFlag] = deque(maxlen=2)

    def observe(self, flag: CopyFlag) -> None:
        if not self.flags or flag.round > self.flags[-1].round:
            self.flags.append(flag)

    def decide(self, round_id: int) -> tuple[bool, int | None, int | None]:
        latest = self.flags[-1] if self.flags else None
        if self.lag1:
            latest = next(
                (flag for flag in self.flags if flag.round == round_id - 2), None
            )
        if self.pattern is not None and round_id > 0:
            wide = self.pattern[(round_id - 1) % len(self.pattern)]
        elif self.rule == "two_hits":
            wide = len(self.flags) == 2 and all(flag.hit for flag in self.flags)
        else:
            wide = latest is not None and latest.hit
        generation = latest.round if latest is not None else None
        lag = round_id - 1 - generation if generation is not None else None
        return wide, generation, lag


@triton.jit
def _match_info_kernel(
    scores,
    lengths,
    mapping,
    sampled,
    metadata,
    L: tl.constexpr,
    NB: tl.constexpr,
    NB_PO2: tl.constexpr,
):
    i = tl.arange(0, NB_PO2)
    score = tl.max(tl.load(scores + i, mask=i < NB, other=0), axis=0)
    slot = tl.load(mapping)
    length = tl.load(lengths + slot)
    hit = (score > 0) & (tl.load(sampled) > 0)
    n = tl.where(hit, (score - 1) // (L + 1), 0)
    pos = tl.where(hit, (score - 1) % (L + 1), 0)
    available = tl.where(hit, length - pos - n, 0)
    tl.store(metadata, hit.to(tl.int32))
    tl.store(metadata + 1, n.to(tl.int32))
    tl.store(metadata + 2, available.to(tl.int32))
    tl.store(metadata + 3, length.to(tl.int32))


class GPUCopyLookup:
    """Reuse the native GPU N-gram kernels with an independent K-wide output."""

    def __init__(
        self,
        max_model_len: int,
        min_n: int,
        max_n: int,
        k: int,
        device: torch.device,
        metadata_width: int = 4,
    ):
        if not 1 <= min_n <= max_n <= 16 or k not in (15, 31):
            raise ValueError("Invalid DSpark copy lookup bounds")
        self.max_model_len, self.min_n, self.max_n, self.k = (
            max_model_len,
            min_n,
            max_n,
            k,
        )
        self.metadata_width = metadata_width
        self.block_l = (
            256
            if max_model_len >= 1024
            else 128
            if max_model_len >= 256
            else 64
            if max_model_len >= 64
            else max(16, triton.next_power_of_2(max_model_len))
        )
        self.n_blocks = triton.cdiv(max_model_len, self.block_l)
        self.scratch = torch.zeros((1, self.n_blocks), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((1, k), dtype=torch.int64, device=device)

    def propose(
        self,
        token_ids: torch.Tensor,
        total_len: torch.Tensor,
        mapping: torch.Tensor,
        sampled: torch.Tensor,
        last_sampled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ngram_scan_kernel[(1, self.n_blocks)](
            token_ids,
            token_ids.stride(0),
            mapping,
            total_len,
            sampled,
            self.scratch,
            self.n_blocks,
            self.max_model_len,
            self.min_n,
            self.max_n,
            triton.next_power_of_2(self.max_n),
            self.block_l,
            num_warps=4,
            num_stages=2,
        )
        _ngram_finalize_kernel[(1,)](
            token_ids,
            token_ids.stride(0),
            mapping,
            total_len,
            sampled,
            last_sampled.view(-1),
            self.scratch,
            self.n_blocks,
            self.drafts,
            self.max_model_len,
            self.n_blocks,
            self.k,
            triton.next_power_of_2(self.k),
            triton.next_power_of_2(self.n_blocks),
            num_warps=2,
            num_stages=1,
        )
        metadata = torch.empty(
            (1, self.metadata_width), dtype=torch.int32, device=token_ids.device
        )
        _match_info_kernel[(1,)](
            self.scratch,
            total_len,
            mapping,
            sampled,
            metadata,
            self.max_model_len,
            self.n_blocks,
            triton.next_power_of_2(self.n_blocks),
            num_warps=1,
        )
        return self.drafts, metadata


@triton.jit
def _publish_drafts_kernel(
    native,
    copy,
    metadata,
    native_out,
    wide_out,
    last,
    run_backbone,
    BRANCH: tl.constexpr,
    K: tl.constexpr,
    NATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.arange(0, BLOCK)
    hit = tl.load(metadata) != 0
    if BRANCH:
        skipped = ~tl.load(run_backbone)
        native_id = tl.load(native + col, mask=(col < NATIVE) & ~skipped, other=0)
        native_id = tl.where(
            skipped, tl.load(copy + col, mask=col < NATIVE, other=0), native_id
        )
    else:
        native_id = tl.load(native + col, mask=col < NATIVE, other=0)
    fallback = tl.where(col < NATIVE, native_id, tl.load(last))
    copy_id = tl.load(copy + col, mask=col < K, other=0)
    tl.store(native_out + col, native_id, mask=col < NATIVE)
    tl.store(wide_out + col, tl.where(hit, copy_id, fallback), mask=col < K)


@triton.jit
def _stage_drafts_kernel(
    native,
    wide,
    destination,
    K: tl.constexpr,
    WIDE: tl.constexpr,
    NATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.arange(0, BLOCK)
    if WIDE:
        value = tl.load(wide + col, mask=col < K, other=-1)
    else:
        value = tl.load(native + col, mask=col < NATIVE, other=-1)
    tl.store(destination + col, value, mask=col < K)


class DSparkCopyWide:
    """Single-request buffers and event-query-only host width selection."""

    def __init__(self, config: "VllmConfig", states: "RequestState"):
        spec = config.speculative_config
        assert spec is not None and spec.dspark_copy_wide
        self.states = states
        self.k, self.native_k = spec.dspark_copy_k_wide, spec.num_speculative_tokens
        self.rule, self.pattern = (
            spec.dspark_copy_width_rule,
            spec.dspark_copy_width_pattern,
        )
        self.lag1 = spec.dspark_copy_lag1
        from vllm.v1.worker.gpu.spec_decode.dspark.device_branch import (
            DSparkDeviceBranch,
        )

        self.device_branch: DSparkDeviceBranch | None = None
        if spec.dspark_device_branch:
            self.device_branch = DSparkDeviceBranch(
                states.device, spec.dspark_branch_prediction_pattern
            )
        self.lookup = GPUCopyLookup(
            config.model_config.max_model_len,
            spec.dspark_copy_n_min,
            spec.dspark_copy_n_max,
            self.k,
            states.device,
            6 if self.device_branch is not None else 4,
        )
        self.native_drafts = torch.zeros(
            (1, self.native_k), dtype=torch.int64, device=states.device
        )
        self.wide_drafts = torch.zeros(
            (1, self.k), dtype=torch.int64, device=states.device
        )
        self.pending: deque[tuple[int, int, np.ndarray, torch.cuda.Event]] = deque()
        self.generation = 0
        self.req_id: str | None = None
        self.round_id = 0
        self.policy = CopyWidthPolicy(self.rule, self.pattern, self.lag1)
        self.current: dict = {}
        self.last_payload = None
        self.record_history = False
        self.history: list[dict] = []
        logger.info(
            "DSpark copy-wide enabled: native=%d K=%d n=[%d,%d] "
            "rule=%s pattern=%s; lag1=%s device_branch=%s",
            self.native_k,
            self.k,
            spec.dspark_copy_n_min,
            spec.dspark_copy_n_max,
            self.rule,
            self.pattern,
            self.lag1,
            self.device_branch is not None,
        )

    def discard(self, req_id: str) -> None:
        if req_id != self.req_id:
            return
        self.req_id = None
        self.generation += 1
        self.pending.clear()
        self.policy = CopyWidthPolicy(self.rule, self.pattern, self.lag1)
        if self.device_branch is not None:
            self.device_branch.reset()
        self.last_payload = None
        self.current = {}
        self.native_drafts.zero_()
        self.wide_drafts.zero_()

    def add_request(self, req_id: str) -> None:
        if self.req_id is not None:
            self.discard(self.req_id)
        self.req_id = req_id
        self.generation += 1
        self.round_id = 0
        self.policy = CopyWidthPolicy(self.rule, self.pattern, self.lag1)
        if self.device_branch is not None:
            self.device_branch.reset()

    def prepare_batch(self, req_id: str, has_prefill: bool) -> bool:
        assert req_id == self.req_id
        while self.pending and self.pending[0][3].query():
            generation, round_id, values, _ = self.pending.popleft()
            if generation == self.generation:
                self.policy.observe(
                    CopyFlag(round_id, bool(values[0, 0]), int(values[0, 1]))
                )
        wide, decision_round, lag = self.policy.decide(self.round_id)
        wide = wide and not has_prefill
        fallback_possible = False
        if self.device_branch is not None and not wide and not has_prefill:
            # A missing r-2 D2H flag can disagree with the GPU prediction. Reserve
            # all seven slots without reading the current skip flag on the CPU.
            fallback_possible = self.round_id >= 2 and decision_round is None
            if self.pattern is not None and self.round_id >= 2:
                fallback_possible |= any(
                    flag.round == self.round_id - 2 and flag.hit
                    for flag in self.policy.flags
                )
            pattern = self.device_branch.pattern
            if pattern is not None and self.round_id:
                fallback_possible |= pattern[(self.round_id - 1) % len(pattern)]
        self.current = dict(
            copy_wide=wide,
            wide_round=self.round_id,
            copy_flag_round=decision_round,
            copy_flag_lag=lag,
            av_confidence_valid=not wide,
            candidate_source="dspark" if not wide else "pending_gpu_flag",
            branch_fallback_possible=fallback_possible,
        )
        if self.last_payload is not None:
            self.current["_copy_wide_payload"] = self.last_payload
        if self.record_history:
            self.history.append(dict(self.current))
        return wide

    def stage_drafts(self, mapping: torch.Tensor) -> None:
        _stage_drafts_kernel[(1,)](
            self.native_drafts,
            self.wide_drafts,
            self.states.draft_tokens,
            self.k,
            self.current["copy_wide"],
            self.native_k,
            triton.next_power_of_2(self.k),
            num_warps=1,
        )

    def propose_copy(self, batch: "InputBatch", sampled: torch.Tensor) -> torch.Tensor:
        _, metadata = self.lookup.propose(
            self.states.all_token_ids.gpu,
            self.states.total_len.gpu,
            batch.idx_mapping,
            sampled,
            self.states.last_sampled_tokens,
        )
        if self.device_branch is not None:
            self.device_branch.predict(metadata, self.round_id)
        return metadata

    def publish_output(self, output: "AsyncOutput") -> None:
        values = output.copy_wide_payload_np
        assert values is not None
        self.pending.append((self.generation, self.round_id, values, output.copy_event))
        self.last_payload = values

    def publish_drafts(self, native: torch.Tensor, metadata: torch.Tensor) -> None:
        _publish_drafts_kernel[(1,)](
            native,
            self.lookup.drafts,
            metadata,
            self.native_drafts,
            self.wide_drafts,
            self.states.last_sampled_tokens,
            self.device_branch.run_backbone if self.device_branch is not None else None,
            self.device_branch is not None,
            self.k,
            self.native_k,
            triton.next_power_of_2(self.k),
            num_warps=1,
        )
        self.states.draft_tokens.copy_(self.wide_drafts)
        self.round_id += 1
