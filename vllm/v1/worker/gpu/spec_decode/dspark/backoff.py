# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Forced single-request DSpark warm-OFF schedule."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class ForcedDSparkBackoff:
    on_steps: int = 4
    off_tokens: int = 4
    start_off: bool = False
    request_id: str | None = field(default=None, init=False)
    proposal_valid: bool = field(default=False, init=False)
    probe_next: bool = field(default=False, init=False)
    context_only: bool = field(default=False, init=False)
    _on_remaining: int = field(default=0, init=False)
    _off_remaining: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.on_steps < 1 or self.off_tokens < 1:
            raise ValueError("on_steps and off_tokens must be positive")

    def discard(self, request_id: str) -> None:
        if self.request_id == request_id:
            self.request_id = None
            self.proposal_valid = False
            self.probe_next = False

    def verification_budget(self, request_id: str, width: int) -> int | None:
        if self.request_id != request_id or not self.proposal_valid:
            return 0
        # A restored proposal must be verified even while stale confidence is zero.
        return width if self.probe_next else None

    def select_context_only(self, request_id: str) -> bool:
        if self.request_id != request_id:
            self.request_id = request_id
            self.proposal_valid = False
            self.probe_next = False
            self._on_remaining = 0 if self.start_off else self.on_steps
            self._off_remaining = self.off_tokens
        self.context_only = self._on_remaining == 0
        if self.context_only:
            self._off_remaining -= 1
            if self._off_remaining == 0:
                self._on_remaining = self.on_steps
                self._off_remaining = self.off_tokens
        else:
            self._on_remaining -= 1
        return self.context_only

    def publish(self) -> None:
        valid = not self.context_only
        self.probe_next = valid and not self.proposal_valid
        self.proposal_valid = valid


def maybe_create_dspark_backoff(config: "VllmConfig") -> ForcedDSparkBackoff | None:
    additional = config.additional_config
    if not isinstance(additional, dict):
        return None
    options = additional.get("dspark_warm_off")
    if options is None:
        return None
    spec = config.speculative_config
    parallel = config.parallel_config
    if (
        spec is None
        or spec.method != "dspark"
        or not spec.enable_adaptive_verification
        or spec.draft_sample_method != "greedy"
        or config.scheduler_config.max_num_seqs != 1
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or not additional.get("dspark_single_token_graph", False)
    ):
        raise ValueError(
            "dspark_warm_off requires DSpark AV, greedy drafting, TP=PP=DP=1, "
            "max_num_seqs=1 and dspark_single_token_graph"
        )
    return ForcedDSparkBackoff(**options)
