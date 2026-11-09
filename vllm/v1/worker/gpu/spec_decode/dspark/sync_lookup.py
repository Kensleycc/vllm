# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU synchronization cost probe for native single-request DSpark AV."""

import contextlib
import time

import numpy as np


class SuffixIndex:
    """Incremental rolling hashes with exact collision verification."""

    def __init__(self, prompt, capacity):
        self.capacity = capacity
        self.tokens = np.empty(capacity, dtype=np.int32)
        self.prefix = np.empty(capacity + 1, dtype=np.uint64)
        self.hashes = np.zeros((14, capacity), dtype=np.uint64)
        self.powers = np.array(
            [pow(16777619, n, 1 << 64) for n in range(3, 17)], dtype=np.uint64
        )
        self.length = 0
        self.prefix[0] = 0
        self.append(prompt)

    def append(self, values):
        old = self.length
        end = old + len(values)
        if end > self.capacity:
            raise RuntimeError("CPU suffix probe exceeded its context capacity")
        self.tokens[old:end] = values
        for position in range(old, end):
            self.prefix[position + 1] = (
                int(self.prefix[position]) * 16777619 + int(self.tokens[position]) + 1
            ) & ((1 << 64) - 1)
        for index, n in enumerate(range(3, 17)):
            start, stop = max(0, old - n + 1), end - n + 1
            if start < stop:
                self.hashes[index, start:stop] = (
                    self.prefix[start + n : stop + n]
                    - self.prefix[start:stop] * self.powers[index]
                )
        self.length = end

    def query(self):
        length = self.length
        for n in range(min(16, length), 2, -1):
            current = length - n
            value = (
                int(self.prefix[length])
                - int(self.prefix[current]) * int(self.powers[n - 3])
            ) & ((1 << 64) - 1)
            positions = np.flatnonzero(self.hashes[n - 3, :current] == value)
            for position in positions[::-1]:
                if np.array_equal(
                    self.tokens[position : position + n],
                    self.tokens[current:length],
                ):
                    available = min(31, length - int(position) - n)
                    return n, int(position), available
        return 0, -1, 0


class DSparkSyncLookupProbe:
    def __init__(self, capacity):
        self.capacity = capacity
        self.request_id = None
        self.index = None
        self.records = []
        self.range_context = contextlib.nullcontext
        SuffixIndex(np.array([1, 2, 3, 1, 2, 3], dtype=np.int32), 64).query()

    def discard(self, request_id):
        if request_id == self.request_id:
            self.request_id = None
            self.index = None

    def observe(self, output, batch, request_states):
        assert batch.num_reqs == 1
        request_id = batch.req_ids[0]
        if request_id not in request_states.req_id_to_index:
            return
        with self.range_context(f"hybrid_sync_lookup/{len(self.records)}"):
            start = time.perf_counter_ns()
            output.copy_event.synchronize()
            copied = time.perf_counter_ns()
            setup_start = copied
            initialized = request_id != self.request_id
            if initialized:
                slot = int(batch.idx_mapping_np[0])
                prompt_len = int(request_states.prompt_len.np[slot])
                prompt = request_states.all_token_ids.gpu[slot, :prompt_len]
                self.index = SuffixIndex(prompt.cpu().numpy(), self.capacity)
                self.request_id = request_id
            setup_end = time.perf_counter_ns()
            count = int(output.num_sampled_tokens_np[0])
            assert self.index is not None
            self.index.append(output.sampled_token_ids[0, :count])
            hit = self.index.query()
            finish = time.perf_counter_ns()
            self.records.append(
                {
                    "step": len(self.records),
                    "prefill": bool(batch.has_prefill),
                    "context_tokens": self.index.length,
                    "sampled_tokens": count,
                    "initialized": initialized,
                    "wait_ms": (copied - start) / 1e6,
                    "setup_ms": (setup_end - setup_start) / 1e6,
                    "lookup_update_ms": (finish - setup_end) / 1e6,
                    "total_probe_ms": (finish - start) / 1e6,
                    "cpu_start_ns": start,
                    "cpu_end_ns": finish,
                    "match_n": hit[0],
                    "source_position": hit[1],
                    "available": hit[2],
                }
            )
