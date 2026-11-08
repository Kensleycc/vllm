# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu.spec_decode.dspark.backoff import ForcedDSparkBackoff


@pytest.mark.parametrize("window", [1, 2, 4, 8])
def test_off_window_means_exactly_n_future_ar_tokens(window):
    backoff = ForcedDSparkBackoff(on_steps=2, off_tokens=window)
    for _ in range(2):
        assert not backoff.select_context_only("r")
        backoff.publish()
    assert backoff.verification_budget("r", 7) is None

    for _ in range(window):
        assert backoff.select_context_only("r")
        backoff.publish()
        assert backoff.verification_budget("r", 7) == 0

    assert not backoff.select_context_only("r")
    backoff.publish()
    assert backoff.verification_budget("r", 7) == 7
    assert not backoff.select_context_only("r")
    backoff.publish()
    assert backoff.verification_budget("r", 7) is None


def test_first_proposal_can_be_off_and_restore_a_probe():
    backoff = ForcedDSparkBackoff(on_steps=2, off_tokens=2, start_off=True)
    for _ in range(2):
        assert backoff.select_context_only("r")
        backoff.publish()
    assert not backoff.select_context_only("r")
    backoff.publish()
    assert backoff.verification_budget("r", 7) == 7


def test_new_request_and_preemption_do_not_inherit_an_off_window():
    backoff = ForcedDSparkBackoff(on_steps=1, off_tokens=8)
    assert not backoff.select_context_only("r")
    backoff.publish()
    assert backoff.select_context_only("r")
    backoff.publish()
    backoff.discard("unrelated")
    assert backoff.verification_budget("r", 7) == 0
    backoff.discard("r")
    assert not backoff.select_context_only("r")
    backoff.publish()
    assert not backoff.select_context_only("new")
    backoff.publish()
    assert backoff.verification_budget("new", 7) == 7


@pytest.mark.parametrize("on_steps,off_tokens", [(0, 1), (1, 0), (-1, 4)])
def test_empty_or_negative_windows_are_rejected(on_steps, off_tokens):
    with pytest.raises(ValueError, match="positive"):
        ForcedDSparkBackoff(on_steps=on_steps, off_tokens=off_tokens)
