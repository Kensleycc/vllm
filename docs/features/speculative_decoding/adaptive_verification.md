# Adaptive Verification

The experimental `dspark_device_branch` option extends single-request
`dspark_copy_wide` with a Torch 2.14 CUDA Graph IF node. It leaves context KV
maintenance outside the conditional and skips native query/backbone/head work
only for a predicted wide round with a fresh GPU copy hit. It enables
`dspark_copy_lag1`: the host uses exactly the flag from two Target rounds ago,
without waiting. A missing flag narrows verification; if the GPU skipped the
proposal, the device supplies seven copy candidates and verifies all seven.
Invalid confidence snapshots carry a validity bit in the existing asynchronous
copy event. Both options default off; branching requires native Draft FULL
graphs, batch1/TP1 and the pinned conditional-node API, otherwise startup fails.

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability*, the running product of that request's per-position acceptance estimates, and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests: position 5 of a confident request can outrank position 1 of a doubtful one, so one request keeps its full block while another could be trimmed after a token or two.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification is supported for model-based drafters (EAGLE, MTP, DFlash, DSpark and standalone draft models). The per-position acceptance estimates come from one of two sources:

- **Confidence head**: DSpark checkpoints that ship a trained confidence head use its predictions directly.
- **Online acceptance estimator**: every other drafter predicts acceptance from its draft logits with a small logistic model, recalibrated during serving against the target's verification results.

## Usage

It is off by default. Enable it in the speculative config with `enable_adaptive_verification`. For example, with DSpark:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": true
  }'
```

Or with DFlash on MiMo-V2.5-Pro, whose drafter ships in the checkpoint's `dflash/` subdirectory:

```bash
hf download XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash --local-dir MiMo-V2.5-Pro-FP4-DFlash

vllm serve MiMo-V2.5-Pro-FP4-DFlash \
  --tensor-parallel-size 4 --trust-remote-code --max-model-len auto \
  --speculative-config '{
    "method": "dflash",
    "model": "MiMo-V2.5-Pro-FP4-DFlash/dflash",
    "num_speculative_tokens": 8,
    "enable_adaptive_verification": true
  }'
```

Set `enable_adaptive_verification: false` to verify the full block for every request.

## Requirements and limitations

- The attention backend must tolerate device-decided query lengths, since the CPU lengths only bound them from above. Backends that plan off the CPU lengths are excluded by the attention selector, and rejected at startup for models that hard-wire their backend.
- Full cudagraphs are required: step costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup.
- Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries) or pipeline parallelism (cost curves and confidences exist only on the last rank).
- Without a confidence head, `use_local_argmax_reduction` is rejected. The estimator needs the full draft logits, which local argmax reduction never materializes.

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```

## Experimental DSpark copy-wide rounds

`dspark_copy_wide` is disabled by default. It adds GPU N-gram copy candidates to
single-request DSpark AV while retaining the native seven-token draft backbone
on every ordinary proposal. The host selects a verification width from the latest
completed asynchronous copy flag; the GPU selects the candidate source using its
newest accepted context. Width selection does not wait for a new copy and does
not maintain a CPU suffix index.

For a seven-token DSpark checkpoint, add these speculative settings:

```json
{
  "method": "dspark",
  "model": "/models/dspark-block7",
  "num_speculative_tokens": 7,
  "draft_sample_method": "greedy",
  "enable_adaptive_verification": true,
  "dspark_copy_wide": true,
  "dspark_copy_n_min": 8,
  "dspark_copy_n_max": 16,
  "dspark_copy_k_wide": 31,
  "dspark_copy_width_rule": "latest"
}
```

Use V2, TP1/PP1/DP1 without context parallelism, and `max_num_seqs=1`.
`max_num_seqs>1` automatically disables copy-wide. Standard rejection and greedy
native drafts are required. The attention backend must support the wider variable
query lengths. With `TRITON_ATTN`, set `VLLM_TRITON_SPLITK_MAX_Q=33` and include
the real wide shape in the CUDA Graph capture sizes: 32 for K31, or 16 for K15.
Keep the native 1/2/4/8 sizes as well. Extra candidate/KV capacity and a wider
Target graph are reserved even while narrow rounds run.

Narrow rounds use the original DSpark candidates and AV confidence rule. Wide
rounds verify all K candidates: a fresh GPU hit selects copy candidates, while a
miss selects the native seven candidates followed by rejectable last-token
placeholders. Native confidence does not rank the copy candidate view. The
independent native confidence double buffer is available again on the following
narrow round. Context rows beyond the draft graph's eight-row prefix use the
existing eager KV staging before the unchanged draft graph.

`dspark_copy_width_rule="two_hits"` requires two distinct completed hit flags.
`dspark_copy_width_pattern` is an optional repeating boolean sequence for
diagnosing forced wide/narrow transitions; leave it unset during normal runs.
Request removal and slot reuse discard both pending flags and candidate state.
