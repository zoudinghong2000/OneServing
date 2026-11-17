# Experimental RedKnot P40b-I2 interfaces

This branch adds a serving-owned model family and KV allocator for the P40b-I2 experiment. It is based on redknot-p32a2-pr2 (6213d8b); it does not replace the PR #3 TP2 model implementation.

Serving owns the 64-layer forward, workspaces, online KV pool, physical page allocation, device submission and compile cache. The optional RedKnot provider supplies inline attention/GDN operators, three FFN phases and cache policy. The phases are expanded before PyPTO parsing so allocation scopes stay identical to the measured P40b model. There are no per-layer Python callbacks.

Install the compatible alternative provider from [Muyuan's experimental entry](https://gitcode.com/Windwindzzz/muyuan/tree/redknot-p40b-i2-r1/plugins/redknot-oneserving/experimental/interfaces), then start:

```bash
export ONESERVING_BACKEND=redknot PREFIX_CACHE=0
export REDKNOT_CONFIG=/path/to/interfaces/default.toml
export ONESERVING_COMPILE_CACHE=/path/to/writable/cache
bash run.sh --backend long-context --model-root /path/to/Qwen3.8-27B --devices 0 --max-active 1
```

The supported family is full Qwen3.8-27B BF16, 32K, one active request. Geometry derives a default 1,025 physical pages, each 128 tokens. ONESERVING_KV_PAGES or ONESERVING_KV_BUDGET_MIB can select a larger pool; the compiled shape and toolchain are part of the cache identity. This does not make shapes universally dynamic, automatically consume all free HBM, or validate 64K or ACL Graph.

The plugin keeps sparse_ffn=true/false. Inference ownership and the effective page geometry are exposed through /metrics provenance.operator_interface. Startup logs serving-compile keys and cache hits. Native execution with the provider disabled continues through the existing serving path.

The frozen six-case, same-card serial experiment measured P40b / I2 reuse TTFT of 1.532 / 1.499 s and restore latency of 91.3 / 95.5 ms. It did not reproduce the previous extra 400 ms. 59 of 60 complete logits vectors were bitwise identical; the known zstd safe reuse difference remains unresolved. These are short-output timing/logit checks, not full-answer accuracy evaluation.

CPU ABI, frozen-function and page-lifecycle checks are shipped with the compatible provider. Raw NPU evidence, source hashes, per-case results and version history are indexed in the Muyuan benchmark report. No new model math is introduced by this repository publication.
