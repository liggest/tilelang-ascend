# Original standalone linear-attention sources

Two complete sources from TA
`0deb8b93da11b7f0becbcded732d9633b7ca4e34` are mapped to
`linear_attention_and_rnn/{linear_attention_causal,linear_attention_normalize}.py`.
The exclusive ledger is `maint/ascend_a2a3/ecosystem_linear_rnn_ports.json`.
All device validation remains pending.

Both original no-argument CLI loops, factories, references and flags remain.
Imports do not clear cache, seed RNG, set print options, query hardware or run
kernels. Cache-clearing CLI execution requires a caller-owned directory:

```sh
TILELANG_CACHE_DIR=/tmp/my-owned-linear-cache \
python examples/ascend_a2a3/linear_attention_and_rnn/linear_attention_causal.py
```

The causal example keeps all 12 original inputs, including H20 and L4096.
Its actual lru keys are `(2,128,64)` and `(20,128,64)`, with 2 builds/10 reuses;
the original comment saying all shapes share H2 is not the actual matrix.
B/L stay dynamic, both explicit workspaces remain, and every persistent work
item retains the original workspace initialization and C/V handshake.
The original rtol is 1e-2, with atol 1e-2/2e-2/4e-2 according to the original L
branches. The raw cache-enabled bit is restored on CLI success or failure;
the original explicit lru still reuses its compiled kernels while JIT cache
is disabled.

Its original core query/except24 is resolved once, under a lock, at first
factory construction. An imported factory can therefore be used independently
of `main()` without import-time device initialization. The resolved value is
bound before the unchanged kernel algorithm. An explicit caller core override
is retained. 24 is the original query-failure fallback, not a device or
platform-identification claim.

The normalized example retains its original `(2,32,512,128,64,64)` input,
two-stage call order, intermediate accumulator, workspace, abs(Q/K), reference
and rtol/atol 1e-3. Its legacy wrapper-private target access now uses the main
JIT object's public target; only the default None maps to auto. The original
PTO/AscendC branches and unused temporary helpers remain intact.

Source verification explicitly samples core 20/core 24 and the original
except → 24 path without querying actual properties or initializing NPU context.
Eight requests per framework cover the four original factories under these
environment samples; source/config/ABI identities permit 6 actual SDK compiles
and 2 exact reuses per framework. Source evidence includes the complete dynamic
B/L ABI, named allocation intervals, every synchronization/barrier and the
integer proof for differently expanded but equal GM addresses.

These host/source/SDK checks do not execute the original 13 numerical inputs,
14 runtime calls or13 golden assertions. The test companion lives in the real
pytest root, separate from the original CLI tasks; it does not substitute for
an actual bench/device run. No performance claim is made.
