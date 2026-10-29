# Complete original four-version PA benchmark

Run the same original entry, for example:

```bash
python examples/ascend_a2a3/sparse_flash_attention/bench_sfa/bench_sfa.py --file sparse_flash_attn_pa_no_cv_pipeline
```

The default remains `sparse_flash_attn_pa_no_cv_pipeline`. The other original
choices are `sparse_flash_attn_pa_baseline`, `sparse_flash_attn_pa_developer`
and `sparse_flash_attn_pa`. All four resolve only exact adjacent migrated leaf
files, without changing `sys.path` or accepting an unrelated same-named module.
No reference, factory, input, threshold or original failure/exit was rewritten.

The complete original `test_op` and `test_reject_invalid_topk` are retained.
Each selected version runs the original three default inputs with
KV_S2560/6400/48000, actualKV2560, B=T1/H128/D512/rope64/topk2048. The pipelined
version also runs the original topk128 and256 inputs at KV_S2560. Thus all four
CLI selections together contain fourteen numerical inputs. Each invocation
also preserves the original rejection checks for topk0/64/192.

The initializer and rejection factory are always imported from the pipelined
leaf, independently of `--file`, exactly as in TA. The CLI sets the NPU default
device and seed42 through the original public initializer. This standalone
process behavior is intentionally retained: `main()` does not promise to
restore a caller's default device or RNG. Only the raw cache-enabled bit is
restored in `finally`. The initializer itself is unchanged and is not patched
or wrapped in another `torch.device` context.

Importing this benchmark and calling `--help` do not invoke the initializer,
seed, disable cache, import TorchNPU, allocate tensors or compile/launch kernels.
TorchNPU registration is explicit only after parsing/valid selection and before
the original initializer/reference setup. The original full flow then runs in
its original order and prints `Kernel Output Match!` only after its original
assertions/rejection checks. No fake success marker or skipped case was added.
The golden remains `torch_npu.npu_sparse_flash_attention` first tuple output,
with dtype conversion and rtol/atol1e-2, equal_nan=True.

Nondevice validation checks both complete original function ASTs and every
original executable statement after only the explicit leaf-import conversion
and delayed TorchNPU import. Fourteen original requests map to the eight full
factory source requests already checked in PA1/PA2. Their initial SDK inputs,
identities, libraries and unchanged adjacent leaf bytes were verified and
reused by the original delivery; this history remains frozen.
Actual four-version benchmark/device execution remains pending.

`../GQA_pto.cpp` is deliberately empty: the fixed original Git blob was empty.
It is preserved as a source asset, not presented as a PTO implementation,
compiled library or numerical validation. Do not populate it with a new kernel
or count its existence as a supported operator.

## Current normal compiler refresh

All eight source requests were regenerated after restoration of the original
Simplify/RemoveNoOp/Combine phase contracts (normal compiler SHA256
`a9cefba866baa68644769faf202515a91490ac970b4dccb99078c8a6c40f867f`).
Four complete C++ files equal fixed-source TA after only header relocation;
the other four differ only by two int32 multiplication operand swaps. The
original fourteen CLI inputs have batch=seq_len=1 and cid0..23, so those
operands/products remain in1..24. Every named view's order, type, count,
offset and complete host ABI match the original TA source.

The changed modern inputs received six fresh successful public Bisheng
compilations plus two exact same-batch identity reuses (developer50/375 reuse
developer20). Full configs, actual source-before/ELF-after and two actual
680-header snapshots were preserved. No old TA or main ELF substitutes for a
changed input. Current host ON14/OFF14 pass; device matrix remains pending.
Evidence: `refactor_main/reports/p3_PA3_GRC3_root_a9_review/`.
