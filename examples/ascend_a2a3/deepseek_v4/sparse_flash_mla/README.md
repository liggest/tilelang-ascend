# DeepSeek-v4 sparse MLA for A2/A3

The SWA, HCA and CSA kernels, public API and reference helpers retain the
original inputs, thresholds and script entry points. Configure CANN and
Torch-NPU, then run from this directory:

```bash
python sparse_flash_mla_api.py
python sparse_flash_mla_kernel.py
```

The API script runs the three original bf16 FAST numerical scenarios. The
standalone kernel script retains its original SWA compilation example.
