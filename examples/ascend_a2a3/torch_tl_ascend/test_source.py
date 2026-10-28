import argparse
import torch


def ref_flash_attention(q, k, v):
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1]) ** 0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)


def _run_original(flash_attention_fwd):
    B, S, H, D = 4, 4096, 16, 128

    torch.manual_seed(0)

    q = torch.randn((B, H, S, D), dtype=torch.float16)
    k = torch.randn((B, H, S, D), dtype=torch.float16)
    v = torch.randn((B, H, S, D), dtype=torch.float16)

    torch.npu.synchronize()
    print("init successful!")

    kernel = flash_attention_fwd(B, S, H, D)
    output = kernel(q, k, v)
    ref_output = ref_flash_attention(q, k, v)
    torch.npu.synchronize()

    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

    print("Test Passed!")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Original fixed BHSD [4,16,4096,128] flash-attention test")
    return parser.parse_args(argv)


def _require_owned_cache():
    import os
    from pathlib import Path

    requested = os.environ.get("TILELANG_CACHE_DIR")
    if not requested:
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running the packaged-source test")
    import tilelang

    if Path(tilelang.env.TILELANG_CACHE_DIR).resolve() != Path(requested).resolve():
        raise RuntimeError("TILELANG_CACHE_DIR conflicts with the active cache-directory override; test was refused")


def __getattr__(name):
    if name == "flash_attention_fwd":
        from torch_tl_ascend.op_source.flash_attn_bhsd import flash_attention_fwd

        return flash_attention_fwd
    raise AttributeError(name)


def main(argv=None):
    parse_args(argv)
    _require_owned_cache()
    import torch_npu  # noqa: F401
    from torch_tl_ascend.op_source.flash_attn_bhsd import flash_attention_fwd

    with torch.device("npu"), torch.random.fork_rng(devices=[]):
        _run_original(flash_attention_fwd)


if __name__ == "__main__":
    main()
