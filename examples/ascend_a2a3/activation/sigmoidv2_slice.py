"""Original TA example; runtime effects execute only through main()."""

from tilelang.env import CacheState
import tilelang
import tilelang.ascend.a2a3.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoidv2():
    dtype = "float"

    @T.prim_func
    def main(
        input: T.Tensor([4, 8], dtype),
        output: T.Tensor([4, 8], dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            input_shared = T.alloc_ub((4, 8), dtype)
            output_shared = T.alloc_ub((4, 8), dtype)

            T.copy(input, input_shared)
            for i in range(4):
                T.tile.sigmoid(output_shared[i, :], input_shared[i, :])
            T.copy(output_shared, output)

    return main


def main():
    import torch_npu  # noqa: F401

    _prior_cache_enabled = CacheState.is_enabled()
    try:
        with torch.device("npu"):
            torch.manual_seed(42)
            tilelang.disable_cache()
            dtype = torch.float
            input = torch.randn([4, 8], dtype=dtype)
            func = sigmoidv2()
            print("init successful!")
            output = func(input)
            torch.npu.synchronize()
            ref_output = torch.sigmoid(input)
            torch.testing.assert_close(ref_output, output, rtol=0.01, atol=0.01)
            print("Kernel Output Match!")
    finally:
        if _prior_cache_enabled:
            CacheState.enable()
        else:
            CacheState.disable()


if __name__ == "__main__":
    main()
