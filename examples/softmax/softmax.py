import tilelang
from tilelang import DataType, language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def softmax(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            
            a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            exp_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            max_val_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            max_cur_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            sum_val_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            sum_cur_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            tmp_ub = T.alloc_ub([3 * DataType(dtype).bits // 8 * block_M // VEC_NUM * block_N], "uint8")

            with T.Scope("V"):
                T.tile.fill(max_val_ub, -2**30)
                for by in T.serial(n_num):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM:bx * block_M + (vid + 1) * block_M // VEC_NUM,
                             by * block_N:(by + 1) * block_N], a_ub)
                    T.reduce_max(a_ub, max_cur_ub, tmp_ub, dim=-1)
                    T.tile.max(max_val_ub, max_val_ub, max_cur_ub)

                T.tile.fill(sum_val_ub, 0.0)
                for by in T.serial(n_num):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM:bx * block_M + (vid + 1) * block_M // VEC_NUM,
                             by * block_N:(by + 1) * block_N], a_ub)
                    for i in T.serial(block_M // VEC_NUM):
                        T.tile.sub(a_ub[i, :], a_ub[i, :], max_val_ub[i])
                    T.tile.exp(exp_ub, a_ub)
                    T.reduce_sum(exp_ub, sum_cur_ub, tmp_ub, dim=-1)
                    T.tile.add(sum_val_ub, sum_val_ub, sum_cur_ub)

                for by in T.serial(n_num):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM:bx * block_M + (vid + 1) * block_M // VEC_NUM,
                             by * block_N:(by + 1) * block_N], a_ub)
                    for i in T.serial(block_M // VEC_NUM):
                        T.tile.sub(a_ub[i, :], a_ub[i, :], max_val_ub[i])
                    T.tile.exp(exp_ub, a_ub)
                    for i in T.serial(block_M // VEC_NUM):
                        T.tile.div(exp_ub[i, :], exp_ub[i, :], sum_val_ub[i])
                    T.copy(exp_ub, B[bx * block_M + vid * block_M // VEC_NUM:bx * block_M + (vid + 1) * block_M // VEC_NUM,
                                    by * block_N:(by + 1) * block_N])

    return main


torch.manual_seed(0)
test_configs = [
    (64, 64, 32, 64, "float"),
    (256, 256, 64, 128, "float"),
    (1024, 1024, 64, 128, "float"),
    (1024, 4096, 64, 256, "float"),
    (4096, 1024, 128, 128, "float"),
]

for M, N, block_M, block_N, dtype in test_configs:
    print(f"Testing softmax with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")
    func = softmax(M, N, block_M, block_N, dtype=dtype)
    print("Init successful!")
    a = torch.randn(M, N, dtype=getattr(torch, dtype)).npu()
    b = func(a)
    ref_b = torch.softmax(a, dim=-1)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-1, atol=1e-1)
    print("Test passed!")

print("Kernel Output Match!")