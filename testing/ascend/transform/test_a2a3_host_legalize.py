"""Hardware-free contract check for the first migrated A2/A3 native pass."""

from tilelang import tvm


def test_a2a3_host_legalize_preserves_body_and_tiling_attrs():
    make_pass = tvm.ffi.get_global_func("tl.transform.A2A3HostLegalize")
    original = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(1)).with_attr("global_symbol", "main")

    result = make_pass()(tvm.IRModule.from_expr(original))["main"]

    tvm.ir.assert_structural_equal(result.body, original.body)
    assert "tiling_map" in result.attrs
    assert "var_sequence" in result.attrs
    assert len(result.attrs["tiling_map"]) == 0
    assert len(result.attrs["var_sequence"]) == 0
