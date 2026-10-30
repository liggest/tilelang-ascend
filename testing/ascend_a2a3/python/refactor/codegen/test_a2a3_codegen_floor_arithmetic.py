"""Actual A2/A3 emitters, executed by host C++ against Python integer arithmetic."""

import ctypes
import random
import re
import shutil
import subprocess

import pytest
from tilelang import tvm

from test_a2a3_codegen import _call, _make_func

T = tvm.tirx


def _emit(body, variables=(), buffers=(), backend="ascend"):
    marked = T.SeqStmt([_call("tl.ascend_src_code", "// FLOOR_HOST_BEGIN"), body, _call("tl.ascend_src_code", "// FLOOR_HOST_END")])
    f = _make_func(marked, buffers)
    f = T.PrimFunc([*f.params, *variables], f.body, buffer_map=f.buffer_map, attrs=f.attrs)
    builder = tvm.ffi.get_global_func(f"target.build.tilelang_{backend}_a2a3")
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    return builder(tvm.IRModule.from_expr(f), target, "A3").inspect_source()


def _result(expr):
    out = T.decl_buffer((1,), expr.dtype, name="floor_result", scope="local.var")
    return T.SeqStmt([T.AllocBuffer(out), T.BufferStore(out, expr, [0])])


def _math_source(source, occurrence=0):
    start = source.find("template <typename A, typename B>")
    helpers = source[start : source.index('extern "C" __global__')] if start >= 0 else ""
    body = source.split("// FLOOR_HOST_BEGIN")[occurrence + 1].split("// FLOOR_HOST_END", 1)[0]
    return helpers, body


def _host(tmp_path, source, dtype, parameters=None, setup="", extra="", occurrence=0):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    helpers, body = _math_source(source, occurrence)
    aliases = "using uchar = unsigned char; using ushort = unsigned short; using uint = unsigned int;"
    signed = dtype.startswith("int")
    bits = int(re.search(r"\d+", dtype).group())
    abi = "int64_t" if signed else "uint64_t"
    ctype = ("int" if signed else "uint") + str(bits) + "_t"
    parameters = parameters or f"{abi} input_a, {abi} input_b"
    setup = setup or f"{ctype} a = static_cast<{ctype}>(input_a); {ctype} b = static_cast<{ctype}>(input_b);"
    path = tmp_path / "floor.cpp"
    path.write_text(
        "#include <cstdint>\n#define __aicore__\n#define AICORE\n"
        + aliases
        + "\n"
        + helpers
        + extra
        + f'\nextern "C" {abi} run({parameters}) {{\n'
        + setup
        + body
        + "\nreturn floor_result;\n}\n"
    )
    lib = tmp_path / "floor.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", "-O2", str(path), "-o", str(lib)], check=True, capture_output=True, text=True
    )
    module = ctypes.CDLL(str(lib))
    fn = module.run
    fn.argtypes = [ctypes.c_int64 if signed else ctypes.c_uint64] * 2
    fn.restype = ctypes.c_int64 if signed else ctypes.c_uint64
    return fn, module


@pytest.mark.parametrize("backend", ["ascend", "pto"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"])
def test_native_floor_matches_independent_python_bigint(tmp_path, backend, dtype):
    bits, signed = int(re.search(r"\d+", dtype).group()), dtype.startswith("int")
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if dtype == "int8":
        cases = [(a, b) for a in range(low, high + 1) for b in range(low, high + 1) if b]
    else:
        boundary = {low, low + 1, high - 1, high, 0, 1, 7, 9, 16, 17}
        if signed:
            boundary.update([-17, -16, -9, -7, -1])
        boundary = sorted(v for v in boundary if low <= v <= high)
        cases = [(a, b) for a in boundary for b in boundary if b]
        rng = random.Random(71)
        cases += [
            (rng.randint(low, high), rng.choice([-1, 1]) * rng.randint(1, high))
            if signed
            else (rng.randint(low, high), rng.randint(1, high))
            for _ in range(128)
        ]
    a, b = T.Var("a", dtype), T.Var("b", dtype)
    for name, constructor in [("div", T.FloorDiv), ("mod", T.FloorMod)]:
        folder = tmp_path / name
        folder.mkdir()
        text = _emit(_result(constructor(a, b)), [a, b], backend=backend)
        fn, keep_alive = _host(folder, text, dtype)
        if not signed:
            assert "template <typename A" not in text
        for lhs, rhs in cases:
            expected = lhs // rhs if name == "div" else lhs % rhs
            # A non-representable quotient is outside the original TIR domain;
            # modulo MIN/-1 is representable and must still execute safely.
            if name == "div" and not low <= expected <= high:
                continue
            actual = fn(lhs, rhs)
            assert actual == expected, (backend, dtype, name, lhs, rhs, actual, expected)
            if name == "mod":
                assert (lhs // rhs) * rhs + actual == lhs
                assert actual == 0 or (actual > 0) == (rhs > 0)


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_proven_floor_keeps_original_direct_emission(backend):
    i = T.Var("i", "int32")
    for expr in [T.FloorDiv(i, T.IntImm("int32", 3)), T.FloorMod(i, T.IntImm("int32", 3)), T.FloorMod(-i, T.IntImm("int32", -3))]:
        text = _emit(T.For(i, 0, 16, T.ForKind.SERIAL, _result(expr)), backend=backend)
        assert "template <typename A" not in text
    a = T.Var("a", "int32")
    text = _emit(_result(T.FloorDiv(a * 3, T.IntImm("int32", 3))), [a], backend=backend)
    assert "template <typename A" not in text


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_shared_floor_node_does_not_leak_safe_branch_proof(tmp_path, backend):
    a, b = T.Var("a", "int32"), T.Var("b", "int32")
    expr = T.FloorMod(a, T.IntImm("int32", 3))
    out = T.decl_buffer((1,), "int32", name="floor_result", scope="local.var")
    body = T.SeqStmt([T.AllocBuffer(out), T.IfThenElse(a >= 0, T.BufferStore(out, expr, [0]), None), T.BufferStore(out, expr, [0])])
    text = _emit(body, [a, b], backend=backend)
    fn, keep_alive = _host(tmp_path, text, "int32")
    for value in [-17, -9, -1, 0, 1, 17]:
        assert fn(value, 0) == value % 3


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_select_keeps_division_lazy_and_has_no_invented_proof(tmp_path, backend):
    a, b = T.Var("a", "int32"), T.Var("b", "int32")
    expr = T.Select(a != 0, T.FloorDiv(b, a), T.IntImm("int32", 7))
    text = _emit(_result(expr), [a, b], backend=backend)
    fn, keep_alive = _host(tmp_path, text, "int32")
    assert fn(0, 99) == 7
    assert fn(3, -7) == -7 // 3
    expr = T.Select(a >= 0, T.FloorMod(a, T.IntImm("int32", 3)), T.IntImm("int32", 7))
    assert "template <typename A" in _emit(_result(expr), [a, b], backend=backend)


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_buffer_operands_evaluated_once_and_impure_bind_not_a_proof(tmp_path, backend):
    inp = T.decl_buffer((2,), "int32", name="input", scope="global")
    a, b = T.Var("a", "int32"), T.Var("b", "int32")
    text = _emit(_result(T.FloorMod(inp[0], inp[1])), buffers=[inp], backend=backend)
    extra = "struct HostTensor { int32_t values[2]; int calls = 0; int32_t GetValue(int i) { ++calls; return values[i]; } int32_t* operator+(int i) { ++calls; return values + i; } };"
    receiver = "input" if backend == "ascend" else "input_handle"
    setup = f"HostTensor {receiver}{{{{int32_t(input_a), int32_t(input_b)}}}};"
    # The generated expression is unchanged. The host tensor only replaces the
    # SDK GlobalTensor receiver, and counts the actual generated memory reads.
    helpers, body = _math_source(text)
    assert body.count("input.GetValue" if backend == "ascend" else "*(input_handle") == 2
    text = text.replace("// FLOOR_HOST_END", f"if ({receiver}.calls != 2) __builtin_trap();\n// FLOOR_HOST_END")
    fn, keep_alive = _host(tmp_path, text, "int32", setup=setup, extra=extra)
    assert fn(-7, 3) == -7 % 3
    body = T.SeqStmt([T.Bind(a, inp[0]), T.Bind(b, inp[1]), _result(T.FloorMod(a, b))])
    assert "template <typename A" in _emit(body, buffers=[inp], backend=backend)


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_int32_ir_shape_keeps_actual_int64_abi_arithmetic(tmp_path, backend):
    a, b = T.Var("a", "int32"), T.Var("b", "int32")
    inp = T.decl_buffer((a, 64), "float32", name="input", scope="global")
    for name, constructor in [("div", T.FloorDiv), ("mod", T.FloorMod)]:
        folder = tmp_path / name
        folder.mkdir()
        text = _emit(_result(constructor(a, b)), [b], [inp], backend)
        assert "int64_t a" in text
        fn, keep_alive = _host(folder, text, "int32", setup="int64_t a = input_a; int32_t b = int32_t(input_b);")
        for lhs, rhs in [(-(1 << 40) + 7, (1 << 30) - 3), ((1 << 40) + 7, -(1 << 30) + 3), (-(1 << 40) + 7, -(1 << 30) + 3)]:
            expected = lhs // rhs if name == "div" else lhs % rhs
            assert fn(lhs, rhs) == expected


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_unsigned_actual_promoted_domain_does_not_treat_max_as_minus_one(tmp_path, backend):
    # A real thread-index terminal is emitted as auto/uint32 even though its IR
    # Var is int32. Keep those original usual arithmetic conversions.
    index, b = T.Var("index", "int32"), T.Var("b", "int32")
    thread = T.IterVar(tvm.ir.Range(0, 4), index, T.IterVar.ThreadIndex, "blockIdx.x")
    body = T.AttrStmt(thread, "thread_extent", 4, _result(T.FloorMod(index, b)))
    text = _emit(body, [b], backend=backend)
    assert "a2a3_floor_mod(" in text
    # The actual generated body obtains the uint32 hardware index. Replace just
    # that SDK terminal with a host input; leave the helper and operation intact.
    text = re.sub(r"auto index = (?:AscendC::GetBlockIdx\(\)|get_block_idx\(\));", "uint32_t index = uint32_t(input_a);", text)
    text = text.replace("index = index / 2;", "index = index;")
    text = text.replace("if ASCEND_IS_AIV", "if (false)")
    fn, keep_alive = _host(
        tmp_path,
        text,
        "int32",
        setup="int32_t b = int32_t(input_b); uint64_t ffts_Addr = 0;",
        extra="inline void set_ffts_base_addr(uint64_t) {}",
    )
    assert fn(3, -1) == 3 % ((1 << 32) - 1)


@pytest.mark.parametrize("backend", ["ascend", "pto"])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_minus_one_mod_guard_precedes_exact_proof(tmp_path, backend, dtype):
    a, b = T.Var("a", dtype), T.Var("b", dtype)
    text = _emit(_result(T.FloorMod(a, T.IntImm(dtype, -1))), [a, b], backend=backend)
    fn, keep_alive = _host(tmp_path, text, dtype)
    bits = int(dtype[3:])
    assert fn(-(1 << (bits - 1)), 0) == 0
    assert fn(-17, 0) == 0


@pytest.mark.parametrize("backend", ["ascend", "pto"])
@pytest.mark.parametrize("kind,constructor", [("div", T.FloorDiv), ("mod", T.FloorMod)])
@pytest.mark.parametrize("reverse_keys", [False, True])
def test_module_helper_cannot_be_shadowed_in_later_function(tmp_path, backend, kind, constructor, reverse_keys):
    def function(name, argument):
        a, b = T.Var(argument, "int32"), T.Var("b", "int32")
        body = T.SeqStmt(
            [
                _call("tl.ascend_src_code", "// FLOOR_HOST_BEGIN"),
                _result(constructor(a, T.IntImm("int32", 3))),
                _call("tl.ascend_src_code", "// FLOOR_HOST_END"),
            ]
        )
        f = _make_func(body).with_attr("global_symbol", name)
        return T.PrimFunc([*f.params, a, b], f.body, buffer_map=f.buffer_map, attrs=f.attrs)

    names = ["alpha", "omega"] if not reverse_keys else ["omega", "alpha"]
    keys = [tvm.ir.GlobalVar(name) for name in names]
    module = tvm.IRModule({key: function(key.name_hint, "a") for key in keys})
    # The native builder iterates this actual persistent Map order. Keep its
    # keys and give the second function a local that shadows the first helper.
    ordered = list(module.functions)
    helper_name = "a2a3_floor_" + kind
    module.update_func(ordered[1], function(ordered[1].name_hint, helper_name))
    builder = tvm.ffi.get_global_func(f"target.build.tilelang_{backend}_a2a3")
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    text = builder(module, target, "A3").inspect_source()
    assert text.count("template <typename A, typename B>") == 1
    for index, argument in [(0, "a"), (1, helper_name)]:
        folder = tmp_path / str(index)
        folder.mkdir()
        fn, keep_alive = _host(
            folder, text, "int32", setup=f"int32_t {argument} = int32_t(input_a); int32_t b = int32_t(input_b);", occurrence=index
        )
        for value in [-17, -1, 0, 1, 17]:
            assert fn(value, 0) == (value // 3 if kind == "div" else value % 3)
