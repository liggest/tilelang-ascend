"""CPU allocator-control tests; protocol tokens are not kernel/golden outputs."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[4] / "examples_experiment/mha_sink_bwd_bhsd/mha_sink_bwd_bhsd.py"


def _scope(allocator):
    tree = ast.parse(_SOURCE.read_text())
    names = {"_zero_ws_local", "_zero_ws_lock", "_orig_torch_empty"}
    functions = {"_torch_empty_is_zeroing", "_patched_torch_empty", "_zeroed_npu_workspace"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
    ]
    torch = SimpleNamespace(empty=allocator)
    env = dict(torch=torch, threading=threading, contextmanager=contextmanager)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SOURCE), "exec"), env)
    assert torch.empty is allocator  # Extraction itself must not install a patch.
    return torch, env


class _Token:
    def __init__(self, device="npu", error=None):
        self.device = SimpleNamespace(type=device)
        self.zeros = 0
        self.error = error

    def zero_(self):
        self.zeros += 1
        if self.error is not None:
            raise self.error
        return self


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_mha_scope_forwards_allocation_then_original_zero_condition(device):
    token = _Token(device)
    calls = []
    marker = object()

    def allocator(*args, **kwargs):
        calls.append((args, kwargs, token.zeros))
        return token

    torch, env = _scope(allocator)
    with env["_zeroed_npu_workspace"]():
        assert torch.empty(marker, dtype=marker) is token
    assert calls == [((marker,), {"dtype": marker}, 0)]
    assert token.zeros == int(device == "npu")
    assert torch.empty is allocator
    assert env["_torch_empty_is_zeroing"]() is False


def test_mha_scope_entry_existing_wrapper_and_nested_zero_once():
    calls = []
    token = _Token()
    torch, env = _scope(lambda: None)

    def wrapper():
        calls.append("entry")
        return token

    torch.empty = wrapper
    with env["_zeroed_npu_workspace"]():
        installed = torch.empty
        with env["_zeroed_npu_workspace"]():
            assert torch.empty is installed
            assert torch.empty() is token
        assert torch.empty is installed
        assert env["_torch_empty_is_zeroing"]() is True
    assert torch.empty is wrapper
    assert calls == ["entry"]
    assert token.zeros == 1


@pytest.mark.parametrize("phase", ["allocation", "zero", "body"])
def test_mha_scope_preserves_exception_identity_and_restores(phase):
    error = RuntimeError(phase)
    token = _Token(error=error if phase == "zero" else None)

    def allocator():
        if phase == "allocation":
            raise error
        return token

    torch, env = _scope(allocator)
    with pytest.raises(RuntimeError) as caught, env["_zeroed_npu_workspace"]():
        if phase == "body":
            raise error
        torch.empty()
    assert caught.value is error
    assert torch.empty is allocator
    assert env["_torch_empty_is_zeroing"]() is False


def test_mha_scope_other_thread_keeps_entry_allocator_semantics():
    # No .device field: the foreign thread must not even inspect the token.
    token = object()
    error = ValueError("original allocator")

    def allocator(fail=False):
        if fail:
            raise error
        return token

    torch, env = _scope(allocator)
    with ThreadPoolExecutor(max_workers=1) as pool, env["_zeroed_npu_workspace"]():
        assert pool.submit(torch.empty).result(timeout=3) is token
        future = pool.submit(torch.empty, True)
        with pytest.raises(ValueError) as caught:
            future.result(timeout=3)
        assert caught.value is error
    assert torch.empty is allocator


def test_mha_scope_serializes_same_component_and_restores_after_both():
    token = _Token("cpu")

    def allocator():
        return token

    torch, env = _scope(allocator)
    started = threading.Event()
    entered = threading.Event()

    def worker():
        started.set()
        with env["_zeroed_npu_workspace"]():
            entered.set()
            return torch.empty()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with env["_zeroed_npu_workspace"]():
            future = pool.submit(worker)
            assert started.wait(3)
            assert not entered.wait(0.05)
        assert future.result(timeout=3) is token
    assert entered.is_set()
    assert torch.empty is allocator


def test_mha_scope_does_not_clobber_unrelated_replacement():
    def original():
        return None

    def foreign():
        return None

    torch, env = _scope(original)
    with env["_zeroed_npu_workspace"]():
        torch.empty = foreign
    assert torch.empty is foreign
    assert env["_torch_empty_is_zeroing"]() is False


def test_mha_scope_original_k1_only_consumer_and_dynamic_runtime_allocator():
    tree = ast.parse(_SOURCE.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_attention")
    scopes = [n for n in ast.walk(klass) if isinstance(n, ast.With)]
    assert len(scopes) == 1
    assert ast.unparse(scopes[0].items[0].context_expr) == "_zeroed_npu_workspace()"
    assert len(scopes[0].body) == 1
    assert ast.unparse(scopes[0].body[0]) == "o = kernel_fwd(q, k, v, sinks_broad_kernel, mask_fwd)"
    # The shared adapter performs an attribute lookup at allocation time; no
    # alternate execution mode or shared allocator hook is introduced here.
    pyx = _SOURCE.parents[2] / "tilelang/jit/adapter/cython/cython_wrapper.pyx"
    assert "tensor_list[i] = torch.empty(*shape, dtype=dtype, device=device)" in pyx.read_text()
