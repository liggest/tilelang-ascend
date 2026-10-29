"""CPU ownership/retry fixtures, never original attention outputs or kernels."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "examples/ascend_a2a3/attention_sink/example_gqa_sink_bwd_bhsd/test_gqa_sink_bwd_bhsd.py"


def _load():
    spec = importlib.util.spec_from_file_location("_attention_sink_owned_cleanup", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair():
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as source:
        source.write(b"// CPU cleanup ownership fixture, not an attention kernel\n")
    path = Path(source.name)
    library = path.with_suffix(".so")
    library.write_bytes(b"CPU fixture, not an ELF or numerical result")
    return path, library


@pytest.mark.parametrize("replace_inode,symlink", [(False, False), (True, False), (False, True)])
def test_cleanup_only_observed_identity_and_never_other_libraries(replace_inode, symlink):
    module = _load()
    source, owned = _pair()
    other_source, other = _pair()
    old_inode = owned.stat().st_ino
    try:
        module._remember_owned_tmp_library(str(source), str(owned))
        if replace_inode:
            replacement = owned.with_suffix(".replacement")
            replacement.write_bytes(b"different owner")
            replacement.replace(owned)
            assert owned.stat().st_ino != old_inode
        if symlink:
            owned.unlink()
            owned.symlink_to(other)
        module._cleanup_tmp_compilation_files()
        assert owned.exists() is (replace_inode or symlink)
        assert other.read_bytes() == b"CPU fixture, not an ELF or numerical result"
        assert source.is_file() and other_source.is_file()
    finally:
        for path in [source, owned, other_source, other]:
            path.unlink(missing_ok=True)


def test_nested_owned_pairs_share_registry_without_glob():
    module = _load()
    outer_source, outer = _pair()
    inner_source, inner = _pair()
    other_source, other = _pair()
    try:
        module._remember_owned_tmp_library(str(outer_source), str(outer))
        module._remember_owned_tmp_library(str(inner_source), str(inner))
        module._cleanup_tmp_compilation_files()
        module._cleanup_tmp_compilation_files()
        assert not outer.exists() and not inner.exists()
        assert other.exists() and outer_source.exists() and inner_source.exists()
    finally:
        for path in [outer_source, outer, inner_source, inner, other_source, other]:
            path.unlink(missing_ok=True)


def test_untrusted_exception_argv_cannot_claim_a_library():
    module = _load()
    source, library = _pair()
    try:
        try:
            raise RuntimeError(f"Compilation Failed! {['bisheng', str(source), '-o', str(library)]!r}")
        except RuntimeError as error:
            module._remember_failed_compilation(error)
        module._cleanup_tmp_compilation_files()
        assert library.exists()
    finally:
        source.unlink(missing_ok=True)
        library.unlink(missing_ok=True)


def test_actual_compiler_failure_frame_proves_argv_owner(monkeypatch):
    from tilelang.ascend.a2a3.compiler import CompileConfig, LibraryGenerator, _command_options

    module = _load()
    # This CPU protocol needs no native target kind, SDK discovery or device.
    config = CompileConfig(
        "ascendc",
        "A3",
        "npu",
        "dav-2201",
        "/not-executed/CPU-fixture",
        "/CPU-fixture/ascend",
        "/CPU-fixture/package",
        "/CPU-fixture/templates",
        "/tmp",
        None,
        (),
        _command_options("/CPU-fixture/ascend", "/CPU-fixture/package", "/CPU-fixture/templates", None),
    )
    # Real production method/frame/metadata, but subprocess is a CPU fixture.
    generator = LibraryGenerator(replace(config, compiler_path="/not-executed/CPU-fixture"))
    generator.update_lib_code("// no kernel or compiler process is executed")
    created = []

    def compiler_failure(command, **kwargs):
        source = next(Path(arg) for arg in command if arg.endswith(".cpp"))
        library = Path(command[-1])
        assert command[-2] == "-o" and library == source.with_suffix(".so")
        library.write_bytes(b"partial CPU fixture output")
        created.extend([source, library])
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", compiler_failure)
    other_source, other = _pair()
    try:
        with pytest.raises(RuntimeError, match="Compilation Failed") as captured:
            generator.compile_lib()
        module._remember_failed_compilation(captured.value)
        module._cleanup_tmp_compilation_files()
        assert created[0].exists() and not created[1].exists()
        assert other.exists()
    finally:
        for path in [*created, other_source, other]:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize("compile_error", [False, True])
def test_original_retry_count_return_and_exception_identity(compile_error, monkeypatch):
    module = _load()
    calls = []
    sleeps = []
    error = RuntimeError("Compilation Failed! CPU retry fixture") if compile_error else ValueError("original input error")
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module, "_capture_bisheng_stderr", lambda exc: ("fixture stderr", "CPU fixture"))

    def failing():
        calls.append(1)
        raise error

    with pytest.raises(type(error)) as captured:
        module._run_with_retry(failing)
    assert captured.value is error
    assert len(calls) == (3 if compile_error else 1)
    assert sleeps == ([1, 1] if compile_error else [])
    if compile_error:
        assert error._captured_stderr == "fixture stderr"
    result = object()
    assert module._run_with_retry(lambda: result) is result
