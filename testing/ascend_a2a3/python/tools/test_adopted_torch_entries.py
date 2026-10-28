"""Original interop CLI/import contracts, without NPU allocations or launch."""

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import tilelang

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/torch_tl_ascend"


def _module(name):
    spec = importlib.util.spec_from_file_location("adopted_interop_" + name, ASSET / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["test_source", "test_torch"])
def test_import_help_and_invalid_arguments_precede_backend_registration_package_load_or_execution(name, monkeypatch, capsys):
    original = builtins.__import__

    def guard(package, *args, **kwargs):
        if package.split(".")[0] in {"torch_npu", "torch_tl_ascend"}:
            pytest.fail("import/help cannot load an extension or register the NPU backend")
        return original(package, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(torch, "manual_seed", lambda *args: pytest.fail("import/help seeded caller"))
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: pytest.fail("import/help allocated inputs"))
    module = _module(name)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "fixed BHSD" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        module.main(["--shape", "1"])
    assert caught.value.code == 2


@pytest.mark.parametrize("conflict", [False, True])
def test_source_owned_cache_refusal_precedes_package_load_or_backend_registration(conflict, monkeypatch, tmp_path):
    module = _module("test_source")
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch_npu", "torch_tl_ascend"}:
            pytest.fail("refused source test loaded backend/package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=message):
        module.main([])


@pytest.mark.parametrize("name", ["test_source", "test_torch"])
def test_exception_before_original_execution_restores_caller_default_device_and_CPU_rng(name, monkeypatch, tmp_path):
    module = _module(name)
    path = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", path)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", path)
    factory_token = object()
    original = builtins.__import__

    def only_import_bridge(package, *args, **kwargs):
        if package in {"torch_npu", "torch_tl_ascend"}:
            return SimpleNamespace()
        if package == "torch_tl_ascend.op_source.flash_attn_bhsd":
            return SimpleNamespace(flash_attention_fwd=factory_token)
        return original(package, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", only_import_bridge)
    monkeypatch.setattr(module, "torch", SimpleNamespace(device=lambda name: torch.device("cpu"), random=torch.random))
    prior_rng = torch.random.get_rng_state().clone()
    entries = []

    def stop_before_original(*args):
        entries.append(args)
        assert torch.get_default_device() == torch.device("cpu")
        torch.manual_seed(0)
        raise RuntimeError("CPU sentinel before original inputs or kernel calls")

    monkeypatch.setattr(module, "_run_original", stop_before_original)
    with torch.device("meta"):
        with pytest.raises(RuntimeError, match="CPU sentinel before original"):
            module.main([])
        assert torch.get_default_device() == torch.device("meta")
    assert entries == [(factory_token,) if name == "test_source" else ()]
    assert torch.equal(torch.random.get_rng_state(), prior_rng)


@pytest.mark.parametrize("name", ["test_source", "test_torch"])
def test_original_CPU_reference_keeps_float32_softmax_and_float16_output(name):
    q = torch.zeros((1, 1, 2, 2), dtype=torch.float16)
    v = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]], dtype=torch.float16)
    assert torch.equal(_module(name).ref_flash_attention(q, q, v), torch.tensor([[[[3.0, 5.0], [3.0, 5.0]]]], dtype=torch.float16))


def test_original_packaged_source_alias_resolves_to_real_JIT_without_compilation(monkeypatch):
    package = os.environ.get("A2A3_TORCH_EXTENSION_PACKAGE")
    if not package:
        pytest.skip("requires the real independently built extension/source package")
    monkeypatch.syspath_prepend(str(Path(package).parent))
    from tilelang.jit import JITImpl

    module = _module("test_source")
    factory = module.flash_attention_fwd
    assert isinstance(factory, JITImpl) and factory._kernel_cache == {}
    assert Path(factory.func.orig_func.__code__.co_filename).resolve() == (Path(package) / "op_source/flash_attn_bhsd.py").resolve()


def test_shell_help_and_missing_owned_venv_stop_before_any_package_command(tmp_path):
    source = ASSET / "test_example.sh"
    help_result = subprocess.run(["bash", str(source), "--help"], capture_output=True, text=True, check=False)
    assert help_result.returncode == 0 and "task-owned" in help_result.stdout
    environment = os.environ.copy()
    environment.pop("A2A3_TORCH_INTEROP_VENV", None)
    result = subprocess.run(["bash", str(source)], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 1 and "refusing package changes" in result.stderr
    assert "Installing" not in result.stdout and "Running test_" not in result.stdout


def test_shell_heredoc_refuses_actual_descriptor_conflict(monkeypatch, tmp_path):
    source = (ASSET / "test_example.sh").read_text()
    guard = source.split("<<'PYGUARD' || exit $?\n", 1)[1].split("\nPYGUARD", 1)[0]
    monkeypatch.setenv("A2A3_TORCH_INTEROP_VENV", sys.prefix)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
    with pytest.raises(SystemExit, match="conflicts with the active cache-directory override"):
        exec(compile(guard, "actual_interop_shell_guard", "exec"), {})


@pytest.mark.parametrize(
    "version,failure,expected_commands,success",
    [
        ("missing", "", ["guard", "show", "install", "test_torch", "test_source"], True),
        ("0.1.0", "", ["guard", "show", "test_torch", "test_source"], True),
        ("0.0.0", "", ["guard", "show", "uninstall", "install", "test_torch", "test_source"], True),
        ("missing", "install", ["guard", "show", "install"], False),
        ("0.0.0", "uninstall", ["guard", "show", "uninstall"], False),
        ("0.0.0", "install", ["guard", "show", "uninstall", "install"], False),
        ("0.1.0", "test_torch", ["guard", "show", "test_torch"], False),
        ("0.1.0", "test_source", ["guard", "show", "test_torch", "test_source"], False),
    ],
)
def test_shell_preserves_original_version_install_test_order_and_failure_exits(version, failure, expected_commands, success, tmp_path):
    """Command-level simulation only; no fake kernels or numerical outcomes."""
    import json
    import shutil

    stage = tmp_path / "owned-script"
    stage.mkdir()
    shutil.copyfile(ASSET / "test_example.sh", stage / "test_example.sh")
    (stage / "src/torch_tl_ascend").mkdir(parents=True)
    (stage / "src/torch_tl_ascend/VERSION").write_text("0.1.0\n")
    for directory in ["build", "dist", "src/prior.egg-info"]:
        (stage / directory).mkdir()
        (stage / directory / "owned-sentinel").write_text("keep until original mismatch cleanup")
    tools = tmp_path / "command-protocol-only"
    tools.mkdir()
    python = tools / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,json\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "if args==['-']: sys.stdin.read(); command='guard'\n"
        "elif args[:3]==['-m','pip','show']: command='show'\n"
        "elif args[:3]==['-m','pip','uninstall']: command='uninstall'\n"
        "elif args[:2]==['setup.py','install']: command='install'; assert '--user' not in args\n"
        "elif args==['test_torch.py']: command='test_torch'\n"
        "elif args==['test_source.py']: command='test_source'\n"
        "else: raise AssertionError(args)\n"
        "with Path(os.environ['SIM_COMMAND_LOG']).open('a') as f: f.write(json.dumps({'command':command,'argv':args})+'\\n')\n"
        "if command=='show':\n"
        " version=os.environ['SIM_PACKAGE_VERSION']\n"
        " if version=='missing': raise SystemExit(1)\n"
        " print('Name: torch_tl_ascend\\nVersion: '+version)\n"
        "if command==os.environ['SIM_FAIL_COMMAND']:raise SystemExit(7)\n"
    )
    python.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    environment = os.environ.copy()
    environment.update(
        PATH=str(tools) + os.pathsep + os.environ["PATH"],
        SIM_COMMAND_LOG=str(log),
        SIM_PACKAGE_VERSION=version,
        SIM_FAIL_COMMAND=failure,
    )
    result = subprocess.run(["bash", str(stage / "test_example.sh")], env=environment, capture_output=True, text=True, check=False)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["command"] for row in commands] == expected_commands
    assert result.returncode == (0 if success else 1)
    assert ("All Test Passed!" in result.stdout) is success
    assert (stage / "build/owned-sentinel").exists() is (version != "0.0.0")
