"""Pure stdlib benchmark/reference protocols; no msprof, runtime or NPU."""

import argparse
import builtins
import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[4] / "examples/ascend_a2a3/flash_attention/fa_opt"


def _module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("fa_tool_" + relative.replace("/", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", ["run.py", "reference/ascendc.py"])
def test_import_help_does_not_load_Torch_TileLang_or_run_subprocess(relative, monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "tilelang"}:
            pytest.fail("import/help loaded heavy runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    module = _module(relative)
    if relative == "run.py":
        monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("help launched process"))
        monkeypatch.setattr(module.sys, "argv", [str(ROOT / relative), "--help"])
        call = module.run_benchmark
    else:

        def call():
            return module.main(["--help"])

    with pytest.raises(SystemExit) as error:
        call()
    assert error.value.code == 0 and "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["zip", "product"])
@pytest.mark.parametrize("explicit_heads", [False, True])
def test_original_zip_product_head_fallback_matrix(mode, explicit_heads):
    module = _module("run.py")
    args = argparse.Namespace(
        B=[2, 1],
        S=[128, 256],
        H=[16, 32],
        D=[128, 512],
        q_heads=[12, 24] if explicit_heads else None,
        kv_heads=[1, 2] if explicit_heads else None,
        iter_mode=mode,
    )
    cases = module.generate_test_cases(args)
    assert len(cases) == (2 if mode == "zip" else (32 if explicit_heads else 16))
    assert cases[0] == ((2, 128, 12, 1, 128) if explicit_heads else (2, 128, 16, 16, 128))


def test_original_zip_length_error_is_exit1(capsys):
    module = _module("run.py")
    args = argparse.Namespace(B=[1], S=[128, 256], H=[16], D=[128], q_heads=None, kv_heads=None, iter_mode="zip")
    with pytest.raises(SystemExit) as error:
        module.generate_test_cases(args)
    assert error.value.code == 1 and "same length" in capsys.readouterr().out


@pytest.mark.parametrize("header,value,expected", [("OP Type", "12.5", 12.5), ("Op Type", "N/A", None), ("OP Type", "", None)])
def test_actual_csv_header_and_missing_time_protocol(header, value, expected, tmp_path, monkeypatch):
    module = _module("run.py")
    slept = []
    monkeypatch.setattr(module.time, "sleep", slept.append)
    directory = tmp_path / "PROF_actual" / "mindstudio_profiler_output"
    directory.mkdir(parents=True)
    (directory / "op_statistic_test.csv").write_text(f"{header},Total Time(us)\nmain_kernel,{value}\n")
    assert module.get_total_time(str(tmp_path), "main_kernel") == expected
    assert slept == [2.0]
    assert module.get_total_time(str(tmp_path / "missing"), "main_kernel") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Mismatched elements: 3/8\nAssertionError", "Mismatched elements: 3/8"),
        ("AssertionError: fail", "AssertionError"),
        ("Traceback (most recent call last):\nValueError: bad", "RuntimeError: ValueError: bad..."),
        ("no error", ""),
    ],
)
def test_original_error_summary_precedence(text, expected):
    assert _module("run.py").parse_error_info(text) == expected


@pytest.mark.parametrize("returncode,stdout,expected", [(0, "ok", "Pass"), (1, "no text", "Fail"), (0, "AssertionError", "Fail")])
def test_original_subprocess_status_uses_returncode_and_keywords(returncode, stdout, expected, monkeypatch):
    module = _module("run.py")
    recorded = []

    class Process:
        def __init__(self, *args, **kwargs):
            recorded.append((args, kwargs))
            self.returncode = returncode

        def communicate(self, timeout=None):
            assert timeout is None
            return stdout, None

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "get_total_time", lambda *args: None)
    result = module.run_and_analyze("original msprof command", "/unused", "main_kernel", "protocol only")
    assert result[1] == expected
    assert recorded[0][1]["start_new_session"] is True and recorded[0][1]["shell"] is True


def test_original_timeout_sends_group_SIGINT_and_waits_without_force_kill(monkeypatch):
    module = _module("run.py")
    calls = []

    class Process:
        pid = 12345
        returncode = -signal.SIGINT

        def communicate(self, timeout=None):
            calls.append(("communicate", timeout))
            if timeout is not None:
                raise module.subprocess.TimeoutExpired("original command", timeout)
            return "partial profiler output", None

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(module.os, "killpg", lambda pid, signum: calls.append((pid, signum)))
    monkeypatch.setattr(module, "get_total_time", lambda *args: 7.0)
    assert module.run_and_analyze("original command", "/unused", "main_kernel", "protocol only", timeout=2.0) == (7.0, "Timeout", "")
    assert calls == [("communicate", 2.0), (12345, signal.SIGINT), ("communicate", None)]


def test_original_shell_zip_flags_and_reference_default_args():
    shell = (ROOT / "bench.sh").read_text()
    assert "--B 2 2 1" in shell and "--S 131072 65536 32768" in shell
    assert "--q-heads 12 12 12" in shell and "--kv-heads 1 1 1" in shell
    assert "./auto_pipeline/h16_d128.py" in shell and "./reference/ascendc.py" in shell
    args = _module("reference/ascendc.py").parse_args([])
    assert (args.B, args.S, args.H, args.D, args.q_heads, args.kv_heads, args.no_check) == (4, 4096, 32, 512, None, None, False)


@pytest.mark.parametrize("tl_status,ac_status,expected", [("Pass", "Pass", "Pass"), ("Timeout", "Pass", "Both_Fail")])
def test_original_performance_commands_and_weak_report_status(tl_status, ac_status, expected, tmp_path, monkeypatch):
    module = _module("run.py")
    requests = []
    args = SimpleNamespace(
        tl="/original/tl.py", ascendc="/original/reference.py", kernel_tl="main_kernel", kernel_ascendc="FlashAttentionScore", timeout=3.0
    )

    def observe(command, output, kernel, description, timeout):
        requests.append((command, output, kernel, description, timeout))
        return (2.0, tl_status, "") if len(requests) == 1 else (4.0, ac_status, "")

    monkeypatch.setattr(module, "run_and_analyze", observe)
    _, row = module.run_performance_case((2, 131072, 12, 1, 128, args, str(tmp_path), "PERFORMANCE"))
    assert row[-1] == expected and row[-2] == "200.00%"
    assert "msprof --output=" in requests[0][0]
    assert "/original/tl.py --B 2 --S 131072 --q-heads 12 --kv-heads 1 --D 128" in requests[0][0]
    assert "--no-check" not in requests[0][0] and "--no-check" in requests[1][0]
    assert [r[2] for r in requests] == ["main_kernel", "FlashAttentionScore"]


@pytest.mark.parametrize(
    "kind,path,kernel", [("TL", "/original/tl.py", "main_kernel"), ("AC", "/original/reference.py", "FlashAttentionScore")]
)
def test_original_simulator_commands_preserve_flags(kind, path, kernel, tmp_path, monkeypatch):
    module = _module("run.py")
    observed = []
    args = SimpleNamespace(
        tl="/original/tl.py",
        ascendc="/original/reference.py",
        kernel_tl="main_kernel",
        kernel_ascendc="FlashAttentionScore",
        timeout=3.0,
        soc_version="Ascend910_9382",
    )

    def observe(*values, **kwargs):
        observed.append((values, kwargs))
        return 7.0, "Timeout", ""

    monkeypatch.setattr(module, "run_and_analyze", observe)
    assert module.run_sim_task((5, kind, 2, 131072, 12, 1, 128, args, str(tmp_path))) == (5, kind, 7.0, "Timeout", "")
    command = observed[0][0][0]
    assert 'msprof op simulator --soc-version=Ascend910_9382 --kernel-name="' + kernel + '"' in command
    assert path in command and "--no-check" in command
