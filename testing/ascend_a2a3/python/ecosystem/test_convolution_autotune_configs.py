"""Check the real tuner's callable boundary without compiling or probing devices."""

import importlib.util
import inspect
from itertools import product
from pathlib import Path

import pytest
from tilelang.autotuner import tuner as tuner_module


_ROOT = Path(__file__).resolve().parents[4]


_SHAPES = [
    pytest.param((128, 128, 128), id="perfect_alignment"),
    pytest.param((128, 1024, 128), id="m_padding"),
    pytest.param((128, 256, 128), id="n_padding"),
    pytest.param((128, 512, 128), id="k_padding"),
    pytest.param((128, 256, 128), id="all_dim_padding"),
    pytest.param((256, 2304, 256), id="multi_block"),
]
# Frozen original ordered candidates for all six padded CI requests.
_ORIGINAL_CONFIGS = [
    {"block_M": block_m, "block_N": block_n, "K_L1": block_k} for block_m, block_n, block_k in product((64, 128), (64, 128), (32, 64, 128))
]
pytestmark = pytest.mark.usefixtures("isolated_a2a3_cache")


def _example():
    path = _ROOT / "examples_experiment/convolution/example_convolution_autotune.py"
    spec = importlib.util.spec_from_file_location("_convolution_config_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("fixed", [False, True], ids=["tune", "fixed_cli_kwargs"])
def test_original_candidates_through_main_tuner_callback(shape, fixed, monkeypatch):
    example = _example()
    public = example.matmul
    factory = public.jit_impl.func
    kwargs = {"block_M": 128, "block_N": 128, "K_L1": 128} if fixed else {}
    original_parameters = (shape, tuple(kwargs.items()))
    autotuner = tuner_module.AutoTuner(factory, configs=public.configs)
    autotuner.set_kernel_parameters(original_parameters, inspect.signature(factory).parameters)
    assert public.warmup == 25 and public.rep == 100

    class ReachedCompilerBoundary(Exception):
        pass

    class HostOnlyBoundary:
        def freeze(self):
            # AutoTuner.run has already invoked the actual callback here.
            # Stop before target/toolchain discovery, caches or compilation.
            assert autotuner.configs == _ORIGINAL_CONFIGS
            assert example.get_configs(shape, kwargs) == _ORIGINAL_CONFIGS
            assert autotuner._kernel_parameters == original_parameters
            raise ReachedCompilerBoundary

    autotuner.compile_args = HostOnlyBoundary()
    # Logger initialization writes autotuner.log in cwd and is unrelated to
    # this callback contract. All callback invocation code remains production.
    monkeypatch.setattr(tuner_module, "_init_logger_handlers", lambda: None)
    with pytest.raises(ReachedCompilerBoundary):
        autotuner.run()
