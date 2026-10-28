import sys
import shutil
import inspect
import importlib.util
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import tilelang
from tilelang import JITKernel
from tilelang.jit.adapter import CythonKernelAdapter

HERE = Path(__file__).parent
PACKAGE_ROOT = HERE.parent / "src" / "torch_tl_ascend"
PACKAGE_SOURCE_ROOT = PACKAGE_ROOT / "op_source"
EXAMPLE_ROOT = HERE.parent.parent

SO_NAME = "libop.so"


def _ensure_lib_path():
    # Both the shared and A2/A3 library generators remember a supplied load path.
    # The original process-wide LibraryGenerator.load_lib patch is unnecessary.
    pass


def _require_owned_cache():
    requested = os.environ.get("TILELANG_CACHE_DIR")
    if not requested:
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before compiling this interop operator")
    if Path(tilelang.env.TILELANG_CACHE_DIR).resolve() != Path(requested).resolve():
        raise RuntimeError("TILELANG_CACHE_DIR conflicts with the active cache-directory override; compilation was refused")


@lru_cache(maxsize=None)
def _load_op_module(source_path):
    # Preserve normal module/factory memo reuse without adding a caller sys.path.
    name = __name__ + "_adopted_flash_" + hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source_path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def get_jit_op_func():
    source_path = (EXAMPLE_ROOT / "flash_attention" / "flash_attn_bhsd.py").resolve()
    op_func = _load_op_module(source_path).flash_attention_fwd

    from tilelang.jit import JITImpl

    if not isinstance(op_func, JITImpl):
        raise AttributeError(f"{op_func!r} is not decorated by @tilelang.jit")

    return op_func


def get_kernel(op_func, *args, force_compile=False, **kwargs) -> JITKernel:
    _require_owned_cache()
    from tilelang.env import CacheState

    caller_cache_state = CacheState._enabled
    try:
        if force_compile:
            # As in TA, this bypasses shared caching only. A warm decorated
            # factory may still reuse its own memo; do not change that contract.
            tilelang.disable_cache()
        kernel = op_func(*args, **kwargs)
        return kernel
    finally:
        CacheState._enabled = caller_cache_state


def source_path_of(op_func):
    from tilelang.jit import JITImpl

    if isinstance(op_func, JITImpl):
        op_func = op_func.func
    while hasattr(op_func, "__wrapped__"):
        op_func = op_func.__wrapped__
    return Path(inspect.getfile(op_func))  # file of the original function


def so_path_of(kernel: JITKernel):
    adapter: CythonKernelAdapter = kernel.adapter
    if not adapter.lib_generator.libpath:
        raise RuntimeError("Compiled library path not found!")
    return Path(adapter.lib_generator.libpath)


def update_package_files(force_compile=False):
    _require_owned_cache()
    _ensure_lib_path()
    op_func = get_jit_op_func()

    op_source_path = source_path_of(op_func)
    print(f"Grabbing {op_source_path.suffix} of {op_func!r} ({op_source_path.as_posix()})")
    package_source_path = PACKAGE_SOURCE_ROOT / op_source_path.name
    PACKAGE_SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy(op_source_path, package_source_path)

    B, S, H, D = 4, 4096, 16, 128
    print("Grabbing .so of", op_func, f"with B={B}, S={S}, H={H}, D={D}")
    kernel = get_kernel(op_func, B, S, H, D, force_compile=force_compile)

    lib_so_path = so_path_of(kernel)
    package_so_path = PACKAGE_ROOT / SO_NAME

    shutil.copy(lib_so_path, package_so_path)
    print("Got", package_so_path.name)


def update_so(dst_so_path: Path, force_compile=False):
    _require_owned_cache()
    _ensure_lib_path()
    op_func = get_jit_op_func()

    B, S, H, D = 4, 4096, 16, 128
    print("Grabbing .so of", op_func, f"with B={B}, S={S}, H={H}, D={D}")
    kernel = get_kernel(op_func, B, S, H, D, force_compile=force_compile)

    lib_so_path = so_path_of(kernel)

    shutil.copy(lib_so_path, dst_so_path)
    print("Got", dst_so_path.as_posix())
