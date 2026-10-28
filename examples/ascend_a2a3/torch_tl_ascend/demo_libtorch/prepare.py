import argparse
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "lib"


@lru_cache(maxsize=1)
def _compiler_helpers():
    spec = importlib.util.spec_from_file_location("torch_libtorch_compile_flash", HERE.parent / "compile_tl_op/flash_attention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare the original fixed flash-attention libop.so for libtorch")
    parser.parse_args(argv)
    if not os.environ.get("TILELANG_CACHE_DIR"):
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before preparing libop.so")
    flash_attention = _compiler_helpers()
    flash_attention._require_owned_cache()
    LIB.mkdir(exist_ok=True)
    flash_attention.update_so(LIB / flash_attention.SO_NAME)


if __name__ == "__main__":
    main()
