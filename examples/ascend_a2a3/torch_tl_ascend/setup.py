import importlib.util
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup

HERE = Path(__file__).parent
SRC = HERE / "src"
PACKAGE_PATH = SRC / "torch_tl_ascend"
VERSION_FILE = PACKAGE_PATH / "VERSION"


def _build_requested(arguments):
    # Metadata/help must not import TorchNPU, compile a kernel or write libop.so.
    commands = {"build", "build_py", "build_ext", "bdist", "bdist_wheel", "install", "develop", "editable_wheel"}
    return not any(arg in {"-h", "--help", "--help-commands", "-n", "--dry-run"} for arg in arguments) and any(
        arg in commands for arg in arguments
    )


def _extension_options(build_requested):
    sources = [cpp_path.as_posix() for cpp_path in SRC.glob("*.cpp")]
    if not build_requested:
        return {"ext_modules": [Extension("torch_tl_ascend._inner", sources)]}

    # Keep the original operator-before-extension build order so build_py also
    # includes libop.so. No build dependency is imported by metadata/help.
    from torch.utils.cpp_extension import BuildExtension, CppExtension
    import torch_npu

    spec = importlib.util.spec_from_file_location("torch_tl_ascend_build_flash", HERE / "compile_tl_op/flash_attention.py")
    flash_attention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flash_attention)
    flash_attention.update_package_files()
    torch_npu_path = Path(torch_npu.__file__).parent
    return {
        "ext_modules": [
            CppExtension(
                "torch_tl_ascend._inner",
                sources,
                include_dirs=[(torch_npu_path / "include").as_posix()],
                library_dirs=[PACKAGE_PATH.as_posix(), (torch_npu_path / "lib").as_posix()],
                libraries=["torch_npu"] + [so_path.stem.removeprefix("lib") for so_path in PACKAGE_PATH.glob("*.so")],
                extra_link_args=["-Wl,-rpath,$ORIGIN"],
            )
        ],
        "cmdclass": {"build_ext": BuildExtension},
    }


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    setup(
        name="torch_tl_ascend",
        version=VERSION_FILE.read_text(encoding="utf-8").strip(),
        description="Torch TL Ascend",
        packages=find_packages(where=SRC),
        package_dir={"": SRC.as_posix()},
        package_data={PACKAGE_PATH.name: ["*.so"]},
        script_args=arguments,
        **_extension_options(_build_requested(arguments)),
    )


if __name__ == "__main__":
    main()
