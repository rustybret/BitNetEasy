# Copyright (c) Microsoft. All rights reserved.
# Metal backend for BitNet inference on Apple Silicon

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension
import torch
import os
import subprocess


def get_include_dirs():
    """Get include directories for PyTorch."""
    include_dirs = []
    torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
    include_dirs.append(torch_include)
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    if os.path.exists(torch_api_include):
        include_dirs.append(torch_api_include)
    return include_dirs


class BuildExtensionWithMetal(BuildExtension):
    """BuildExtension that also pre-compiles the Metal shader to a .metallib."""

    def build_extensions(self):
        super().build_extensions()
        self._compile_metallib()

    def _compile_metallib(self):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        metal_src = os.path.join(src_dir, "bitnet_kernels.metal")
        air_file  = os.path.join(src_dir, "bitnet_kernels.air")
        lib_file  = os.path.join(src_dir, "bitnet_kernels.metallib")

        if not os.path.exists(metal_src):
            print("Warning: bitnet_kernels.metal not found, skipping metallib compilation")
            return

        print("Compiling Metal shaders...")
        try:
            subprocess.run(
                ["xcrun", "metal", "-c", metal_src, "-o", air_file],
                check=True,
            )
            subprocess.run(
                ["xcrun", "metallib", air_file, "-o", lib_file],
                check=True,
            )
            os.remove(air_file)
            print(f"Metal library written to {lib_file}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Metal shader compilation failed: {e}")
            print("The extension will fall back to runtime shader compilation.")


include_dirs = get_include_dirs()

setup(
    name="bitnet_metal",
    version="0.1.0",
    ext_modules=[
        Extension(
            "bitnet_metal",
            sources=["metal_backend.mm"],
            include_dirs=include_dirs,
            extra_compile_args=["-std=c++17", "-ObjC++"],
            extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
            language="objc++",
        )
    ],
    cmdclass={"build_ext": BuildExtensionWithMetal},
    package_data={
        "": ["*.metal", "*.metallib"],
    },
)
