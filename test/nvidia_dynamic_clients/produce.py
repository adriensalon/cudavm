"""Intentionally regenerate the Windows x64 NVIDIA-linked client, outside candidate builds."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    toolkit = args.toolkit.resolve()
    inputs = root.parent / "kernel_launch"
    header = toolkit / "include/cuda.h"
    library = toolkit / "lib/x64/cuda.lib"
    if not header.is_file() or not library.is_file():
        parser.error("genuine NVIDIA headers/import library are required")
    if not re.search(r"^#define CUDA_VERSION\s+12040\s*$", header.read_text(), re.MULTILINE):
        parser.error("this pinned producer profile requires NVIDIA CUDA 12.4 headers")
    if not os.environ.get("VCToolsVersion") or not os.environ.get("WindowsSDKVersion"):
        parser.error("run in a Windows x64 Visual Studio C++ toolchain environment")
    output = root / "windows_x64/driver_client.exe"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cudavm-nvidia-client-") as folder:
        work = Path(folder)
        ptx = (inputs / "kernel_launch.ptx").read_text()
        cubin = (inputs / "kernel_launch.cubin").read_bytes()
        (work / "driver_images.hpp").write_text(
            '#pragma once\nstatic const char driver_ptx[] = R"client_ptx(' + ptx + ')client_ptx";\n'
            'alignas(8) static const unsigned char driver_cubin[] = {'
            + ','.join(str(byte) for byte in cubin) + '};\n', encoding="utf-8")
        command = [args.cxx, "-std=c++17", "-O2", "-D_MT", "-Xclang", "--dependent-lib=libcmt",
                   "-I" + str(toolkit / "include"), "-I" + str(work),
                   str(root / "driver_client.cpp"), str(library), "-o", str(output),
                   "-Wl,/Brepro"]
        subprocess.run(command, cwd=work, check=True)
    provenance = {
        "platform": "windows-x86_64", "runtime_linkage": "none (Driver-only client)",
        "host_compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
        "host_crt": "Microsoft static release CRT, Visual C++ tools " + os.environ.get("VCToolsVersion", "unidentified").strip(),
        "windows_sdk": os.environ.get("WindowsSDKVersion", "unidentified").rstrip("\\"),
        "header_version": "CUDA 12.4", "minimum_driver_api": 12000,
        "import_library": "NVIDIA CUDA 12.4 lib/x64/cuda.lib",
        "command": [arg.replace(str(toolkit), "<NVIDIA-toolkit>").replace(str(root), "<fixture>")
                    .replace(str(work), "<generated-images>") for arg in command],
        "gpu_images": "Embedded unmodified kernel_launch PTX and NVIDIA ptxas SM75 cubin; see ../kernel_launch/provenance.json",
        "sha256": {"driver_client.cpp": sha256(root / "driver_client.cpp"),
                   "produce.py": sha256(root / "produce.py"),
                   "windows_x64/driver_client.exe": sha256(output),
                   "../kernel_launch/kernel_launch.ptx": sha256(inputs / "kernel_launch.ptx"),
                   "../kernel_launch/kernel_launch.cubin": sha256(inputs / "kernel_launch.cubin")},
        "producer_sha256": {"include/cuda.h": sha256(header), "lib/x64/cuda.lib": sha256(library)}
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
