"""Intentionally regenerate the genuine CUDA 12.4 static Runtime client."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    toolkit = args.toolkit.resolve()
    header = toolkit / "include/cuda.h"
    nvcc = toolkit / "bin/nvcc.exe"
    if not re.search(r"^#define CUDA_VERSION\s+12040\s*$", header.read_text(), re.MULTILINE):
        parser.error("this producer profile requires genuine NVIDIA CUDA 12.4")
    if not os.environ.get("VCToolsVersion") or not os.environ.get("WindowsSDKVersion"):
        parser.error("run in a Windows x64 Visual Studio C++ toolchain environment")
    output = root / "windows_x64/runtime_client.exe"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cudavm-static-cudart-") as folder:
        command = [str(nvcc), "--cudart=static", "--cudadevrt=none", "--use-local-env",
                   "--allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                   "-std=c++17", "-O3", "-gencode=arch=compute_75,code=sm_75",
                   "-Xcompiler", "/MT", "-Xlinker", "/MANIFEST:NO", "--keep", "--keep-dir", folder,
                   str(root / "runtime_client.cu"), "-o", str(output)]
        subprocess.run(command, cwd=folder, check=True)
        images = {path.name: sha256(path) for path in Path(folder).glob("*.cubin")}
        if not images:
            raise RuntimeError("producer did not retain its original NVIDIA cubin")
    for suffix in [".lib", ".exp"]:
        output.with_suffix(suffix).unlink(missing_ok=True)
    provenance = {
        "platform": "windows-x86_64", "runtime_linkage": "genuine NVIDIA CUDA 12.4 cudart_static.lib",
        "runtime_version": 12040, "device_runtime": "none", "default_stream": "legacy",
        "application_manifest": "none; application-local DLL redirection marker supplied at deployment",
        "host_crt": "Microsoft static release CRT, Visual C++ tools " + os.environ["VCToolsVersion"].strip(),
        "windows_sdk": os.environ["WindowsSDKVersion"].rstrip("\\"),
        "nvcc": subprocess.check_output([str(nvcc), "--version"], text=True).strip(),
        "command": [arg.replace(str(toolkit), "<NVIDIA-toolkit>").replace(str(root), "<fixture>")
                    .replace(folder, "<producer-work>") for arg in command],
        "original_cubin_sha256": images,
        "sha256": {name: sha256(root / name) for name in
                   ["runtime_client.cu", "produce.py", "windows_x64/runtime_client.exe"]},
        "producer_sha256": {name: sha256(toolkit / name) for name in
                            ["bin/nvcc.exe", "include/cuda.h", "include/cuda_runtime.h", "lib/x64/cudart_static.lib"]}}
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
