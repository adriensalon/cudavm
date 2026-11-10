"""Exercise only the installed public SDK and ordinary host/build tools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class ProfileUnavailable(Exception):
    pass


def host_environment():
    env = os.environ.copy()
    if not env.get("INCLUDE") or not env.get("LIB"):
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        require(vswhere.is_file(), "an x64 Visual Studio C++ environment is required")
        installation = subprocess.check_output([str(vswhere), "-latest", "-property", "installationPath"], text=True).strip()
        require(installation, "Visual Studio C++ toolchain is available")
        batch = Path(installation) / "Common7/Tools/VsDevCmd.bat"
        command = f'"{env["COMSPEC"]}" /d /s /c ""{batch}" -arch=x64 -host_arch=x64 >nul && set"'
        output = subprocess.check_output(command, text=True)
        for line in output.splitlines():
            if "=" in line and not line.startswith("="):
                key, value = line.split("=", 1)
                env[key.upper()] = value
    for key in list(env):
        if key.upper().startswith("CUDA_PATH") or key.upper() in {"CUDA_HOME", "CUDACXX", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CUDAVM_CUDA_DRIVER", "CUDAVM_DEVICES"}:
            env.pop(key)
    env["CUDAVM_BACKENDS"] = ""
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["build", "config", "work", "mode", "cmake", "ninja", "host"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parent
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    root = work / (args.mode + "_" + uuid.uuid4().hex)
    root.mkdir()
    require(root.parent == work and root != work, "test deployment stays under its owned work directory")
    env = host_environment()
    log = root / "commands.log"

    def run(command, *, expected=0, cwd=None):
        command = list(map(str, command))
        result = subprocess.run(command, cwd=cwd or root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(subprocess.list2cmdline(command) + "\n" + result.stdout + "\n")
        if result.returncode == 77 and env.get("CUDAVM_BACKENDS") == "vulkan":
            raise ProfileUnavailable("selected Vulkan backend has no available device")
        require(result.returncode == expected if expected is not None else result.returncode != 0,
                f"command returned {result.returncode}: {subprocess.list2cmdline(command)}\n{result.stdout[-12000:]}\nEvidence: {root}")
        return result.stdout

    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"profile": "windows-x64-clang-v1", "header_baseline": "13.4.1",
                                 "host_crt_tools": env.get("VCTOOLSVERSION"),
                                 "windows_sdk": env.get("WINDOWSSDKVERSION")}, indent=2) + "\n")
    for tool in [args.cmake, args.ninja, args.host]:
        run([tool, "--version"])

    original = root / "original SDK"
    run([args.cmake, "--install", args.build, "--config", args.config, "--prefix", original, "--component", "cuda_sdk"])
    sdk = root / "relocated SDK"
    original.rename(sdk)
    require(not original.exists(), "original SDK is unavailable after relocation")
    for name in ["bin/nvcc.exe", "bin/nvcuda.dll", "bin/cudart64_13.dll", "bin/cudavm_core.dll", "lib/x64/cuda.lib", "lib/x64/cudart.lib", "lib/x64/cudart_static.lib", "lib/x64/cudadevrt.lib", "include/cuda_runtime.h", "include/cuda.h", "include/crt/host_defines.h", "version.json"]:
        require((sdk / name).is_file(), "installed SDK component exists: " + name)
    for source, installed in [("cudart/cuda_runtime.h", "cuda_runtime.h"), ("cuda/cuda.h", "cuda.h"), ("nvcc/crt/host_defines.h", "crt/host_defines.h")]:
        require((sdk / "include" / source).read_bytes() == (sdk / "include" / installed).read_bytes(), "flat public headers retain pinned module bytes")
    # Preserve host compiler, linker, Windows SDK and OS paths; expose neither
    # NVIDIA compiler tools nor the candidate source/build executable paths.
    tools = [str(sdk / "bin"), str(Path(args.host).parent), str(Path(args.cmake).parent), str(Path(args.ninja).parent)]
    for path in env["PATH"].split(os.pathsep):
        lower = path.lower()
        if path and ("microsoft visual studio" in lower or "windows kits" in lower or lower.startswith(env["SYSTEMROOT"].lower())):
            tools.append(path)
    env["PATH"] = os.pathsep.join(dict.fromkeys(tools))
    nvcc = sdk / "bin/nvcc.exe"
    version = run([nvcc, "--version"])
    require("release 13.4" in version and "cudavm" in version, "compiler compatibility and implementation identities are explicit")
    resources = json.loads(run([nvcc, "--print-resources"]))
    require(resources["origin"] == "installed-sdk" and Path(resources["public_include"]).resolve() == sdk / "include", "compiler resources belong to selected relocated SDK")
    source = root / "consumer sources"
    source.mkdir()
    for pattern in ["*.cu", "*.cpp", "*.hpp"]:
        for file in fixture.glob(pattern):
            shutil.copyfile(file, source / file.name)
    launch = source / "launch.cu"

    def gpu(executable):
        env["CUDAVM_BACKENDS"] = "vulkan"
        unavailable = root / "unavailable consumer sources"
        require(source.parent == root and unavailable.parent == root, "only owned fixture inputs move during execution")
        source.rename(unavailable)
        try:
            run([executable])
        finally:
            unavailable.rename(source)
            env["CUDAVM_BACKENDS"] = ""

    if args.mode == "discovery":
        header = sdk / "include/cudart/cuda_runtime.h"
        saved = header.read_bytes()
        header.unlink()
        try:
            error = run([nvcc, "-c", launch, "-o", root / "missing.obj"], expected=None)
            require("installed SDK public header is unavailable" in error, "missing selected resources cannot fall back to source or another toolkit")
        finally:
            header.write_bytes(saved)
    elif args.mode == "protocol":
        obj = root / "compiled object.obj"
        depfile = root / "dependencies.d"
        response = root / "compile options.rsp"
        response.write_text(subprocess.list2cmdline(["-c", "-O1", "-std=c++17", "-DFIXTURE_FACTOR=5", "-I" + str(source), "-MMD", "-MF", str(depfile), str(launch), "-o", str(obj)]), encoding="utf-8-sig")
        run([nvcc, "@" + str(response)])
        require(obj.read_bytes()[:2] == b"\x64\x86", "compiler produces ordinary x64 COFF")
        dependencies = depfile.read_text()
        require("parameters.hpp" in dependencies and "compiled" in dependencies and ".nvcc-" not in dependencies, "depfile tracks real headers and public output")
        executable = sdk / "bin/cpu.exe"
        run([nvcc, obj, "-o", executable])
        run([executable, "cpu"])
        preprocessed = run([nvcc, "-E", "-DFIXTURE_FACTOR=11", "-UFIXTURE_FACTOR", "-DFIXTURE_FACTOR=5", launch])
        require("fixture_factor = 5" in preprocessed, "preprocessing applies command-line definitions")
        dependency_only = run([nvcc, "-MM", launch])
        require("parameters.hpp" in dependency_only, "dependency-only phase follows user headers")
        before = hashlib.sha256(obj.read_bytes()).digest()
        broken = source / "broken.cu"
        broken.write_text("__global__ void broken( {\n", encoding="utf-8")
        run([nvcc, "-c", broken, "-o", obj], expected=None)
        require(hashlib.sha256(obj.read_bytes()).digest() == before, "failed compilation preserves previous valid output")
        run([nvcc, "--not-a-supported-option", "-c", launch, "-o", obj], expected=None)
        run([nvcc, "--forward-unknown-to-host-compiler", "--not-a-host-option", "-c", launch, "-o", obj], expected=None)
        run([nvcc, "-c", launch, "-o", obj])
        for phase, output, marker in [("-ptx", root / "program.ptx", b".version"), ("-cubin", root / "program.cubin", b"CUDAVM-PROGRAM-IMAGE\n")]:
            run([nvcc, phase, launch, "-o", output])
            require(marker in output.read_bytes(), "public GPU output representation")
    elif args.mode == "execution":
        for linkage in ["static", "shared"]:
            executable = sdk / ("bin/" + linkage + ".exe")
            run([nvcc, "-std=c++17", "-O2", "-DFIXTURE_FACTOR=11", "-UFIXTURE_FACTOR", "-DFIXTURE_FACTOR=5", "--cudart=" + linkage, launch, "-o", executable])
            run([executable, "cpu"])
            gpu(executable)
        executable = sdk / "bin/device_version.exe"
        run([nvcc, "-rdc=true", source / "device_version.cu", "-o", executable])
        gpu(executable)
    elif args.mode == "archives":
        objects = []
        for unit in ["kernel", "helper", "dead"]:
            obj = root / (unit + ".obj")
            run([nvcc, "-dc", "-O1", source / (unit + ".cu"), "-o", obj])
            objects.append(obj)
        main_obj = root / "main.obj"
        run([nvcc, "-c", source / "main.cpp", "-o", main_obj])
        archive = root / "device.lib"
        run([nvcc, "-lib", *objects, "-o", archive])
        executable = sdk / "bin/implicit.exe"
        run([nvcc, main_obj, archive, "-o", executable])
        gpu(executable)
        dlink = root / "dlink.obj"
        run([nvcc, "-dlink", main_obj, archive, "-o", dlink])
        executable = sdk / "bin/ordinary_link.exe"
        run([args.host, "/nologo", "/MT", main_obj, archive, dlink, "/Fe" + str(executable), "/link", "/LIBPATH:" + str(sdk / "lib/x64"), "cudart_static.lib", "cuda.lib"])
        gpu(executable)
        before = dlink.read_bytes()
        run([nvcc, "-dlink", objects[0], "-o", dlink], expected=None)
        require(dlink.read_bytes() == before, "failed device link preserves completed output")
    elif args.mode in {"cmake", "cmake_rdc"}:
        shutil.copyfile(fixture.parent / "cuda_build_integration/CMakeLists.txt", source / "CMakeLists.txt")
        build = root / "consumer build"
        run([args.cmake, "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + args.ninja,
             "-DCMAKE_CXX_COMPILER=" + args.host, "-DCMAKE_CUDA_COMPILER=" + str(nvcc), "-DCMAKE_CUDA_HOST_COMPILER=" + args.host,
             "-DCUDAToolkit_ROOT=" + str(sdk), "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_CUDA_STANDARD=17",
             "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + str(sdk / "bin"), "-DFIXTURE_RDC=" + ("ON" if args.mode == "cmake_rdc" else "OFF")])
        run([args.cmake, "--build", build, "--parallel", "2"])
        gpu(sdk / "bin/cuda_stock.exe")
        run([args.cmake, "--build", build, "--parallel", "2"])
        # A real header dependency must rebuild the CUDA object and executable.
        if args.mode == "cmake":
            header = source / "parameters.hpp"
            header.write_text(header.read_text() + "\nstatic_assert(fixture_bias == 7);\n", encoding="utf-8")
            output = run([args.cmake, "--build", build, "--parallel", "2"])
            require("Building CUDA object" in output, "stock incremental build follows CUDA user-header dependencies")
            gpu(sdk / "bin/cuda_stock.exe")
    else:
        raise RuntimeError("unknown profile mode")
    shutil.copyfile(log, work / (args.mode + ".log"))
    shutil.rmtree(root)
    print("PASS: installed public SDK " + args.mode)


if __name__ == "__main__":
    try:
        main()
    except ProfileUnavailable as error:
        print("SKIP: " + str(error))
        sys.exit(77)
    except Exception as error:
        print(error, file=sys.stderr)
        sys.exit(1)
