#include "driver.hpp"
#include "../core/frontend_api.hpp"
#include "../core/object_link_api.hpp"
#include "../core/program_api.hpp"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace cudavm {
using nvcc_path = std::filesystem::path;
struct nvcc_host_error { int status; };
std::string nvcc_read(const nvcc_path& path) {
    std::ifstream stream(path, std::ios::binary); if (!stream) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}
void nvcc_write(const nvcc_path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc); if (!stream) throw std::runtime_error("cannot write " + path.string());
    stream.write(text.data(), std::streamsize(text.size())); stream.close(); if (!stream) throw std::runtime_error("write failed: " + path.string());
}
void nvcc_check(int status, compiler_result* output) {
    compiler_result_ptr owned(output);
    if (const auto* text = cudavm_core_result_diagnostics_v1(output)) std::fputs(text, stderr);
    if (status) throw std::runtime_error("CUDA compilation failed");
}
std::vector<const char*> nvcc_pointers(const std::vector<std::string>& values) {
    std::vector<const char*> result; for (const auto& value : values) result.push_back(value.c_str()); return result;
}
struct nvcc_work_directory {
    nvcc_path path;
    bool keep;
    nvcc_work_directory(const nvcc_path& parent, bool preserve) : keep(preserve) {
        static std::atomic<unsigned> sequence{0};
        auto root = std::filesystem::absolute(parent.empty() ? nvcc_path(".") : parent).lexically_normal();
        if (!std::filesystem::is_directory(root)) throw std::runtime_error("output directory is unavailable: " + root.string());
        for (unsigned attempt = 0; attempt < 100; ++attempt) {
            path = root / (".nvcc-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(++sequence));
            if (std::filesystem::create_directory(path)) return;
        }
        throw std::runtime_error("cannot create compiler work directory");
    }
    ~nvcc_work_directory() {
        if (keep) return;
        // This exact child was successfully created by this invocation.
        std::error_code error; std::filesystem::remove_all(path, error);
    }
};
void nvcc_publish(const nvcc_path& source, const nvcc_path& destination) {
#if defined(_WIN32)
    if (!MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot publish output: " + destination.string());
#else
    std::filesystem::rename(source, destination);
#endif
}
std::string nvcc_object_record(const std::string& payload, bool completed) {
    std::string bytes(completed ? "CVLNK001" : "CVOBJ001");
    for (unsigned i = 0; i < 8; ++i) bytes += char(uint64_t(payload.size()) >> (8 * i));
    bytes += payload;
    std::string source = "\n#if defined(_MSC_VER)\n#pragma section(\".cudavm\",read)\n__declspec(allocate(\".cudavm\"))\n#endif\n"
        "__attribute__((used,section(\".cudavm\"),aligned(1))) static const unsigned char __cudavm_object_payload[] =\n\"";
    unsigned column = 0;
    for (unsigned char byte : bytes) {
        source += '\\'; source += char('0' + (byte >> 6)); source += char('0' + ((byte >> 3) & 7)); source += char('0' + (byte & 7));
        if (++column == 256) { source += "\"\n\""; column = 0; }
    }
    return source + "\";\n";
}
int nvcc_build_device_runtime(const char* output) {
    try {
        compiler_result* result{}; auto status = cudavm_core_device_runtime_v1(&result);
        std::string bytes;
        if (!status) bytes.assign(cudavm_core_result_output_v1(result), cudavm_core_result_size_v1(result));
        nvcc_check(status, result); nvcc_write(output, nvcc_object_record(bytes, false)); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "device Runtime: %s\n", error.what()); return 1; }
}
struct nvcc_driver {
    nvcc_options options;
    nvcc_path root, compiler;
    bool cl = false;
    std::vector<std::string> definitions;
    nvcc_driver(nvcc_options input) : options(std::move(input)) {
#if !defined(_WIN64) || !(defined(_M_X64) || defined(__x86_64__))
        throw std::runtime_error("the public compiler profile requires Windows x64");
#endif
        root = nvcc_executable_path().parent_path().parent_path();
        if (!std::filesystem::is_regular_file(root / "include/cuda_runtime.h")
            || !std::filesystem::is_regular_file(root / "include/cuda.h"))
            throw std::runtime_error("public nvcc requires its coherent installed SDK beside bin; install the SDK first");
        auto selected = options.host_compiler;
        if (selected.empty()) selected =
#if defined(_WIN32)
            "clang-cl.exe";
#else
            "clang++";
#endif
        if (std::filesystem::is_directory(selected)) selected = (nvcc_path(selected) /
#if defined(_WIN32)
            "clang-cl.exe"
#else
            "clang++"
#endif
            ).string();
        compiler = nvcc_find_program(selected); auto name = compiler.filename().string();
        if (!name.starts_with("clang-cl")) throw std::runtime_error("this public compiler profile requires clang-cl as its host compiler");
        cl = name.starts_with("clang-cl");
        definitions = options.defines;
        definitions.emplace_back("CUDAVM_STATIC=1");
        for (const char* definition : {"__NVCC__=1", "__CUDACC_VER_MAJOR__=13", "__CUDACC_VER_MINOR__=4", "__CUDACC_VER_BUILD__=1"}) definitions.emplace_back(definition);
        if (options.relocatable) definitions.emplace_back("__CUDACC_RDC__=1");
#if defined(_WIN32)
        definitions.emplace_back("_MT=1");
        if (options.host_runtime.starts_with("dll")) definitions.emplace_back("_DLL=1");
        if (options.host_runtime.ends_with("_dbg")) definitions.emplace_back("_DEBUG=1");
#endif
    }
    std::vector<std::string> runtime_libraries() const {
        if (options.runtime == "none") return {};
        std::vector<std::string> result{options.runtime == "shared" ? "cudart.lib" : "cudart_static.lib", "cuda.lib"};
        for (const auto& library : result) if (!std::filesystem::is_regular_file(root / "lib/x64" / library)) throw std::runtime_error("selected SDK library is unavailable: " + library);
        return result;
    }
    void discovery() const {
        if (!options.verbose) return;
        std::string libraries;
        for (const auto& path : runtime_libraries()) { if (!libraries.empty()) libraries += ' '; libraries += nvcc_quote_argument(path); }
        std::fprintf(stderr, "#$ TOP=%s\n#$ NVVMIR_LIBRARY_DIR=%s\n#$ INCLUDES=-I%s\n#$ LIBRARIES=%s\n#$ -arch compute_%s\n",
            root.generic_string().c_str(), (root / "nvvm/libdevice").generic_string().c_str(),
            nvcc_quote_argument((root / "include").generic_string()).c_str(), libraries.c_str(), options.architecture.substr(3).c_str());
    }
    void execute(const std::vector<std::string>& command) const {
        auto status = nvcc_execute(command, options.verbose); if (status) throw nvcc_host_error{status};
    }
    std::vector<std::string> host_arguments() const {
        std::vector<std::string> args{compiler.generic_string()};
        if (cl) {
            args.insert(args.end(), {"/nologo", "/EHsc", "-Werror=unknown-argument", "/std:" + options.standard,
                "/O" + std::to_string(options.optimization == 3 ? 2 : options.optimization)});
            if (!options.optimization) args.back() = "/Od";
            args.push_back(options.host_runtime == "dll" ? "/MD" : options.host_runtime == "dll_dbg" ? "/MDd" : options.host_runtime == "static_dbg" ? "/MTd" : "/MT");
            if (options.debug) args.push_back("/Z7");
        } else {
            args.insert(args.end(), {"-std=" + options.standard, "-O" + std::to_string(options.optimization)});
#if defined(_WIN32)
            args.push_back("-fms-runtime-lib=" + options.host_runtime);
#endif
            if (options.debug) args.push_back("-g");
        }
        args.push_back((cl ? "/I" : "-I") + (root / "include").generic_string());
        for (const auto& value : options.includes) args.push_back((cl ? "/I" : "-I") + value);
        for (const auto& value : definitions) args.push_back((cl ? "/D" : "-D") + value);
        for (const auto& value : options.undefines) args.push_back((cl ? "/U" : "-U") + value);
        for (const auto& flag : options.host_flags) args.push_back(flag);
        return args;
    }
    void host_compile(const nvcc_path& input, const nvcc_path& output) const {
        auto args = host_arguments();
        args.insert(args.end(), {cl ? "/c" : "-c", input.generic_string()});
        if (cl) args.push_back("/Fo" + output.generic_string()); else args.insert(args.end(), {"-o", output.generic_string()});
        execute(args);
    }
    nvcc_path resolve_input(const std::string& value) const {
        nvcc_path path(value);
        if (path.extension() == ".lib" && !path.has_parent_path()) {
            auto name = path.filename().string();
            auto selected = root / "lib/x64" / path;
            if (name == "cuda.lib" || name == "cudart.lib" || name == "cudart_static.lib" || name == "cudadevrt.lib") {
                if (!std::filesystem::is_regular_file(selected)) throw std::runtime_error("selected SDK library is unavailable: " + name);
                return selected;
            }
            if (std::filesystem::is_regular_file(path)) return std::filesystem::absolute(path);
            for (const auto& folder : options.library_paths) if (std::filesystem::is_regular_file(nvcc_path(folder) / path)) return std::filesystem::absolute(nvcc_path(folder) / path);
            if (std::filesystem::is_regular_file(selected)) return selected;
            std::string folders = std::getenv("LIB") ? std::getenv("LIB") : "";
            size_t begin = 0;
            for (;;) {
                auto end = folders.find(';', begin); auto candidate = nvcc_path(folders.substr(begin, end - begin)) / path;
                if (std::filesystem::is_regular_file(candidate)) return std::filesystem::absolute(candidate);
                if (end == std::string::npos) break; begin = end + 1;
            }
        }
        return std::filesystem::absolute(path);
    }
    void frontend(const nvcc_path& input, const nvcc_path& folder, const nvcc_path& depfile, const nvcc_path& target, bool preprocess = false) const {
        auto includes = nvcc_pointers(options.includes), defines = nvcc_pointers(definitions), undefines = nvcc_pointers(options.undefines);
        auto source = std::filesystem::absolute(input).generic_string(), host = (folder / "host.cpp").generic_string(), bitcode = (folder / "device.bc").generic_string();
        auto sdk = root.generic_string(), dep = depfile.generic_string(), object = target.generic_string();
        compiler_translation_unit_request_v2 request{{2, source.c_str(), host.c_str(), bitcode.c_str(), nullptr, dep.empty() ? nullptr : dep.c_str(),
            options.standard.c_str(), options.architecture.c_str(), sdk.c_str(), includes.size(), includes.data(), defines.size(), defines.data()},
            options.optimization, object.c_str(), undefines.size(), undefines.data()};
        compiler_result* result{};
        auto status = preprocess ? cudavm_core_preprocess_translation_unit_v2(&request, &result) : cudavm_core_compile_translation_unit_v2(&request, &result);
        if (preprocess && !status) nvcc_write(folder / "preprocessed.ii", compiler_client_text(cudavm_core_result_output_v1(result)));
        nvcc_check(status, result);
    }
    bool device_link(const std::vector<std::string>& inputs, const nvcc_path& module, bool skip_completed) const {
        auto combined = inputs;
        if (options.device_runtime) {
            auto library = (root / "lib/x64/cudadevrt.lib").generic_string();
            if (!std::filesystem::is_regular_file(library)) throw std::runtime_error("selected SDK device Runtime is unavailable");
            if (std::find(combined.begin(), combined.end(), library) == combined.end()) combined.push_back(library);
        }
        auto pointers = nvcc_pointers(combined); auto path = module.generic_string();
        compiler_object_link_request request{1, pointers.size(), pointers.data(), path.c_str(), options.architecture.c_str(), options.optimization, unsigned(skip_completed)};
        compiler_result* result{}; auto status = cudavm_core_link_objects_v1(&request, &result); nvcc_check(status, result);
        return std::filesystem::is_regular_file(module);
    }
    void cuda_compile(const nvcc_path& input, const nvcc_path& output, const nvcc_path& folder, const nvcc_path& dependency, const nvcc_path& consumer_output) const {
        frontend(input, folder, dependency, options.dependency_target.empty() ? consumer_output : nvcc_path(options.dependency_target));
        auto host = folder / "host.cpp"; auto text = nvcc_read(host);
        nvcc_write(host, text + nvcc_object_record(nvcc_read(folder / "device.bc"), false));
        if (options.relocatable || options.phase == nvcc_phase::device_compile) { host_compile(host, output); return; }
        auto intermediate = folder / "unlinked.obj"; host_compile(host, intermediate);
        auto module = folder / "module.cpp";
        if (!device_link({intermediate.generic_string()}, module, false)) throw std::runtime_error("CUDA object lost its device payload");
        nvcc_write(host, text + nvcc_read(module)); host_compile(host, output);
    }
    int run() {
        discovery();
        if (options.inputs.empty()) throw std::runtime_error("no input files");
        if (options.inputs.size() > 1 && !options.output.empty() && options.phase != nvcc_phase::link && options.phase != nvcc_phase::archive && options.phase != nvcc_phase::device_link)
            throw std::runtime_error("one output file requires one input in this phase");
        if (options.inputs.size() > 1 && (options.phase == nvcc_phase::ptx || options.phase == nvcc_phase::image))
            throw std::runtime_error("this GPU output profile requires one source input");
        nvcc_path output = options.output;
        if (output.empty()) {
            if (options.phase == nvcc_phase::link) output = "a.exe";
            else if (options.phase == nvcc_phase::device_link) output = "a_dlink.obj";
            else if (options.phase == nvcc_phase::archive) output = "a.lib";
            else { output = nvcc_path(options.inputs.front()).filename(); output.replace_extension(options.phase == nvcc_phase::ptx ? ".ptx" : options.phase == nvcc_phase::image ? ".cubin" : ".obj"); }
        }
        output = std::filesystem::absolute(output);
        // Keep temporary output on the destination filesystem for atomic publication.
        auto work_parent = options.keep_directory.empty() ? output.parent_path() : std::filesystem::absolute(options.keep_directory);
        if (!options.keep_directory.empty()) std::filesystem::create_directories(work_parent);
        nvcc_work_directory work(work_parent, options.keep);
        std::vector<std::string> objects;
        unsigned index = 0;
        for (const auto& path : options.inputs) {
            nvcc_path input = resolve_input(path);
            if (!std::filesystem::is_regular_file(input)) throw std::runtime_error("input file is unavailable: " + input.string());
            auto extension = input.extension().string();
            bool cuda = extension == ".cu" || options.cuda_input, source = cuda || extension == ".cpp" || extension == ".cc" || extension == ".cxx" || extension == ".c";
            if (!source) {
                if (options.phase != nvcc_phase::link && options.phase != nvcc_phase::archive && options.phase != nvcc_phase::device_link)
                    throw std::runtime_error("unsupported input for this compilation phase: " + input.string());
                objects.push_back(input.generic_string()); continue;
            }
            auto folder = work.path / std::to_string(index++); std::filesystem::create_directory(folder);
            auto object = folder / "output.obj";
            nvcc_path dependency;
            if (options.dep_compile || options.phase == nvcc_phase::dependencies) dependency = folder / "output.d";
            if (options.phase == nvcc_phase::preprocess || options.phase == nvcc_phase::dependencies) {
                if (!cuda) throw std::runtime_error("this preprocessing profile requires CUDA source");
                frontend(input, folder, dependency, options.dependency_target.empty() ? output : nvcc_path(options.dependency_target), true);
                auto text = nvcc_read(options.phase == nvcc_phase::preprocess ? folder / "preprocessed.ii" : dependency);
                if (options.output.empty()) std::fwrite(text.data(), 1, text.size(), stdout);
                else { nvcc_write(work.path / "output", text); nvcc_publish(work.path / "output", output); }
                continue;
            }
            if (options.phase == nvcc_phase::ptx || options.phase == nvcc_phase::image) {
                if (!cuda) throw std::runtime_error("GPU output requires CUDA source");
                frontend(input, folder, {}, output);
                auto bytes = nvcc_read(folder / "device.bc"); compiler_result* result{}; compiler_program* raw{};
                compiler_input_request request{1, compiler_input_kind::cudavm_ir, bytes.data(), bytes.size(), nullptr, nullptr};
                auto status = cudavm_core_program_import_v1(&request, &raw, &result); nvcc_check(int(status), result); compiler_program_ptr program(raw);
                compiler_program* prepared{}; compiler_pipeline_request pipeline{1, 0, 1, options.optimization};
                status = cudavm_core_program_prepare_v1(program.get(), &pipeline, &prepared, &result); nvcc_check(int(status), result); compiler_program_ptr ready(prepared);
                compiler_emit_request emit; emit.kind = options.phase == nvcc_phase::ptx ? compiler_output_kind::ptx : compiler_output_kind::program_image;
                emit.environment = options.phase == nvcc_phase::ptx ? compiler_environment::cuda : compiler_environment::none;
                status = cudavm_core_program_emit_v1(ready.get(), &emit, &result);
                if (!int(status)) nvcc_write(work.path / "output", std::string(cudavm_core_result_output_v1(result), cudavm_core_result_size_v1(result)));
                nvcc_check(int(status), result); nvcc_publish(work.path / "output", output); continue;
            }
            auto consumer_output = output;
            if (options.inputs.size() > 1 && (options.phase == nvcc_phase::compile || options.phase == nvcc_phase::device_compile))
                consumer_output = std::filesystem::current_path() / input.filename().replace_extension(".obj");
            if (cuda) cuda_compile(input, object, folder, dependency, consumer_output); else host_compile(input, object);
            if (!dependency.empty()) {
                nvcc_path destination = options.depfile.empty() ? nvcc_path(consumer_output).replace_extension(".d") : nvcc_path(options.depfile);
                nvcc_write(destination, nvcc_read(dependency));
            }
            if (options.phase == nvcc_phase::compile || options.phase == nvcc_phase::device_compile) {
                auto destination = output;
                if (options.inputs.size() > 1) destination = std::filesystem::current_path() / input.filename().replace_extension(".obj");
                nvcc_publish(object, destination);
            } else objects.push_back(object.generic_string());
        }
        if (options.phase == nvcc_phase::compile || options.phase == nvcc_phase::device_compile || options.phase == nvcc_phase::preprocess
            || options.phase == nvcc_phase::dependencies || options.phase == nvcc_phase::ptx || options.phase == nvcc_phase::image) return 0;
        if (options.phase == nvcc_phase::archive) {
            auto tool = nvcc_find_program((compiler.parent_path() / "llvm-lib.exe").string());
            auto temporary = work.path / "output.lib";
            std::vector<std::string> command{tool.generic_string(), "/NOLOGO", "/OUT:" + temporary.generic_string()};
            command.insert(command.end(), objects.begin(), objects.end()); execute(command); nvcc_publish(temporary, output); return 0;
        }
        auto module = work.path / "module.cpp";
        auto link_inputs = objects;
        for (const auto& library : options.libraries) {
            auto path = resolve_input(library.ends_with(".lib") ? library : library + ".lib");
            if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("link library is unavailable: " + library);
            link_inputs.push_back(path.generic_string());
        }
        bool linked = device_link(link_inputs, module, options.phase == nvcc_phase::link);
        if (linked || options.phase == nvcc_phase::device_link) {
            auto text = linked ? nvcc_read(module) : std::string("// No relocatable device inputs.\n");
            nvcc_write(module, text + nvcc_object_record({}, true));
            auto object = work.path / "dlink.obj"; host_compile(module, object);
            if (options.phase == nvcc_phase::device_link) { nvcc_publish(object, output); return 0; }
            objects.push_back(object.generic_string());
        }
        auto temporary = work.path / "output.exe";
        auto command = host_arguments(); command.insert(command.end(), objects.begin(), objects.end());
        auto libraries = runtime_libraries(); command.insert(command.end(), libraries.begin(), libraries.end());
        if (cl) {
            command.push_back("/Fe" + temporary.generic_string()); if (options.shared) command.push_back("/LD"); command.push_back("/link");
            command.push_back("/LIBPATH:" + (root / "lib/x64").generic_string());
            for (const auto& path : options.library_paths) command.push_back("/LIBPATH:" + path);
            for (const auto& name : options.libraries) command.push_back(name.ends_with(".lib") ? name : name + ".lib");
            for (const auto& flag : options.link_flags) command.push_back(flag.starts_with("-Wl,") ? flag.substr(4) : flag);
        } else {
            command.insert(command.end(), {"-o", temporary.generic_string()}); if (options.shared) command.push_back("-shared");
            command.push_back("-L" + (root / "lib/x64").generic_string());
            for (const auto& path : options.library_paths) command.push_back("-L" + path);
            for (const auto& name : options.libraries) command.push_back("-l" + name);
            command.insert(command.end(), options.link_flags.begin(), options.link_flags.end());
        }
        execute(command); nvcc_publish(temporary, output); return 0;
    }
};
int run_nvcc_driver(int argc, char** argv) {
    try {
        auto options = parse_nvcc_options(argc, argv);
        if (options.version) { std::puts("nvcc: cudavm Cuda compiler driver\nCuda compilation tools, release 13.4, V13.4.1\ncudavm implementation 0.1.0; CUDA compiler compatibility profile 13.4.1"); return 0; }
        if (options.help) { std::puts("usage: nvcc [options] input...\nWindows x64 Clang host profile: -c, -dc, -dlink, -lib, -E, -M/-MM, -MMD, -ptx, -cubin/-fatbin\n-std=c++17|c++20|c++23 -arch=sm_75 -Ipath -Dname -Uname -o output -ccbin clang-cl.exe\n--cudart=static|shared|none --cudadevrt=none -Xcompiler flags -Xlinker flags @options.rsp\nGPU binary outputs use self-contained cudavm program images."); return 0; }
        return nvcc_driver(std::move(options)).run();
    } catch (const nvcc_host_error& error) { return error.status; }
    catch (const std::exception& error) { std::fprintf(stderr, "nvcc: %s\n", error.what()); return 1; }
}
}
