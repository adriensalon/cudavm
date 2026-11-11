#define NVJITLINK_NO_INLINE
#include "test_framework.hpp"
#include <nvJitLink.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace cudavm {
#define JIT_LIFECYCLE_CALL(name, ...) (versioned ? __##name##_13_4(__VA_ARGS__) : name(__VA_ARGS__))
template<class T> struct jit_lifecycle_guarded {
    alignas(T) std::array<unsigned char, sizeof(T) + 32> bytes;
    jit_lifecycle_guarded() { bytes.fill(0xe7); }
    T* data() { return reinterpret_cast<T*>(bytes.data() + 16); }
    T get() const {
        for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= sizeof(T) + 16) require(bytes[i] == 0xe7, "nvJitLink scalar output byte guards");
        T result{}; std::memcpy(&result, bytes.data() + 16, sizeof(T)); return result;
    }
};
template<class Size, class Get> std::vector<unsigned char> jit_lifecycle_bytes(Size size_query, Get get, bool text) {
    jit_lifecycle_guarded<size_t> size, repeated;
    require(size_query(size.data()) == NVJITLINK_SUCCESS && size_query(repeated.data()) == NVJITLINK_SUCCESS, "nvJitLink result size queries");
    const auto count = size.get(); require(repeated.get() == count, "nvJitLink result size stable"); const auto written = text && count == 0 ? size_t{1} : count;
    std::vector<unsigned char> first(written + 64, 0x27), second(written + 64, 0x53);
    require(get(first.data() + 32) == NVJITLINK_SUCCESS && get(second.data() + 32) == NVJITLINK_SUCCESS, "nvJitLink result retrieval");
    for (size_t i = 0; i < first.size(); ++i) {
        if (i < 32 || i >= written + 32) require(first[i] == 0x27 && second[i] == 0x53, "nvJitLink exact result extent and guards");
        else require(first[i] == second[i], "nvJitLink writes every reported byte and repeated Get agrees");
    }
    std::vector<unsigned char> result(first.begin() + 32, first.end() - 32);
    if (text) require(result.back() == 0 && std::find(result.begin(), result.end() - 1, 0) == result.end() - 1, "nvJitLink exact text termination");
    return result;
}
struct jit_lifecycle_program {
    nvJitLinkHandle value{}; bool versioned; unsigned mode; std::string kernel;
    jit_lifecycle_program(bool pinned, unsigned output_mode = 0) : versioned(pinned), mode(output_mode) {
        std::vector<std::string> options{"-arch=sm_75", "-O0", "-verbose"}; if (mode) options.push_back("-lto"); if (mode == 1) options.push_back("-ptx");
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        jit_lifecycle_guarded<nvJitLinkHandle> output;
        require(JIT_LIFECYCLE_CALL(nvJitLinkCreate, output.data(), static_cast<uint32_t>(pointers.size()), pointers.data()) == NVJITLINK_SUCCESS, "nvJitLink create through selected ABI");
        value = output.get(); require(value != nullptr, "nvJitLink create publishes handle");
        for (auto& option : options) std::fill(option.begin(), option.end(), '!'); std::fill(pointers.begin(), pointers.end(), nullptr);
    }
    ~jit_lifecycle_program() { if (value) JIT_LIFECYCLE_CALL(nvJitLinkDestroy, &value); }
    jit_lifecycle_program(const jit_lifecycle_program&) = delete;
    jit_lifecycle_program& operator=(const jit_lifecycle_program&) = delete;
    void add(unsigned seed) {
        kernel = "jit_lifecycle_" + std::to_string(seed);
        std::string source = ".version 6.5\n.target sm_75\n.address_size 64\n.visible .entry " + kernel + "() { ret; }\n", name = kernel + ".ptx";
        require(JIT_LIFECYCLE_CALL(nvJitLinkAddData, value, NVJITLINK_INPUT_PTX, source.data(), source.size() + 1, name.c_str()) == NVJITLINK_SUCCESS, "nvJitLink eager input addition");
        std::fill(source.begin(), source.end(), '!'); std::fill(name.begin(), name.end(), '!');
    }
    void complete() const { require(JIT_LIFECYCLE_CALL(nvJitLinkComplete, value) == NVJITLINK_SUCCESS, "nvJitLink completes accepted input"); }
    nvJitLinkResult size(size_t* output) const {
        if (mode == 1) return JIT_LIFECYCLE_CALL(nvJitLinkGetLinkedPtxSize, value, output);
        if (mode == 2) return JIT_LIFECYCLE_CALL(nvJitLinkGetLinkedLTOIRSize, value, output);
        return JIT_LIFECYCLE_CALL(nvJitLinkGetLinkedCubinSize, value, output);
    }
    nvJitLinkResult get(void* output) const {
        if (mode == 1) return JIT_LIFECYCLE_CALL(nvJitLinkGetLinkedPtx, value, static_cast<char*>(output));
        if (mode == 2) return JIT_LIFECYCLE_CALL(nvJitLinkGetLinkedLTOIR, value, output);
        return JIT_LIFECYCLE_CALL(nvJitLinkGetLinkedCubin, value, output);
    }
    std::vector<unsigned char> result() const {
        auto bytes = jit_lifecycle_bytes([&](size_t* output) { return size(output); }, [&](void* output) { return get(output); }, mode == 1);
        require(!bytes.empty(), "nvJitLink selected output is nonempty"); if (mode == 1) require(std::string(reinterpret_cast<const char*>(bytes.data())).find(kernel) != std::string::npos, "linked PTX retains accepted kernel name"); return bytes;
    }
    std::vector<unsigned char> log(bool error) const {
        return jit_lifecycle_bytes([&](size_t* output) { return error ? JIT_LIFECYCLE_CALL(nvJitLinkGetErrorLogSize, value, output) : JIT_LIFECYCLE_CALL(nvJitLinkGetInfoLogSize, value, output); },
            [&](void* output) { return error ? JIT_LIFECYCLE_CALL(nvJitLinkGetErrorLog, value, static_cast<char*>(output)) : JIT_LIFECYCLE_CALL(nvJitLinkGetInfoLog, value, static_cast<char*>(output)); }, true);
    }
    void destroy() {
        jit_lifecycle_guarded<nvJitLinkHandle> output; std::memcpy(output.data(), &value, sizeof(value)); require(JIT_LIFECYCLE_CALL(nvJitLinkDestroy, output.data()) == NVJITLINK_SUCCESS, "nvJitLink destroy through selected ABI"); value = nullptr;
        require(output.get() == nullptr, "nvJitLink destroy clears only caller handle slot");
    }
};
void jit_lifecycle_discovery() {
    jit_lifecycle_guarded<unsigned> major, minor; require(nvJitLinkVersion(major.data(), minor.data()) == NVJITLINK_SUCCESS && major.get() == 13 && minor.get() == 4, "nvJitLink pinned version");
    for (bool versioned : {false, true}) { jit_lifecycle_program program(versioned); program.destroy(); }
}
void jit_lifecycle_ownership() {
    for (bool versioned : {false, true}) {
        jit_lifecycle_program program(versioned); program.add(versioned ? 79 : 17); program.versioned = !program.versioned; program.complete(); program.result(); program.destroy();
    }
}
void jit_lifecycle_outputs() {
    for (bool versioned : {false, true}) for (unsigned mode : {0u, 1u, 2u}) {
        jit_lifecycle_program program(versioned, mode); program.add(31 + mode); program.complete(); const auto retained = program.result(), before = retained;
        require(program.result() == retained, "nvJitLink repeated output retrieval"); program.log(false); program.log(true); program.destroy(); require(retained == before, "nvJitLink caller-owned output survives producer destruction");
    }
}
void jit_lifecycle_recovery() {
    for (bool versioned : {false, true}) {
        jit_lifecycle_program program(versioned); size_t size{}; require(program.size(&size) == NVJITLINK_ERROR_INTERNAL, "nvJitLink binary query requires Complete");
        require(JIT_LIFECYCLE_CALL(nvJitLinkAddData, program.value, NVJITLINK_INPUT_PTX, nullptr, size_t{1}, "null.ptx") == NVJITLINK_ERROR_NULL_INPUT && program.log(true).size() > 1, "nvJitLink rejected input and retrievable diagnostic");
        program.add(137); program.complete(); program.result(); program.destroy();
    }
}
void jit_lifecycle_validation() {
    for (bool versioned : {false, true}) {
        const char* options[]{"-arch=sm_75"}; nvJitLinkHandle rejected{};
        require(JIT_LIFECYCLE_CALL(nvJitLinkCreate, nullptr, 1, options) == NVJITLINK_ERROR_INVALID_INPUT && JIT_LIFECYCLE_CALL(nvJitLinkCreate, &rejected, 1, nullptr) == NVJITLINK_ERROR_INVALID_INPUT, "nvJitLink invalid create arguments");
        require(JIT_LIFECYCLE_CALL(nvJitLinkDestroy, nullptr) == NVJITLINK_ERROR_INVALID_INPUT && JIT_LIFECYCLE_CALL(nvJitLinkComplete, nullptr) == NVJITLINK_ERROR_INVALID_INPUT, "nvJitLink null handle arguments");
        jit_lifecycle_program program(versioned); const char input[] = "input";
        require(JIT_LIFECYCLE_CALL(nvJitLinkAddData, program.value, NVJITLINK_INPUT_PTX, input, size_t{0}, "empty.ptx") == NVJITLINK_ERROR_INVALID_INPUT, "nvJitLink zero input extent");
        require(program.size(nullptr) == NVJITLINK_ERROR_INVALID_INPUT, "nvJitLink null result-size output"); program.add(193); program.complete(); require(program.get(nullptr) == NVJITLINK_ERROR_INVALID_INPUT, "nvJitLink null completed-result buffer"); program.result(); program.destroy();
    }
}
void jit_lifecycle_threading() {
    std::array<std::thread, 4> workers; std::array<std::exception_ptr, 4> failures{}; const auto join = [&] { for (auto& worker : workers) if (worker.joinable()) worker.join(); };
    try { for (unsigned worker = 0; worker < workers.size(); ++worker) workers[worker] = std::thread([&, worker] {
        try { jit_lifecycle_program program((worker & 1) != 0); program.add(211 + worker * 19); program.complete(); program.result(); program.destroy(); }
        catch (...) { failures[worker] = std::current_exception(); }
    }); } catch (...) { join(); throw; }
    join(); for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
}
std::string jit_lifecycle_case = "all";
void execute_jit_lifecycle() {
    bool selected = false; const auto run = [&](const char* name, auto execute) { if (jit_lifecycle_case == "all" || jit_lifecycle_case == name) { execute(); selected = true; } };
    run("discovery", jit_lifecycle_discovery); run("ownership", jit_lifecycle_ownership); run("outputs", jit_lifecycle_outputs); run("recovery", jit_lifecycle_recovery); run("validation", jit_lifecycle_validation); run("threading", jit_lifecycle_threading);
    require(selected, "unknown nvjitlink_lifecycle case: " + jit_lifecycle_case);
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::jit_lifecycle_case = argv[1];
    return cudavm::run_host_test(argc, argv, cudavm::execute_jit_lifecycle);
}
