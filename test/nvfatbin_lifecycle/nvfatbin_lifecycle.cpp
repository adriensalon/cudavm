#include "conformance.hpp"
#include <nvFatbin.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

namespace cudavm {
void check_fatbin(nvFatbinResult status, const char* operation) {
    if (status != NVFATBIN_SUCCESS) throw std::runtime_error(std::string(operation) + ": " + nvFatbinGetErrorString(status));
}
#define LIFECYCLE_FATBIN(call) ::cudavm::check_fatbin((call), #call)
template<class T> struct fatbin_guarded {
    alignas(T) std::array<unsigned char, sizeof(T) + 32> bytes;
    fatbin_guarded() { bytes.fill(0xe7); }
    T* data() { return reinterpret_cast<T*>(bytes.data() + 16); }
    T get() const {
        for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= sizeof(T) + 16) require(bytes[i] == 0xe7, "nvFatbin scalar output byte guards");
        T result{}; std::memcpy(&result, bytes.data() + 16, sizeof(T)); return result;
    }
};
struct fatbin_handle {
    nvFatbinHandle value{};
    explicit fatbin_handle(bool compress = false) {
        const char* options[]{"-64", "-cuda", compress ? "-compress-all" : "-compress=false"};
        LIFECYCLE_FATBIN(nvFatbinCreate(&value, options, 3)); require(value != nullptr, "successful nvFatbinCreate returns a handle");
    }
    ~fatbin_handle() { if (value) nvFatbinDestroy(&value); }
    fatbin_handle(const fatbin_handle&) = delete;
    fatbin_handle& operator=(const fatbin_handle&) = delete;
    void add(unsigned seed = 0) const {
        std::string ptx = ".version 6.5\n.target sm_75\n.address_size 64\n.visible .entry lifecycle_" + std::to_string(seed) + "() { ret; }\n";
        std::string architecture = "75", identifier = "lifecycle_" + std::to_string(seed), options;
        LIFECYCLE_FATBIN(nvFatbinAddPTX(value, ptx.data(), ptx.size() + 1, architecture.c_str(), identifier.c_str(), options.c_str()));
        // All argument strings die before the result is sized or retrieved.
    }
    std::vector<unsigned char> image() const {
        fatbin_guarded<size_t> extent, repeated; LIFECYCLE_FATBIN(nvFatbinSize(value, extent.data())); LIFECYCLE_FATBIN(nvFatbinSize(value, repeated.data()));
        const auto bytes = extent.get(); require(bytes > 0 && repeated.get() == bytes, "nvFatbinSize is nonempty and stable for unchanged handle");
        std::vector<unsigned char> first(bytes + 64, 0xa7), second(bytes + 64, 0xd3);
        LIFECYCLE_FATBIN(nvFatbinGet(value, first.data() + 32)); LIFECYCLE_FATBIN(nvFatbinGet(value, second.data() + 32));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 32 || i >= bytes + 32) require(first[i] == 0xa7 && second[i] == 0xd3, "nvFatbinGet exact extent and outer guards");
            else require(first[i] == second[i], "nvFatbinGet writes every reported byte and repeat retrieval agrees");
        }
        return {first.begin() + 32, first.end() - 32};
    }
};
void fatbin_discovery() {
    fatbin_guarded<unsigned> major, minor; LIFECYCLE_FATBIN(nvFatbinVersion(major.data(), minor.data()));
    require(major.get() == 13 && minor.get() == 4, "nvFatbin pinned public version");
    require(nvFatbinGetErrorString(NVFATBIN_SUCCESS) == nullptr, "nvFatbin success error string is null");
    for (const auto status : {NVFATBIN_ERROR_INTERNAL, NVFATBIN_ERROR_ELF_ARCH_MISMATCH, NVFATBIN_ERROR_ELF_SIZE_MISMATCH,
            NVFATBIN_ERROR_MISSING_PTX_VERSION, NVFATBIN_ERROR_NULL_POINTER, NVFATBIN_ERROR_COMPRESSION_FAILED,
            NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED, NVFATBIN_ERROR_UNRECOGNIZED_OPTION, NVFATBIN_ERROR_INVALID_ARCH,
            NVFATBIN_ERROR_INVALID_NVVM, NVFATBIN_ERROR_EMPTY_INPUT, NVFATBIN_ERROR_MISSING_PTX_ARCH, NVFATBIN_ERROR_PTX_ARCH_MISMATCH,
            NVFATBIN_ERROR_MISSING_FATBIN, NVFATBIN_ERROR_INVALID_INDEX, NVFATBIN_ERROR_IDENTIFIER_REUSE, NVFATBIN_ERROR_INTERNAL_PTX_OPTION}) {
        const char* text = nvFatbinGetErrorString(status); require(text && *text, "defined nvFatbin error has a description");
    }
    unsigned output{}; require(nvFatbinVersion(nullptr, &output) == NVFATBIN_ERROR_NULL_POINTER && nvFatbinVersion(&output, nullptr) == NVFATBIN_ERROR_NULL_POINTER, "nvFatbin null version outputs");
}
void fatbin_handles() {
    std::array<fatbin_guarded<nvFatbinHandle>, 24> handles; size_t created = 0;
    try {
        for (auto& handle : handles) {
            LIFECYCLE_FATBIN(nvFatbinCreate(handle.data(), nullptr, 0)); const auto value = handle.get(); require(value != nullptr, "live fatbin handle is nonnull");
            ++created; for (size_t i = 0; i + 1 < created; ++i) require(handles[i].get() != value, "simultaneously live fatbin handles are unique");
        }
        for (auto& handle : handles) { LIFECYCLE_FATBIN(nvFatbinDestroy(handle.data())); require(handle.get() == nullptr, "Destroy clears caller handle without touching guards"); }
    } catch (...) { for (size_t i = 0; i < created; ++i) if (handles[i].get()) nvFatbinDestroy(handles[i].data()); throw; }
}
void fatbin_outputs() {
    for (bool compression : {false, true}) for (unsigned seed : {0u, 71u, 0xffffffffu}) {
        fatbin_handle handle(compression); handle.add(seed); const auto first = handle.image();
        require(handle.image() == first, "unchanged handle retains its complete result across repeated Size/Get calls");
    }
}
void fatbin_lifetime() {
    for (bool compression : {false, true}) {
        std::vector<unsigned char> retained, before;
        {
            fatbin_handle handle(compression); handle.add(17); retained = handle.image(); before = retained;
            handle.add(29); const auto next = handle.image(); require(!next.empty() && retained == before, "later result generation preserves earlier caller-owned image");
            // Two independent copies survive both producer state and handle teardown.
            auto copy = next; require(copy == next, "opaque result can be copied by the caller");
        }
        for (unsigned i = 0; i < 8; ++i) { fatbin_handle unrelated(!compression); unrelated.add(113 + i); unrelated.image(); require(retained == before, "producer destruction and later handles preserve retained output bytes"); }
    }
}
void fatbin_validation() {
    require(nvFatbinCreate(nullptr, nullptr, 0) == NVFATBIN_ERROR_NULL_POINTER, "null create output");
    require(nvFatbinDestroy(nullptr) == NVFATBIN_ERROR_NULL_POINTER, "null destroy input");
    size_t size{}; unsigned char output[32]{};
    require(nvFatbinSize(nullptr, &size) == NVFATBIN_ERROR_NULL_POINTER && nvFatbinGet(nullptr, output) == NVFATBIN_ERROR_NULL_POINTER, "null result handles");
    fatbin_handle handle;
    require(nvFatbinGet(handle.value, output) == NVFATBIN_ERROR_INTERNAL, "Get requires successful Size first");
    require(nvFatbinSize(handle.value, nullptr) == NVFATBIN_ERROR_NULL_POINTER, "null size output");
    const std::string ptx = ".version 6.5\n.target sm_75\n.address_size 64\n.visible .entry validation() { ret; }\n";
    require(nvFatbinAddPTX(handle.value, ptx.c_str(), 0, "75", "empty", "") == NVFATBIN_ERROR_EMPTY_INPUT, "empty input status");
    require(nvFatbinAddPTX(handle.value, ptx.c_str(), ptx.size() + 1, "invalid", "architecture", "") == NVFATBIN_ERROR_INVALID_ARCH, "invalid architecture status");
    handle.add(137); handle.image(); require(nvFatbinGet(handle.value, nullptr) == NVFATBIN_ERROR_NULL_POINTER, "null completed-result buffer"); handle.image();
}
void fatbin_threading() {
    std::array<std::thread, 4> workers; std::array<std::exception_ptr, 4> failures{};
    const auto join = [&] { for (auto& worker : workers) if (worker.joinable()) worker.join(); };
    try {
        for (unsigned worker = 0; worker < workers.size(); ++worker) workers[worker] = std::thread([&, worker] {
            try { for (unsigned round = 0; round < 8; ++round) { fatbin_handle handle((worker & 1) != 0); handle.add(worker * 19 + round); const auto first = handle.image(); require(handle.image() == first, "concurrent independent fatbin handles retain their own outputs"); } }
            catch (...) { failures[worker] = std::current_exception(); }
        });
    } catch (...) { join(); throw; }
    join(); for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
}
std::string fatbin_case = "all";
void execute_nvfatbin_lifecycle() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (fatbin_case == "all" || fatbin_case == name) { execute(); selected = true; } };
    run("discovery", fatbin_discovery); run("handles", fatbin_handles); run("outputs", fatbin_outputs); run("lifetime", fatbin_lifetime); run("validation", fatbin_validation); run("threading", fatbin_threading);
    require(selected, "unknown nvfatbin_lifecycle case: " + fatbin_case);
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::fatbin_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvfatbin_lifecycle);
}
