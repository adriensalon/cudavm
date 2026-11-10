#define __CUDA_API_VERSION_INTERNAL
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda.h>
#include "../../source/cuda/runtime_handshake.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cudavm {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(CUresult value, const char* message) { require(value == CUDA_SUCCESS, message); }
template<class Function> Function symbol(HMODULE module, const char* name) {
    auto result = reinterpret_cast<Function>(GetProcAddress(module, name)); require(result != nullptr, name); return result;
}
struct runtime_test_driver {
    HMODULE module;
    decltype(&cuGetExportTable) exports = symbol<decltype(exports)>(module, "cuGetExportTable");
    decltype(&cuCtxGetCurrent) current = symbol<decltype(current)>(module, "cuCtxGetCurrent");
    decltype(&cuCtxSetCurrent) set_current = symbol<decltype(set_current)>(module, "cuCtxSetCurrent");
    decltype(&cuCtxCreate_v2) create = symbol<decltype(create)>(module, "cuCtxCreate_v2");
    decltype(&cuCtxDestroy_v2) destroy = symbol<decltype(destroy)>(module, "cuCtxDestroy_v2");
    decltype(&cuDevicePrimaryCtxRetain) retain = symbol<decltype(retain)>(module, "cuDevicePrimaryCtxRetain");
    decltype(&cuDevicePrimaryCtxRelease_v2) release = symbol<decltype(release)>(module, "cuDevicePrimaryCtxRelease_v2");
    decltype(&cuDevicePrimaryCtxGetState) state = symbol<decltype(state)>(module, "cuDevicePrimaryCtxGetState");
    const uintptr_t* table(const char* hex) {
        CUuuid id{};
        for (size_t i = 0; i < 16; ++i) {
            auto digit = [](char c) -> unsigned { return c >= '0' && c <= '9' ? unsigned(c - '0') : unsigned(c - 'a' + 10); };
            id.bytes[i] = char(digit(hex[i * 2]) * 16 + digit(hex[i * 2 + 1]));
        }
        const void* output{}; check(exports(&output, &id), "private UUID lookup");
        return static_cast<const uintptr_t*>(output);
    }
};
void md2_vectors() {
    // Published algorithm vectors, independent of the CUDA handshake machinery:
    // https://www.rfc-editor.org/rfc/rfc1319.html#appendix-A.5
    for (auto [message, expected] : {
        std::pair{"", "8350e5a3e24c153df2275c9f80692773"},
        std::pair{"a", "32ec01ec4a6dac72c0ab96fb34c0b5d1"},
        std::pair{"abc", "da853b0d3f88d99b30283a69e6ded6bb"},
        std::pair{"message digest", "ab4f496bfb2a530b219ff33031fe06b0"},
        std::pair{"abcdefghijklmnopqrstuvwxyz", "4e8ddff3650292ab5a4108c3aa47940b"},
        std::pair{"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", "da33def2a42df13975352846c30338cd"}}) {
        runtime_md2 hash;
        for (auto c : std::string_view(message)) hash.append(uint8_t(c));
        auto bytes = hash.finish(); std::string actual;
        for (auto byte : bytes) { char hex[3]; std::snprintf(hex, sizeof(hex), "%02x", byte); actual += hex; }
        if (actual != expected) throw std::runtime_error(std::string("RFC 1319 MD2 vector '") + message + "': " + actual + " != " + expected);
    }
}
void handshake(runtime_test_driver& driver) {
    auto runtime = driver.table("6bd5fb6c5bf4e74a8987d93912fd9df9");
    auto table = driver.table("d4082055bde6704b8d34ba123c66e1f2");
    require(runtime[0] >= 0x58 && table[0] >= 0x18, "versioned private table extents");
    auto query = reinterpret_cast<CUresult(CUDAAPI*)(unsigned, uint64_t, void*)>(table[1]);
    int version{}, count{};
    check(symbol<decltype(&cuDriverGetVersion)>(driver.module, "cuDriverGetVersion")(&version), "Driver version");
    check(symbol<decltype(&cuDeviceGetCount)>(driver.module, "cuDeviceGetCount")(&count), "device count");
    for (auto nonce : {uint64_t(0), uint64_t(0x123456789abcdef), ~uint64_t(0)})
        for (unsigned request = 12040; request <= 12042; ++request) {
            std::array<uint8_t, 18> guarded{}; guarded.fill(0xa7);
            std::array<uint8_t, 16> expected{0x5c,0x67,0xcb,0x03,0x1c,0x18,0x41,0x33,0xe8,0xd1,0x4c,0x1f,0xaa,0x83,0xd3,0x8e};
            if (request == 12041) expected[7] = 0x18;
            if (request == 12042) {
                auto hash = runtime_handshake_begin();
                for (auto word : {uint32_t(version), request, uint32_t(GetCurrentProcessId()), uint32_t(GetCurrentThreadId())}) runtime_handshake_append(hash, word);
                for (auto word : {uint64_t(reinterpret_cast<uintptr_t>(runtime)), uint64_t(reinterpret_cast<uintptr_t>(table)), uint64_t(table[1]), nonce}) runtime_handshake_append(hash, word);
                for (int device = 0; device < count; ++device) {
                    CUuuid id{};
                    check(symbol<decltype(&cuDeviceGetUuid)>(driver.module, "cuDeviceGetUuid")(&id, device), "legacy UUID");
                    hash.append({reinterpret_cast<const uint8_t*>(id.bytes), 16});
                    for (auto attribute : {CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID}) {
                        int value{};
                        check(symbol<decltype(&cuDeviceGetAttribute)>(driver.module, "cuDeviceGetAttribute")(&value, attribute, device), "PCI identity");
                        runtime_handshake_append(hash, uint32_t(value));
                    }
                }
                expected = runtime_handshake_finish(hash);
            }
            check(query(request, nonce, guarded.data() + 1), "private loader handshake");
            require(guarded.front() == 0xa7 && guarded.back() == 0xa7 && std::equal(expected.begin(), expected.end(), guarded.begin() + 1), "native differential handshake and output guards");
        }
}
struct runtime_test_callback { CUcontext owner{}; void* key{}; void* value{}; unsigned calls = 0; } callback;
runtime_test_driver* callback_driver{};
CUresult callback_status = CUDA_ERROR_UNKNOWN;
void CUDAAPI storage_destroy(CUcontext owner, void* key, void* value) {
    callback.owner = owner; callback.key = key; callback.value = value; ++callback.calls;
    // A callback can reenter the Driver without deadlocking the facade mutex.
    CUcontext current{}; callback_status = callback_driver->current(&current);
}
void storage(runtime_test_driver& driver) {
    auto table = driver.table("c693336e1121df11a8c368f355d89593");
    auto insert = reinterpret_cast<CUresult(CUDAAPI*)(CUcontext, void*, void*, void(CUDAAPI*)(CUcontext, void*, void*))>(table[0]);
    auto remove = reinterpret_cast<CUresult(CUDAAPI*)(CUcontext, void*)>(table[1]);
    auto find = reinterpret_cast<CUresult(CUDAAPI*)(void**, CUcontext, void*)>(table[2]);
    CUcontext first{}, second{}; callback_driver = &driver;
    check(driver.create(&first, 0, 0), "first context"); check(driver.create(&second, 0, 0), "second context");
    auto tls = driver.table("42d85a8123f6cb478298f6e78a3aecdc");
    require(tls[0] >= 0x18, "tooling context table extent");
    auto set_tls = reinterpret_cast<CUresult(CUDAAPI*)(CUcontext)>(tls[1]);
    auto get_tls = reinterpret_cast<CUresult(CUDAAPI*)(CUcontext*)>(tls[2]);
    CUcontext tls_current{};
    check(set_tls(first), "private context binding"); check(get_tls(&tls_current), "private context lookup");
    require(tls_current == first, "private context round trip");
    int key{}, value{}, other{}; void* result = &other;
    check(insert(first, &key, &value, storage_destroy), "insert storage");
    require(insert(first, &key, &other, storage_destroy) == CUDA_ERROR_INVALID_HANDLE, "duplicate key rejected");
    require(find(&result, second, &key) == CUDA_ERROR_INVALID_HANDLE && result == &other, "context isolation and failed output preservation");
    check(driver.set_current(first), "bind first context");
    check(find(&result, nullptr, &key), "null-context current lookup"); require(result == &value, "lookup value");
    check(remove(first, &key), "remove storage"); require(!callback.calls, "explicit remove does not invoke destructor");
    require(remove(first, &key) == CUDA_ERROR_INVALID_HANDLE, "missing key removal");
    check(insert(first, &key, &value, storage_destroy), "reinsert storage");
    check(insert(second, &key, &other, nullptr), "same key in independent context");
    check(driver.destroy(first), "destroy storage owner");
    require(callback.calls == 1 && callback.owner == first && callback.key == &key && callback.value == &value, "one destructor with original context, key and value");
    check(callback_status, "callback Driver reentry");
    check(find(&result, second, &key), "surviving context storage"); require(result == &other, "other context survives");
    check(driver.destroy(second), "destroy second context");
}
void primary(runtime_test_driver& driver) {
    auto runtime = driver.table("6bd5fb6c5bf4e74a8987d93912fd9df9");
    auto reserve = reinterpret_cast<CUresult(CUDAAPI*)(CUcontext*, CUdevice)>(runtime[2]);
    CUcontext first{}, second{}, current{}; unsigned flags{}; int active{};
    check(driver.set_current(nullptr), "clear current context");
    check(reserve(&first, 0), "reserve primary context"); check(reserve(&second, 0), "repeat primary reservation");
    check(driver.state(0, &flags, &active), "primary state after private reservation");
    check(driver.current(&current), "current after reservation");
    require(first && first == second && !active && !current, "private reservation neither activates nor binds the primary");
    check(driver.retain(&second, 0), "retain reserved primary"); require(first == second, "public retain preserves private identity");
    check(driver.release(0), "release primary");
}
}
int main(int argc, char** argv) {
    try {
        cudavm::require(argc == 2, "Driver path or native required"); cudavm::md2_vectors();
        std::filesystem::path path(argv[1]);
        if (path == "native") {
            wchar_t system[32768]{}; cudavm::require(GetSystemDirectoryW(system, 32768) != 0, "system directory");
            path = std::filesystem::path(system) / L"nvcuda.dll";
        }
        auto module = LoadLibraryW(path.c_str()); if (!module && std::string_view(argv[1]) == "native") return 77;
        cudavm::require(module != nullptr, "load selected Driver");
        auto status = cudavm::symbol<decltype(&cuInit)>(module, "cuInit")(0);
        if (status == CUDA_ERROR_NO_DEVICE) return 77;
        cudavm::check(status, "initialize Driver");
        cudavm::runtime_test_driver driver{module};
        CUuuid unknown{}; const void* output = reinterpret_cast<void*>(0x1234);
        cudavm::require(driver.exports(&output, &unknown) == CUDA_ERROR_INVALID_VALUE && output == nullptr, "unknown UUID clears output like the native Driver");
        auto callbacks = driver.table("a094798c2e742e7493f20800200c0a66");
        cudavm::require(callbacks[0] >= 0x38, "tooling callback table extent");
        for (auto [slot, expected] : {std::pair{2u, size_t(1024)}, std::pair{6u, size_t(14)}}) {
            const void* flags{}; size_t count{};
            reinterpret_cast<void(CUDAAPI*)(const void**, size_t*)>(callbacks[slot])(&flags, &count);
            cudavm::require(flags && count == expected, "native callback/domain flag extents");
            if (std::string_view(argv[1]) != "native") {
                auto words = static_cast<const uint32_t*>(flags);
                cudavm::require(std::all_of(words, words + count, [](auto flag) { return flag == 0; }), "private tooling stays disabled");
            }
        }
        cudavm::handshake(driver); cudavm::primary(driver); cudavm::storage(driver);
        std::puts("PASS: private Runtime export-table contracts and native differential protocol");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "Runtime exports: %s\n", error.what()); return 1; }
}
