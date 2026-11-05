#include "conformance.hpp"
#include "managed_fixtures.hpp"
#include <cuda/cudavm_cuda.h>
#include <algorithm>
#include <cstring>

namespace cudavm {
void managed_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result) + "; " + cudavmGetDiagnostic());
}
#define MANAGED_RUNTIME(call) ::cudavm::managed_runtime((call), #call)
std::string_view managed_case;
constexpr size_t managed_word_count = 19;
struct managed_slot {
    uintptr_t before = 0x12345678; void* pointer{}; uintptr_t after = 0x87654321;
    void check() const { require(before == 0x12345678 && after == 0x87654321, "managed compiler pointer-slot guards"); }
};
struct managed_tokens { char kernel{}, ordinary{}; };
struct managed_image {
    void** value{};
    managed_image(managed_slot& slot, managed_tokens& tokens, unsigned version) {
        std::vector<unsigned char> bytes;
        if (managed_case == "ptx") bytes = registration_ptx_image(managed_ptx, 6, 5, 75);
        else { require(managed_case == "cubin", "managed input must be ptx or cubin"); bytes.assign(std::begin(managed_cubin), std::end(managed_cubin)); }
        const void* inputs[]{bytes.data(), nullptr};
        registration_wrapper wrapper{0x466243b1, version, bytes.data(), version == 2 ? inputs : nullptr};
        value = __cudaRegisterFatBinary(&wrapper); MANAGED_RUNTIME(cudaGetLastError()); require(value != nullptr, "managed registration handle");
        try {
            std::string kernel("managed_update"), managed("managed_words"), ordinary("ordinary_counter");
            __cudaRegisterFunction(value, &tokens.kernel, kernel.data(), kernel.c_str(), -1, nullptr, nullptr, nullptr, nullptr, nullptr);
            MANAGED_RUNTIME(cudaGetLastError());
            __cudaRegisterManagedVar(value, &slot.pointer, managed.data(), managed.c_str(), 0, managed_word_count * 4, 0, 0);
            MANAGED_RUNTIME(cudaGetLastError());
            __cudaRegisterVar(value, &tokens.ordinary, ordinary.data(), ordinary.c_str(), 0, 4, 0, 1);
            MANAGED_RUNTIME(cudaGetLastError());
            __cudaRegisterFatBinaryEnd(value); MANAGED_RUNTIME(cudaGetLastError());
            std::fill(bytes.begin(), bytes.end(), 0xa7);
            std::fill(kernel.begin(), kernel.end(), '!'); std::fill(managed.begin(), managed.end(), '!'); std::fill(ordinary.begin(), ordinary.end(), '!');
            slot.check();
        } catch (...) { __cudaUnregisterFatBinary(value); value = nullptr; throw; }
    }
    ~managed_image() { if (value) __cudaUnregisterFatBinary(value); }
    void initialize(managed_slot& slot) const {
        const auto result = __cudaInitModule(value); MANAGED_RUNTIME(cudaGetLastError());
        require(result == 1 && slot.pointer != nullptr, "managed initializer publishes host pointer"); slot.check();
    }
    void close() { __cudaUnregisterFatBinary(value); value = nullptr; MANAGED_RUNTIME(cudaGetLastError()); }
    managed_image(const managed_image&) = delete;
};
std::vector<uint32_t> managed_initial() {
    return {0x13579bdf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0xeca86420};
}
void managed_contents(const managed_slot& slot, std::span<const uint32_t> expected) {
    slot.check(); require(slot.pointer != nullptr, "managed pointer remains published");
    expect_words(std::span(static_cast<const uint32_t*>(slot.pointer), managed_word_count), expected, "managed host bytes and guards");
    for (const void* token : {static_cast<const void*>(&slot.pointer), static_cast<const void*>(slot.pointer)}) {
        void* address{}; size_t bytes = 0;
        MANAGED_RUNTIME(cudaGetSymbolAddress(&address, token)); MANAGED_RUNTIME(cudaGetSymbolSize(&bytes, token));
        require(address == slot.pointer && bytes == expected.size_bytes(), "managed slot and pointer symbol aliases");
    }
    unsigned is_managed = 0;
    CONFORMANCE_CUDA(cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, reinterpret_cast<uintptr_t>(slot.pointer)));
    require(is_managed == 1, "Driver identifies managed compiler storage");
    cudaPointerAttributes attributes{}; MANAGED_RUNTIME(cudaPointerGetAttributes(&attributes, slot.pointer));
    require(attributes.type == cudaMemoryTypeManaged && attributes.devicePointer == slot.pointer && attributes.hostPointer == slot.pointer,
            "Runtime identifies managed compiler aliases");
}
void managed_launch(const managed_slot& slot, const managed_tokens& tokens, std::vector<uint32_t>& expected,
                    uint32_t salt, uint32_t ordinary = 23) {
    constexpr unsigned active = 17, threads = 26, fields = 5;
    constexpr uint32_t guard = 0xa1b2c3d4;
    std::vector<uint32_t> output(threads * fields + 2), oracle(output.size(), guard);
    device_buffer buffer(output.size() * 4); CONFORMANCE_CUDA(cuMemsetD32(buffer.value, guard, output.size()));
    void* destination = reinterpret_cast<void*>(static_cast<uintptr_t>(buffer.value + 4));
    void* arguments[]{&destination, &salt};
    MANAGED_RUNTIME(cudaLaunchKernel(&tokens.kernel, dim3(2), dim3(13), arguments, 0, nullptr));
    MANAGED_RUNTIME(cudaDeviceSynchronize());
    CONFORMANCE_CUDA(cuMemcpyDtoH(output.data(), buffer.value, output.size() * 4));
    for (unsigned i = 0; i < active; ++i) {
        const auto old = expected[i + 1], next = old + (salt ^ (i * 0x1234567u));
        const uint64_t address = reinterpret_cast<uintptr_t>(slot.pointer) + (i + 1) * 4;
        const std::array<uint32_t, fields> row{old, next, uint32_t(address), uint32_t(address >> 32), ordinary};
        std::copy(row.begin(), row.end(), oracle.begin() + 1 + i * fields); expected[i + 1] = next;
    }
    expect_words(output, oracle, "managed registered kernel values, pointer relocations and guards");
    managed_contents(slot, expected);
}
void managed_cpu_update(managed_slot& slot, std::vector<uint32_t>& expected, uint32_t seed) {
    for (unsigned i = 1; i <= 17; ++i) expected[i] = seed + i * 0x87654321u;
    std::copy(expected.begin() + 1, expected.end() - 1, static_cast<uint32_t*>(slot.pointer) + 1);
    managed_contents(slot, expected);
}
void managed_ownership(unsigned version) {
    managed_slot slot; managed_tokens tokens;
    auto expected = managed_initial();
    managed_image first(slot, tokens, version); require(!slot.pointer, "registration defers pointer publication");
    first.initialize(slot); void* identity = slot.pointer; managed_contents(slot, expected);
    first.initialize(slot); require(slot.pointer == identity, "repeated managed initialization preserves address");
    managed_cpu_update(slot, expected, 0xfedcba98); managed_launch(slot, tokens, expected, 0x12345678);
    MANAGED_RUNTIME(cudaDeviceReset());
    require(slot.pointer == identity, "explicit Runtime reset retains compiler pointer");
    managed_contents(slot, expected); managed_launch(slot, tokens, expected, 0x89abcdef);
    first.close(); slot.check(); require(!slot.pointer, "final unregister clears compiler pointer");
    { managed_image replacement(slot, tokens, version); replacement.initialize(slot); managed_contents(slot, managed_initial()); }
    require(!slot.pointer, "replacement unregister clears compiler pointer"); slot.check();
}
void managed_coalescing(unsigned version, bool initialize_survivor) {
    managed_slot shared, isolated; managed_tokens first_tokens, second_tokens, independent_tokens;
    managed_image first(shared, first_tokens, version); first.initialize(shared);
    auto expected = managed_initial(); managed_cpu_update(shared, expected, 0x87654321);
    void* identity = shared.pointer;
    managed_image second(shared, second_tokens, version);
    if (initialize_survivor) {
        second.initialize(shared); require(shared.pointer == identity, "coalesced initialization preserves first pointer");
        managed_launch(shared, second_tokens, expected, 0xfedcba98);
    }
    managed_image independent(isolated, independent_tokens, version); independent.initialize(isolated);
    require(isolated.pointer != identity, "equal names in distinct compiler slots do not coalesce");
    auto other = managed_initial(); managed_contents(isolated, other); managed_launch(isolated, independent_tokens, other, 0x76543210);
    managed_contents(shared, expected);
    first.close(); require(shared.pointer == identity, "unregister preserves coalesced surviving owner");
    second.initialize(shared); managed_contents(shared, expected); managed_launch(shared, second_tokens, expected, 0x34567890);
    second.close(); require(!shared.pointer, "last coalesced owner clears pointer"); shared.check();
    managed_contents(isolated, other); independent.close(); require(!isolated.pointer, "independent owner clears own slot"); isolated.check();
}
void managed_contexts(unsigned version) {
    managed_slot slot; managed_tokens tokens; managed_image image(slot, tokens, version); image.initialize(slot);
    auto expected = managed_initial(); managed_cpu_update(slot, expected, 0x10293847);
    void* identity = slot.pointer;
    uint32_t original_ordinary = 0x76543210;
    MANAGED_RUNTIME(cudaMemcpyToSymbol(&tokens.ordinary, &original_ordinary, 4, 0, cudaMemcpyHostToDevice));
    managed_launch(slot, tokens, expected, 0x12345678, original_ordinary);
    CUcontext original{}, separate{}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuCtxCreate(&separate, nullptr, 0, device));
    try {
        image.initialize(slot); require(slot.pointer == identity, "second context retains managed address");
        managed_launch(slot, tokens, expected, 0x89abcdef);
        CONFORMANCE_CUDA(cuCtxSetCurrent(original)); managed_launch(slot, tokens, expected, 0x23456789, original_ordinary);
    } catch (...) { cuCtxSetCurrent(original); cuCtxDestroy(separate); throw; }
    CONFORMANCE_CUDA(cuCtxDestroy(separate)); require(slot.pointer == identity, "foreign context destruction preserves managed compiler storage");
    managed_contents(slot, expected);
    device_buffer survivor(4); CONFORMANCE_CUDA(cuMemsetD32(survivor.value, 0xabcdef01, 1));
    MANAGED_RUNTIME(cudaDeviceReset());
    require(slot.pointer == identity, "reset retains managed address"); managed_contents(slot, expected);
    managed_launch(slot, tokens, expected, 0x456789ab);
    uint32_t actual{}; CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, survivor.value, 4));
    require(actual == 0xabcdef01, "explicit reset preserves independent Driver storage");
}
void execute_managed_registration() {
    CUdevice device{}; int supported = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
    require_capability(supported != 0, "managed compiler storage requires managed-memory capability; this profile remains open on this endpoint");
    for (unsigned version : {1u, 2u}) {
        managed_ownership(version); managed_coalescing(version, false); managed_coalescing(version, true); managed_contexts(version);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    cudavm::managed_case = argv[2];
    return cudavm::run_conformance(2, argv, cudavm::execute_managed_registration);
}
