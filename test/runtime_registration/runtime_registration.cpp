#include "conformance.hpp"
#include "compiler_registration.hpp"
#include "registration_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <cuda/cudavm_cuda.h>
#include <algorithm>
#include <cstring>

namespace cudavm {
void registration_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result) + "; " + cudavmGetDiagnostic());
}
#define REGISTRATION_RUNTIME(call) ::cudavm::registration_runtime((call), #call)
std::string_view registration_case;
struct registration_tokens { char read{}, alias{}, update{}, global{}, constant{}; };
std::vector<unsigned char> registration_input() {
    if (registration_case == "cubin") return {std::begin(registration_cubin), std::end(registration_cubin)};
    require(registration_case == "ptx", "registration input must be ptx or cubin");
    return registration_ptx_image(registration_ptx, 6, 5, 75);
}
struct registered_image {
    void** value{};
    registered_image(registration_tokens& tokens, unsigned version) {
        auto bytes = registration_input();
        const void* inputs[]{bytes.data(), nullptr};
        registration_wrapper wrapper{0x466243b1, version, bytes.data(), version == 2 ? inputs : nullptr};
        value = __cudaRegisterFatBinary(&wrapper);
        REGISTRATION_RUNTIME(cudaGetLastError()); require(value != nullptr, "registration publishes a handle");
        try {
            auto function = [&](char* host, const char* name) {
                std::string device(name), display("different_display_name");
                __cudaRegisterFunction(value, host, device.data(), display.c_str(), -1, nullptr, nullptr, nullptr, nullptr, nullptr);
                REGISTRATION_RUNTIME(cudaGetLastError());
                std::fill(device.begin(), device.end(), '!'); std::fill(display.begin(), display.end(), '!');
            };
            function(&tokens.read, "read_symbols"); function(&tokens.alias, "read_symbols"); function(&tokens.update, "update_symbols");
            for (bool constant : {false, true}) {
                std::string name = constant ? "constant_words" : "mutable_words";
                __cudaRegisterVar(value, constant ? &tokens.constant : &tokens.global, nullptr, name.c_str(),
                                  0, constant ? 7 * 4 : 19 * 4, constant, !constant);
                REGISTRATION_RUNTIME(cudaGetLastError()); std::fill(name.begin(), name.end(), '!');
            }
            __cudaRegisterFatBinaryEnd(value); REGISTRATION_RUNTIME(cudaGetLastError());
            std::fill(bytes.begin(), bytes.end(), 0xa7);
        } catch (...) { __cudaUnregisterFatBinary(value); value = nullptr; throw; }
    }
    ~registered_image() { if (value) __cudaUnregisterFatBinary(value); }
    void close() { __cudaUnregisterFatBinary(value); value = nullptr; REGISTRATION_RUNTIME(cudaGetLastError()); }
    registered_image(const registered_image&) = delete;
};
std::vector<uint32_t> registration_initial() {
    return {0x13579bdf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0xeca86420};
}
constexpr std::array<uint32_t, 7> registration_constants{0, 1, 0x80000000, 0xffffffff, 0x12345678, 0xabcdef01, 17};
void registration_storage(const registration_tokens& tokens, std::span<const uint32_t> expected) {
    for (bool constant : {false, true}) {
        const void* token = constant ? &tokens.constant : &tokens.global;
        const auto contents = constant ? std::span<const uint32_t>(registration_constants) : expected;
        struct { size_t before = 0x12345678, value = 0, after = 0x87654321; } extent;
        REGISTRATION_RUNTIME(cudaGetSymbolSize(&extent.value, token));
        require(extent.value == contents.size_bytes() && extent.before == 0x12345678 && extent.after == 0x87654321, "registered symbol exact size and output guards");
        std::vector<uint32_t> actual(contents.size() + 2, 0xdeadbeef);
        REGISTRATION_RUNTIME(cudaMemcpyFromSymbol(actual.data() + 1, token, contents.size_bytes(), 0, cudaMemcpyDeviceToHost));
        require(actual.front() == 0xdeadbeef && actual.back() == 0xdeadbeef, "registered symbol readback guards");
        expect_words(std::span(actual).subspan(1, contents.size()), contents, "registered symbol bytes");
    }
}
void registration_mutate(const registration_tokens& tokens, std::vector<uint32_t>& expected, uint32_t salt) {
    void* arguments[]{&salt};
    REGISTRATION_RUNTIME(cudaLaunchKernel(&tokens.update, dim3(3), dim3(13), arguments, 0, nullptr));
    REGISTRATION_RUNTIME(cudaDeviceSynchronize());
    for (unsigned i = 0; i < 17; ++i) expected[i + 1] += salt ^ registration_constants[i % 7];
    registration_storage(tokens, expected);
}
void registration_read(const registration_tokens& tokens, std::span<const uint32_t> words) {
    cudaKernel_t first{}, alias{};
    REGISTRATION_RUNTIME(cudaGetKernel(&first, &tokens.read)); REGISTRATION_RUNTIME(cudaGetKernel(&alias, &tokens.alias));
    require(first && first == alias, "registered aliases select the same contextless kernel");
    void* global{}; void* constant{};
    REGISTRATION_RUNTIME(cudaGetSymbolAddress(&global, &tokens.global)); REGISTRATION_RUNTIME(cudaGetSymbolAddress(&constant, &tokens.constant));
    constexpr unsigned count = 19, total = 26, fields = 7;
    constexpr uint32_t guard = 0xa1b2c3d4;
    std::vector<uint32_t> expected(total * fields + 2, guard), actual(expected.size());
    device_buffer buffer(expected.size() * 4);
    void* output = reinterpret_cast<void*>(static_cast<uintptr_t>(buffer.value + 4));
    unsigned active = count;
    for (const char* token : {&tokens.read, &tokens.alias}) {
        CONFORMANCE_CUDA(cuMemsetD32(buffer.value, guard, expected.size()));
        void* arguments[]{&output, &active};
        REGISTRATION_RUNTIME(cudaLaunchKernel(token, dim3(2), dim3(13), arguments, 0, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), buffer.value, actual.size() * 4));
        for (unsigned i = 0; i < count; ++i) {
            const unsigned slot = i % 17 + 1, constant_slot = i % 7;
            const uint64_t address = reinterpret_cast<uintptr_t>(global) + slot * 4;
            const uint64_t constant_address = reinterpret_cast<uintptr_t>(constant) + constant_slot * 4;
            const std::array<uint32_t, fields> row{words[slot], registration_constants[constant_slot],
                (words[slot] ^ registration_constants[constant_slot]) + i, uint32_t(address), uint32_t(address >> 32),
                uint32_t(constant_address), uint32_t(constant_address >> 32)};
            std::copy(row.begin(), row.end(), expected.begin() + 1 + i * fields);
        }
        expect_words(actual, expected, "registered launch bytes, pointer identities, inactive records and guards");
    }
    registration_storage(tokens, words);
}
void registration_retired(const registration_tokens& tokens) {
    cudaKernel_t kernel{};
    require(cudaGetKernel(&kernel, &tokens.read) == cudaErrorInvalidDeviceFunction, "unregistered host function rejects lookup"); cudaGetLastError();
    size_t size = 0;
    require(cudaGetSymbolSize(&size, &tokens.global) == cudaErrorInvalidSymbol, "unregistered host symbol rejects lookup"); cudaGetLastError();
}
void registration_probe() {
    constexpr char source[] = ".version 6.5\n.target sm_75\n.address_size 64\n.visible .entry registration_probe(.param .u64 out) { .reg .b64 %p; .reg .b32 %v; ld.param.u64 %p,[out]; mov.u32 %v,0x43564d31; st.global.u32 [%p],%v; ret; }\n";
    CUmodule module{};
    CONFORMANCE_CUDA(cuModuleLoadData(&module, source));
    struct retire { CUmodule value; ~retire() { cuModuleUnload(value); } } owner{module};
    CUfunction function{}; CONFORMANCE_CUDA(cuModuleGetFunction(&function, module, "registration_probe"));
    device_buffer output(4); CONFORMANCE_CUDA(cuMemsetD32(output.value, 0, 1));
    void* arguments[]{&output.value};
    CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr));
    uint32_t actual{}; CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, output.value, 4));
    require(actual == 0x43564d31, "registration availability probe exact result");
}
void execute_runtime_registration() {
    registration_tokens tokens;
    for (unsigned version : {1u, 2u}) {
        auto expected = registration_initial();
        registered_image first(tokens, version);
        registration_storage(tokens, expected); registration_read(tokens, expected);
        for (unsigned i = 1; i <= 17; ++i) expected[i] = 0x98765432u + i * 0x1234567u;
        REGISTRATION_RUNTIME(cudaMemcpyToSymbol(&tokens.global, expected.data() + 1, 17 * 4, 4, cudaMemcpyHostToDevice));
        registration_mutate(tokens, expected, 0xfedcba98); registration_read(tokens, expected);
        {
            registered_image second(tokens, version);
            auto other = registration_initial(); registration_storage(tokens, other);
            registration_mutate(tokens, other, 0x76543210); registration_read(tokens, other);
            second.close();
        }
        registration_storage(tokens, expected); registration_read(tokens, expected);
        CUcontext original{}, separate{}; CUdevice device{};
        CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        CONFORMANCE_CUDA(cuCtxCreate(&separate, nullptr, 0, device));
        try {
            auto independent = registration_initial(); registration_storage(tokens, independent);
            registration_mutate(tokens, independent, 0xdeadbeef); registration_read(tokens, independent);
            CONFORMANCE_CUDA(cuCtxSetCurrent(original)); registration_storage(tokens, expected);
            CONFORMANCE_CUDA(cuCtxSetCurrent(separate)); registration_storage(tokens, independent);
        } catch (...) { cuCtxSetCurrent(original); cuCtxDestroy(separate); throw; }
        CONFORMANCE_CUDA(cuCtxSetCurrent(original)); CONFORMANCE_CUDA(cuCtxDestroy(separate));
        device_buffer survivor(4); CONFORMANCE_CUDA(cuMemsetD32(survivor.value, 0x10293847, 1));
        REGISTRATION_RUNTIME(cudaDeviceReset()); expected = registration_initial();
        registration_storage(tokens, expected); registration_read(tokens, expected);
        uint32_t preserved{}; CONFORMANCE_CUDA(cuMemcpyDtoH(&preserved, survivor.value, 4));
        require(preserved == 0x10293847, "explicit-context Runtime reset preserves Driver allocation");
        first.close(); registration_retired(tokens);
        { registered_image replacement(tokens, version); registration_storage(tokens, registration_initial()); registration_read(tokens, registration_initial()); }
        registration_retired(tokens);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    cudavm::registration_case = argv[2];
    return cudavm::run_conformance(2, argv, cudavm::execute_runtime_registration, 0, cudavm::registration_probe);
}
