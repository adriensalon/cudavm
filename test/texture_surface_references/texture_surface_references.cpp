#include "test_framework.hpp"
#include "reference_fixture.hpp"

namespace cudavm {
std::string_view reference_case;
template<class T> struct reference_output {
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    reference_output() { before.fill(0x63); after.fill(0xa7); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto v) { return v == 0x63; })
            && std::all_of(after.begin(), after.end(), [](auto v) { return v == 0xa7; }), "reference query exact output guards"); return value;
    }
};
struct reference_module {
    CUmodule value{};
    reference_module() { CONFORMANCE_CUDA(cuModuleLoadData(&value, reference_image)); }
    ~reference_module() { if (value) cuModuleUnload(value); }
    void close() { CONFORMANCE_CUDA(cuModuleUnload(value)); value = nullptr; }
    CUfunction function(const char* name) const { CUfunction result{}; CONFORMANCE_CUDA(cuModuleGetFunction(&result, value, name)); return result; }
    CUtexref texture() const {
        reference_output<CUtexref> first, again;
        CONFORMANCE_CUDA(cuModuleGetTexRef(&first.value, value, "reference_texture"));
        CONFORMANCE_CUDA(cuModuleGetTexRef(&again.value, value, "reference_texture"));
        require(first.get() && first.get() == again.get(), "repeated texture reference lookup identity"); return first.get();
    }
    CUsurfref surface() const {
        reference_output<CUsurfref> first, again;
        CONFORMANCE_CUDA(cuModuleGetSurfRef(&first.value, value, "reference_surface"));
        CONFORMANCE_CUDA(cuModuleGetSurfRef(&again.value, value, "reference_surface"));
        require(first.get() && first.get() == again.get(), "repeated surface reference lookup identity"); return first.get();
    }
};
uint32_t reference_word(unsigned record, unsigned component, unsigned seed) {
    return std::bit_cast<uint32_t>(float(seed * 257 + record % 8 * 7 + record / 8 * 13 + component * 31));
}
void reference_initialize(image_test_storage& storage, unsigned seed) {
    for (unsigned record = 0; record < 32; ++record) for (unsigned c = 0; c < 4; ++c) {
        auto word = reference_word(record, c, seed); std::memcpy(storage.bytes.data() + (record * 4 + c) * 4, &word, 4);
    }
    storage.upload();
}
void reference_bind(CUtexref texture, CUsurfref surface, const image_test_storage& storage) {
    CONFORMANCE_CUDA(cuTexRefSetArray(texture, storage.array, CU_TRSA_OVERRIDE_FORMAT));
    CONFORMANCE_CUDA(cuSurfRefSetArray(surface, storage.array, 0));
    reference_output<CUarray> texture_array, surface_array;
    CONFORMANCE_CUDA(cuTexRefGetArray(&texture_array.value, texture)); CONFORMANCE_CUDA(cuSurfRefGetArray(&surface_array.value, surface));
    require(texture_array.get() == storage.array && surface_array.get() == storage.array, "reference binding queries preserve public array identity");
    CONFORMANCE_CUDA(cuTexRefSetFlags(texture, 0)); CONFORMANCE_CUDA(cuTexRefSetFilterMode(texture, CU_TR_FILTER_MODE_POINT));
    for (int axis = 0; axis < 3; ++axis) CONFORMANCE_CUDA(cuTexRefSetAddressMode(texture, axis, CU_TR_ADDRESS_MODE_CLAMP));
}
void reference_sample(reference_module& module, const char* name, unsigned seed, bool retire = false) {
    constexpr unsigned guards = 16;
    std::vector<uint32_t> result(guards + 32 * 4 + guards, 0xa7b96351);
    device_buffer output(result.size() * 4); CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, result.data(), result.size() * 4));
    auto pointer = output.value + guards * 4; void* args[]{&pointer};
    CONFORMANCE_CUDA(cuLaunchKernel(module.function(name), 1, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
    if (retire) module.close(); else CONFORMANCE_CUDA(cuCtxSynchronize());
    CONFORMANCE_CUDA(cuMemcpyDtoH(result.data(), output.value, result.size() * 4));
    for (unsigned i = 0; i < result.size(); ++i) {
        const auto expected = i < guards || i >= result.size() - guards ? 0xa7b96351 : reference_word((i - guards) / 4, (i - guards) % 4, seed);
        require(result[i] == expected, std::string(name) + " reference output/guard word=" + std::to_string(i));
    }
}
void reference_store(reference_module& module, image_test_storage& storage, unsigned seed) {
    constexpr unsigned guards = 16;
    std::vector<uint32_t> input(guards + 32 * 4 + guards, 0xa7b96351);
    for (unsigned i = 0; i < 32 * 4; ++i) input[guards + i] = reference_word(i / 4, i % 4, seed);
    device_buffer source(input.size() * 4); CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto pointer = source.value + guards * 4; void* args[]{&pointer};
    CONFORMANCE_CUDA(cuLaunchKernel(module.function("reference_write"), 1, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
    CONFORMANCE_CUDA(cuCtxSynchronize()); std::vector<uint32_t> preserved(input.size());
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4)); expect_words(preserved, input, "reference stores preserve source and guards");
    for (unsigned i = 0; i < 32 * 4; ++i) std::memcpy(storage.bytes.data() + i * 4, &input[guards + i], 4);
    storage.verify();
}
void reference_state(CUtexref texture) {
    for (auto format : {CU_AD_FORMAT_UNSIGNED_INT8, CU_AD_FORMAT_SIGNED_INT16, CU_AD_FORMAT_FLOAT}) for (int channels : {1, 2, 4}) {
        CONFORMANCE_CUDA(cuTexRefSetFormat(texture, format, channels));
        reference_output<CUarray_format> actual; reference_output<int> count;
        CONFORMANCE_CUDA(cuTexRefGetFormat(&actual.value, &count.value, texture));
        require(actual.get() == format && count.get() == channels, "reference format/channel state");
    }
    for (int axis = 0; axis < 3; ++axis) for (auto mode : {CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_MIRROR, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_BORDER}) {
        CONFORMANCE_CUDA(cuTexRefSetAddressMode(texture, axis, mode)); reference_output<CUaddress_mode> actual;
        CONFORMANCE_CUDA(cuTexRefGetAddressMode(&actual.value, texture, axis)); require(actual.get() == mode, "reference address mode state");
    }
    for (auto filter : {CU_TR_FILTER_MODE_POINT, CU_TR_FILTER_MODE_LINEAR}) {
        CONFORMANCE_CUDA(cuTexRefSetFilterMode(texture, filter)); reference_output<CUfilter_mode> actual;
        CONFORMANCE_CUDA(cuTexRefGetFilterMode(&actual.value, texture)); require(actual.get() == filter, "reference filter state");
    }
    for (unsigned flags : {0u, unsigned(CU_TRSF_READ_AS_INTEGER), unsigned(CU_TRSF_NORMALIZED_COORDINATES), unsigned(CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES)}) {
        CONFORMANCE_CUDA(cuTexRefSetFlags(texture, flags)); reference_output<unsigned> actual;
        CONFORMANCE_CUDA(cuTexRefGetFlags(&actual.value, texture)); require(actual.get() == flags, "reference flags state");
    }
}
void reference_validation(reference_module& module, CUtexref texture, CUsurfref surface, const image_test_storage& storage) {
    require(cuModuleGetTexRef(nullptr, module.value, "reference_texture") == CUDA_ERROR_INVALID_VALUE, "null reference lookup output");
    CUtexref missing{}; require(cuModuleGetTexRef(&missing, module.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null reference name");
    require(cuModuleGetTexRef(&missing, module.value, "missing_reference") == CUDA_ERROR_NOT_FOUND, "missing reference symbol");
    require(cuTexRefSetArray(texture, storage.array, 0) == CUDA_ERROR_INVALID_VALUE, "texture binding requires override-format flag");
    require(cuSurfRefSetArray(surface, storage.array, 1) == CUDA_ERROR_INVALID_VALUE, "surface binding flags must be zero");
    require(cuTexRefGetArray(nullptr, texture) == CUDA_ERROR_INVALID_VALUE, "null reference array output");
    require(cuSurfRefGetArray(nullptr, surface) == CUDA_ERROR_INVALID_VALUE, "null surface reference array output");
    CUarray missing_array{};
    require(cuTexRefGetArray(&missing_array, reinterpret_cast<CUtexref>(surface)) == CUDA_ERROR_INVALID_VALUE, "reference kinds are distinct");
    require(cuTexRefSetFormat(texture, CU_AD_FORMAT_FLOAT, 3) == CUDA_ERROR_INVALID_VALUE, "invalid reference channel count");
    require(cuTexRefSetAddressMode(texture, 3, CU_TR_ADDRESS_MODE_CLAMP) == CUDA_ERROR_INVALID_VALUE, "invalid reference axis");
    require(cuTexRefSetFilterMode(texture, static_cast<CUfilter_mode>(99)) == CUDA_ERROR_INVALID_VALUE, "invalid reference filter");
    require(cuTexRefSetFlags(texture, 0x80000000u) == CUDA_ERROR_INVALID_VALUE, "unknown reference flags");
}
void execute_image_references() {
    for (bool runtime : {false, true}) {
        reference_module module; auto texture = module.texture(); auto surface = module.surface();
        image_test_storage first(CU_RESOURCE_TYPE_ARRAY, CU_AD_FORMAT_FLOAT, cudaChannelFormatKindFloat, 32, 4, 8, 4, runtime, CUDA_ARRAY3D_SURFACE_LDST);
        image_test_storage second(CU_RESOURCE_TYPE_ARRAY, CU_AD_FORMAT_FLOAT, cudaChannelFormatKindFloat, 32, 4, 8, 4, !runtime, CUDA_ARRAY3D_SURFACE_LDST);
        reference_initialize(first, 1); reference_initialize(second, 7);
        if (reference_case == "state") reference_state(texture);
        if (reference_case == "validation") reference_validation(module, texture, surface, first);
        reference_bind(texture, surface, first);
        if (reference_case == "surface") {
            reference_store(module, first, 13); reference_sample(module, "reference_read", 13);
        } else if (reference_case == "rebind") {
            reference_sample(module, "reference_sample", 1); reference_bind(texture, surface, second);
            reference_sample(module, "reference_sample", 7); reference_store(module, second, 19);
            reference_sample(module, "reference_read", 19);
        } else reference_sample(module, "reference_sample", 1, reference_case == "lifetime");
        first.verify(); second.verify();
    }
}
void reference_probe() {
    constexpr char probe[] = R"ptx(.version 7.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 output) { .reg .u64 %out; ld.param.u64 %out,[output]; st.global.u32 [%out],0x19375928; ret; })ptx";
    CUmodule module{}; CONFORMANCE_CUDA(cuModuleLoadData(&module, probe));
    struct module_cleanup { CUmodule value; ~module_cleanup() { cuModuleUnload(value); } } cleanup{module};
    CUfunction function{}; CONFORMANCE_CUDA(cuModuleGetFunction(&function, module, "probe"));
    device_buffer output(4); auto pointer = output.value; void* args[]{&pointer};
    CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr)); CONFORMANCE_CUDA(cuCtxSynchronize());
    unsigned actual{}; CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, pointer, 4)); require(actual == 0x19375928, "independent reference execution probe");
}
void reference_discovery() {
    for (auto [name, version] : std::array<std::pair<const char*, int>, 6>{{{"cuModuleGetTexRef", 2000}, {"cuModuleGetSurfRef", 3000},
        {"cuTexRefSetArray", 2000}, {"cuSurfRefSetArray", 3000}, {"cuTexRefGetFlags", 2000}, {"cuTexRefGetFormat", 2000}}}) {
        void* driver{}; CUdriverProcAddressQueryResult query{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &driver, version - 1, CU_GET_PROC_ADDRESS_LEGACY_STREAM, &query));
        require(!driver && query == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "legacy reference introduction boundary");
        CONFORMANCE_CUDA(cuGetProcAddress(name, &driver, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM, &query));
        require(driver && query == CU_GET_PROC_ADDRESS_SUCCESS, "legacy reference discovery at introduction");
        void* runtime{}; cudaDriverEntryPointQueryResult other{};
        image_runtime(cudaGetDriverEntryPointByVersion(name, &runtime, version, cudaEnableLegacyStream, &other));
        require(runtime == driver && other == cudaDriverEntryPointSuccess, "Runtime resolves the same reference entry");
    }
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_conformance(argc, argv, cudavm::reference_discovery);
    if (argc == 3) { cudavm::reference_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_image_references, 0, cudavm::reference_probe);
}
