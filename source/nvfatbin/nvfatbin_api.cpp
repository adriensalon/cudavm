#include "container_api.hpp"
#include <nvFatbin.h>
#include <nvfatbin/export.h>
#include <cctype>
#include <memory>
#include <mutex>
#include <unordered_map>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif

namespace cudavm {
namespace {
struct nvfatbin_program {
    std::mutex mutex;
    fatbin_compression compression;
    nvFatbinResult configuration = NVFATBIN_SUCCESS;
    std::vector<fatbin_record> records;
    size_t decoded_size = 0;
    std::string result;
};
struct nvfatbin_registry {
    std::mutex mutex;
    uintptr_t next = 0;
    std::unordered_map<nvFatbinHandle, std::shared_ptr<nvfatbin_program>> programs;
};
nvfatbin_registry& nvfatbin_programs() { static nvfatbin_registry value; return value; }
struct nvfatbin_error { nvFatbinResult result; };
void nvfatbin_require(bool valid, nvFatbinResult error = NVFATBIN_ERROR_NULL_POINTER) {
    if (!valid) throw nvfatbin_error{error};
}
template<class Function> nvFatbinResult nvfatbin_api(Function function) noexcept {
    try { function(); return NVFATBIN_SUCCESS; }
    catch (const nvfatbin_error& error) { return error.result; }
    catch (...) { return NVFATBIN_ERROR_INTERNAL; }
}
template<class Function> nvFatbinResult nvfatbin_with_program(nvFatbinHandle handle, Function function) noexcept {
    return nvfatbin_api([&] {
        nvfatbin_require(handle);
        std::shared_ptr<nvfatbin_program> program;
        { auto& registry = nvfatbin_programs(); std::lock_guard lock(registry.mutex);
          auto found = registry.programs.find(handle); nvfatbin_require(found != registry.programs.end(), NVFATBIN_ERROR_INTERNAL); program = found->second; }
        std::lock_guard lock(program->mutex); nvfatbin_require(program->configuration == NVFATBIN_SUCCESS, program->configuration);
        function(*program);
    });
}
void nvfatbin_options(nvfatbin_program& program, const char** options, size_t count) {
    bool none = false;
    for (size_t i = 0; i < count; ++i) {
        nvfatbin_require(options[i]); std::string option(options[i]);
        if (option == "-64" || option == "-cuda" || option == "-c") continue;
        if (option == "-compress=true") program.compression.enabled = true;
        else if (option == "-compress=false") program.compression.enabled = false;
        else if (option == "-compress-all") program.compression.force = true;
        else if (option.starts_with("-compress-mode=")) {
            auto mode = option.substr(15);
            nvfatbin_require(mode == "none" || mode == "default" || mode == "speed" || mode == "size" || mode == "balance", NVFATBIN_ERROR_UNRECOGNIZED_OPTION);
            none = mode == "none"; program.compression.threshold = mode == "size" ? 3 : 4;
        } else if (option == "-host=linux" || option == "-host=windows" || option == "-host=mac") continue;
        else throw nvfatbin_error{NVFATBIN_ERROR_UNRECOGNIZED_OPTION};
    }
    if (none) program.compression.enabled = false;
}
unsigned nvfatbin_architecture(std::string_view value) {
    constexpr unsigned architectures[]{50,52,53,60,61,62,70,72,75,80,86,87,89,90};
    for (auto arch : architectures) if (value == std::to_string(arch)) return arch;
    throw nvfatbin_error{NVFATBIN_ERROR_INVALID_ARCH};
}
void nvfatbin_add(nvfatbin_program& program, fatbin_kind kind, const void* data, size_t size,
                 const char* architecture, const char* identifier, const char* options) {
    program.result.clear();
    nvfatbin_require(data && architecture);
    nvfatbin_require(size != 0, NVFATBIN_ERROR_EMPTY_INPUT);
    nvfatbin_require(size <= fatbin_limit - program.decoded_size && program.records.size() < 4096, NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED);
    unsigned arch = nvfatbin_architecture(architecture);
    if (options) for (auto* p = options; *p; ++p) nvfatbin_require(std::isspace(static_cast<unsigned char>(*p)), NVFATBIN_ERROR_UNRECOGNIZED_OPTION);
    std::string code(static_cast<const char*>(data), size);
    if (kind == fatbin_kind::ptx) {
        while (!code.empty() && code.back() == 0) code.pop_back();
        nvfatbin_require(!code.empty(), NVFATBIN_ERROR_EMPTY_INPUT);
        nvfatbin_require(code.find('\0') == code.npos, NVFATBIN_ERROR_INTERNAL);
        auto header = fatbin_ptx_header(code);
        nvfatbin_require(header.size() >= 2 && header[0] == ".version", NVFATBIN_ERROR_MISSING_PTX_VERSION);
        auto dot = header[1].find('.');
        nvfatbin_require(dot != header[1].npos && dot && dot + 1 < header[1].size() && header[1].find('.', dot + 1) == header[1].npos
            && std::all_of(header[1].begin(), header[1].end(), [](char c) { return c == '.' || (c >= '0' && c <= '9'); }), NVFATBIN_ERROR_MISSING_PTX_VERSION);
        nvfatbin_require(header.size() >= 4 && header[2] == ".target", NVFATBIN_ERROR_MISSING_PTX_ARCH);
        nvfatbin_require(header[3] == "sm_" + std::to_string(arch), NVFATBIN_ERROR_PTX_ARCH_MISMATCH);
        nvfatbin_require(header.size() >= 6 && header[4] == ".address_size" && header[5] == "64", NVFATBIN_ERROR_INTERNAL);
        code += '\0';
    } else if (has_program_image(std::string_view(code))) {
        try {
            auto image = decode_program_image(code);
            nvfatbin_require(image.producer == program_image_ir_profile, NVFATBIN_ERROR_INVALID_NVVM);
            nvfatbin_require(image.architecture == "sm_" + std::to_string(arch), kind == fatbin_kind::cubin ? NVFATBIN_ERROR_ELF_ARCH_MISMATCH : NVFATBIN_ERROR_INVALID_ARCH);
        } catch (const nvfatbin_error&) { throw; }
        catch (...) { throw nvfatbin_error{kind == fatbin_kind::cubin ? NVFATBIN_ERROR_ELF_SIZE_MISMATCH : NVFATBIN_ERROR_INVALID_NVVM}; }
    } else if (kind == fatbin_kind::cubin) {
        unsigned actual;
        try { actual = fatbin_elf_architecture(code); }
        catch (...) { throw nvfatbin_error{NVFATBIN_ERROR_ELF_SIZE_MISMATCH}; }
        nvfatbin_require(actual == arch, NVFATBIN_ERROR_ELF_ARCH_MISMATCH);
    } else throw nvfatbin_error{NVFATBIN_ERROR_INVALID_NVVM};
    fatbin_record record{kind, arch, identifier ? identifier : "", std::move(code)};
    nvfatbin_require(record.identifier.size() <= 4096, NVFATBIN_ERROR_INTERNAL);
    nvfatbin_require(record.code.size() <= fatbin_limit - program.decoded_size, NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED);
    auto added_size = record.code.size();
    program.records.push_back(std::move(record)); program.decoded_size += added_size;
}
}
}
using namespace cudavm;
extern "C" CUDAVM_NVFATBIN_API const char* nvFatbinGetErrorString(nvFatbinResult result) {
    switch (result) {
    case NVFATBIN_SUCCESS: return nullptr;
#define CUDAVM_FATBIN_ERROR(name) case name: return #name
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_INTERNAL); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_ELF_ARCH_MISMATCH);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_ELF_SIZE_MISMATCH); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_MISSING_PTX_VERSION);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_NULL_POINTER); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_COMPRESSION_FAILED);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_UNRECOGNIZED_OPTION);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_INVALID_ARCH); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_INVALID_NVVM);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_EMPTY_INPUT); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_MISSING_PTX_ARCH);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_PTX_ARCH_MISMATCH); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_MISSING_FATBIN);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_INVALID_INDEX); CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_IDENTIFIER_REUSE);
    CUDAVM_FATBIN_ERROR(NVFATBIN_ERROR_INTERNAL_PTX_OPTION);
#undef CUDAVM_FATBIN_ERROR
    default: return "NVFATBIN_ERROR_UNKNOWN";
    }
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinCreate(nvFatbinHandle* output, const char** options, size_t count) {
    return nvfatbin_api([&] {
        nvfatbin_require(output && (!count || options));
        auto program = std::make_shared<nvfatbin_program>();
        program->configuration = nvfatbin_api([&] { nvfatbin_options(*program, options, count); });
        auto& registry = nvfatbin_programs(); std::lock_guard lock(registry.mutex);
        nvfatbin_require(registry.next != UINTPTR_MAX, NVFATBIN_ERROR_INTERNAL);
        auto handle = reinterpret_cast<nvFatbinHandle>(++registry.next); registry.programs.emplace(handle, program); *output = handle;
        nvfatbin_require(program->configuration == NVFATBIN_SUCCESS, program->configuration);
    });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinDestroy(nvFatbinHandle* handle) {
    return nvfatbin_api([&] { nvfatbin_require(handle && *handle); auto& registry = nvfatbin_programs(); std::lock_guard lock(registry.mutex);
        nvfatbin_require(registry.programs.erase(*handle) != 0, NVFATBIN_ERROR_INTERNAL); *handle = nullptr; });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinAddPTX(nvFatbinHandle handle, const char* data, size_t size, const char* arch, const char* id, const char* options) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program& program) { nvfatbin_add(program, fatbin_kind::ptx, data, size, arch, id, options); });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinAddCubin(nvFatbinHandle handle, const void* data, size_t size, const char* arch, const char* id) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program& program) { nvfatbin_add(program, fatbin_kind::cubin, data, size, arch, id, nullptr); });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinAddLTOIR(nvFatbinHandle handle, const void* data, size_t size, const char* arch, const char* id, const char* options) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program& program) { nvfatbin_add(program, fatbin_kind::ltoir, data, size, arch, id, options); });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinAddIndex(nvFatbinHandle handle, const void* data, size_t size, const char*) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program&) { nvfatbin_require(data); nvfatbin_require(size, NVFATBIN_ERROR_EMPTY_INPUT); throw nvfatbin_error{NVFATBIN_ERROR_INVALID_INDEX}; });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinAddReloc(nvFatbinHandle handle, const void* data, size_t size) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program&) { nvfatbin_require(data); nvfatbin_require(size, NVFATBIN_ERROR_EMPTY_INPUT); throw nvfatbin_error{NVFATBIN_ERROR_MISSING_FATBIN}; });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinAddTileIR(nvFatbinHandle handle, const void* data, size_t size, const char*, const char*) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program&) { nvfatbin_require(data); nvfatbin_require(size, NVFATBIN_ERROR_EMPTY_INPUT); throw nvfatbin_error{NVFATBIN_ERROR_INTERNAL}; });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinSize(nvFatbinHandle handle, size_t* size) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program& program) { nvfatbin_require(size);
        if (program.result.empty()) program.result = encode_fatbin(program.records, program.compression); *size = program.result.size(); });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinGet(nvFatbinHandle handle, void* output) {
    return nvfatbin_with_program(handle, [&](nvfatbin_program& program) { nvfatbin_require(output); nvfatbin_require(!program.result.empty(), NVFATBIN_ERROR_INTERNAL);
        std::memcpy(output, program.result.data(), program.result.size()); });
}
extern "C" CUDAVM_NVFATBIN_API nvFatbinResult nvFatbinVersion(unsigned* major, unsigned* minor) {
    return nvfatbin_api([&] { nvfatbin_require(major && minor); *major = 13; *minor = 4; });
}
