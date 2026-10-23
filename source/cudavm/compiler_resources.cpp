#include "compiler.hpp"
#include "embedded_headers.hpp"
#include "device_math_source.hpp"
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <llvm/Support/MemoryBuffer.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace cudavm {
namespace {
std::string embedded_root() {
    return (std::filesystem::current_path() / "__cudavm_builtin_resources__").generic_string();
}
void module_marker() {}
std::filesystem::path module_directory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&module_marker), &module)) return {};
    std::wstring path(32768, L'\0');
    auto size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (!size || size >= path.size()) return {};
    path.resize(size);
    return std::filesystem::path(path).parent_path();
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&module_marker), &info) || !info.dli_fname) return {};
    return std::filesystem::absolute(info.dli_fname).parent_path();
#endif
}
frontend_resources installed(const std::filesystem::path& include) {
    for (const char* required : {"cudart/cuda_runtime.h", "nvcc/crt/host_defines.h", "cuda/cuda.h", "nvrtc/cudavm_nvrtc.h"}) {
        if (!std::filesystem::is_regular_file(include / required))
            throw std::runtime_error("installed SDK public header is unavailable: " + (include / required).string());
    }
    auto builtin = embedded_root();
    return {"installed-sdk", builtin, builtin + "/cuda_runtime_wrapper.h",
        (include / "cudart").string(), (include / "nvcc").string(), include.string()};
}
}
frontend_resources resolve_resources(const std::string& root) {
    if (!root.empty()) return installed((std::filesystem::absolute(root) / CUDAVM_INSTALL_INCLUDE_DIR).lexically_normal());
    auto module = module_directory();
    for (const auto& [includes, package] : {std::pair{CUDAVM_INCLUDES_FROM_BIN, CUDAVM_PACKAGE_FROM_BIN},
            std::pair{CUDAVM_INCLUDES_FROM_LIB, CUDAVM_PACKAGE_FROM_LIB}}) {
        auto candidate = (module / includes).lexically_normal();
        // The existing CMake package distinguishes an SDK from a nearby source
        // checkout's include tree. No compiler resource manifest is needed.
        if (std::filesystem::is_regular_file(module / package)
                && std::filesystem::is_regular_file(candidate / "nvrtc/cudavm_nvrtc.h")) return installed(candidate);
    }
    std::error_code error;
    if (!std::filesystem::equivalent(module, std::filesystem::path(CUDAVM_BUILD_MODULE_DIR), error)
        && !std::filesystem::equivalent(module, std::filesystem::path(CUDAVM_BUILD_NVCC_DIR), error))
        throw std::runtime_error("installed SDK public headers are unavailable; set resources_root to the SDK prefix for a copied/static compilation client");
    auto builtin = embedded_root();
    return {"build-tree", builtin, builtin + "/cuda_runtime_wrapper.h",
        CUDAVM_CUDA_INCLUDE_DIR, CUDAVM_CUDA_CRT_INCLUDE_DIR, CUDAVM_PUBLIC_INCLUDE_DIR};
}
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> frontend_files(const frontend_resources& value) {
    auto files = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    for (const auto& file : embedded_headers()) {
        auto path = value.resource_dir + "/" + file.path;
        // The data has static storage and its original trailing NUL. Keep it
        // in the library instead of duplicating all headers for each request.
        if (!files->addFile(path, 0, llvm::MemoryBuffer::getMemBuffer(
                llvm::StringRef(file.contents, file.size), path)))
            throw std::runtime_error("conflicting embedded compiler resource: " + path);
    }
    static const auto math_header=device_math_header();
    auto math_path=value.resource_dir+"/cudavm_device_math.h";
    if(!files->addFile(math_path,0,llvm::MemoryBuffer::getMemBuffer(math_header,math_path)))
        throw std::runtime_error("conflicting embedded device math header");
    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(files);
    return overlay;
}
llvm::json::Object describe_resources(const frontend_resources& value) {
    return llvm::json::Object{{"origin", value.origin}, {"frontend", "in-process-static-Clang"}, {"builtin_headers", "embedded"},
        {"public_headers", "module-include-directories"},
        {"cuda_include", value.cuda_include},
        {"cuda_crt_include", value.cuda_crt_include}, {"public_include", value.public_include}};
}
}
