#include "test_framework.hpp"
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cudavm {
struct abi_exports_client {
#if defined(_WIN32)
    HMODULE value = LoadLibraryA(CUDAVM_ABI_CLIENT_PATH);
    ~abi_exports_client() { if (value) FreeLibrary(value); }
    void* symbol(const char* name) const { return value ? reinterpret_cast<void*>(GetProcAddress(value, name)) : nullptr; }
#else
    void* value = dlopen(CUDAVM_ABI_CLIENT_PATH, RTLD_NOW | RTLD_LOCAL);
    ~abi_exports_client() { if (value) dlclose(value); }
    void* symbol(const char* name) const { return value ? dlsym(value, name) : nullptr; }
#endif
};
std::string_view host_abi_exports_case = "all";
void execute_host_abi_exports() {
    abi_exports_client client;
    require(client.value != nullptr, "load independently built public ABI client");
    using entry = int (*)(const char*, const char*, unsigned);
    const auto execute = reinterpret_cast<entry>(client.symbol("cudavm_host_abi_client"));
    require(execute != nullptr, "resolve independent client's C entry point");
    const std::array<std::string_view, 3> cases{"driver", "runtime", "combined"};
    bool selected = false;
    for (unsigned group = 0; group < cases.size(); ++group) if (host_abi_exports_case == "all" || host_abi_exports_case == cases[group]) {
        selected = true;
        const auto result = execute(CUDAVM_ABI_DRIVER_PATH, CUDAVM_ABI_RUNTIME_PATH, group);
        require(result == 0, "independent " + std::string(cases[group]) + " ABI client status " + std::to_string(result));
    }
    require(selected, "unknown host_abi_exports case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::host_abi_exports_case = argv[1];
    return cudavm::run_host_test(argc, argv, cudavm::execute_host_abi_exports);
}
