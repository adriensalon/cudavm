#pragma once
#include "backend.hpp"
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cudavm {
class dynamic_library {
#if defined(_WIN32)
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
public:
    explicit dynamic_library(std::initializer_list<const char*> names, bool application_search = false, CUresult missing = CUDA_ERROR_NO_DEVICE) {
        for (const char* name : names) {
#if defined(_WIN32)
            handle_ = LoadLibraryExA(name, nullptr, application_search ? LOAD_LIBRARY_SEARCH_DEFAULT_DIRS : LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
            (void)application_search;
            handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
            if (handle_) break;
        }
        if (!handle_) throw failure(missing, application_search ? "shader compiler/validator library is unavailable" : "backend driver/loader library is unavailable");
    }
    ~dynamic_library() {
#if defined(_WIN32)
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
    }
    dynamic_library(const dynamic_library&) = delete;
    dynamic_library& operator=(const dynamic_library&) = delete;
    void* find(const char* name) const {
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(handle_, name));
#else
        return dlsym(handle_, name);
#endif
    }
    template<class T> T get(const char* name) const {
        auto value = reinterpret_cast<T>(find(name));
        if (!value) throw failure(CUDA_ERROR_NOT_SUPPORTED, std::string("backend entry point unavailable: ") + name);
        return value;
    }
};
}
