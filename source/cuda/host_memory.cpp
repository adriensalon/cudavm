#include "backend.hpp"
#include <map>
#include <mutex>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace cudavm {
namespace {
size_t page_size() {
#if defined(_WIN32)
    SYSTEM_INFO info{}; GetSystemInfo(&info); return info.dwPageSize;
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}
struct page_locks {
    std::mutex mutex;
    struct record { size_t references = 0; bool unlock = true; };
    std::map<uintptr_t, record> references;
};
page_locks& locks() { static auto* value = new page_locks; return *value; }
struct locked_host : host_allocation {
    bool owned;
    std::vector<uintptr_t> pages;
    locked_host(void* pointer, size_t bytes, bool allocate) : host_allocation(bytes), owned(allocate) {
        const auto page = page_size();
        if (owned) {
#if defined(_WIN32)
            address = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
            address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (address == MAP_FAILED) address = nullptr;
#endif
            require(address, CUDA_ERROR_OUT_OF_MEMORY, "host virtual allocation failed");
        } else address = pointer;
        try {
            auto start = reinterpret_cast<uintptr_t>(address);
            require(start && bytes && bytes <= std::numeric_limits<uintptr_t>::max() - start,
                    CUDA_ERROR_INVALID_VALUE, "invalid host registration range");
            auto first = start - start % page, end = start + bytes;
            pages.reserve((end - first - 1) / page + 1);
            auto& table = locks(); std::lock_guard lock(table.mutex);
            try {
                for (auto at = first; at < end;) {
                    auto [entry, inserted] = table.references.try_emplace(at);
                    if (inserted) {
#if defined(_WIN32)
                        MEMORY_BASIC_INFORMATION info{};
                        const auto access = VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info));
                        const auto protection = info.Protect & 0xff;
                        const bool writable = protection == PAGE_READWRITE || protection == PAGE_WRITECOPY ||
                                              protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY;
                        if (!access || info.State != MEM_COMMIT || !writable || (info.Protect & PAGE_GUARD)) {
                            table.references.erase(entry);
                            throw failure(CUDA_ERROR_INVALID_VALUE, "host registration requires committed writable memory");
                        }
                        PSAPI_WORKING_SET_EX_INFORMATION resident{}; resident.VirtualAddress = reinterpret_cast<void*>(at);
                        if (!K32QueryWorkingSetEx(GetCurrentProcess(), &resident, sizeof(resident))) {
                            table.references.erase(entry);
                            throw failure(CUDA_ERROR_INVALID_VALUE, "cannot inspect existing host page lock");
                        }
                        entry->second.unlock = !resident.VirtualAttributes.Locked;
                        if (!VirtualLock(reinterpret_cast<void*>(at), page)) {
                            auto error = GetLastError(); table.references.erase(entry);
                            throw failure(error == ERROR_WORKING_SET_QUOTA || error == ERROR_NOT_ENOUGH_MEMORY ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_VALUE,
                                          "host page locking failed: " + std::to_string(error));
                        }
#else
                        if (mlock(reinterpret_cast<void*>(at), page)) {
                            auto error = errno; table.references.erase(entry);
                            throw failure(error == ENOMEM || error == EAGAIN ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_VALUE,
                                          "host page locking failed: " + std::to_string(error));
                        }
#endif
                    }
                    ++entry->second.references; pages.push_back(at);
                    if (end - at <= page) break;
                    at += page;
                }
            } catch (...) { unlock_pages(table); throw; }
        } catch (...) { release_storage(); throw; }
    }
    void unlock_pages(page_locks& table) noexcept {
        for (auto at : pages) {
            auto entry = table.references.find(at);
            if (--entry->second.references == 0) {
                if (entry->second.unlock) {
#if defined(_WIN32)
                VirtualUnlock(reinterpret_cast<void*>(at), page_size());
#else
                munlock(reinterpret_cast<void*>(at), page_size());
#endif
                }
                table.references.erase(entry);
            }
        }
        pages.clear();
    }
    void release_storage() noexcept {
        if (owned && address) {
#if defined(_WIN32)
            VirtualFree(address, 0, MEM_RELEASE);
#else
            munmap(address, size);
#endif
            address = nullptr;
        }
    }
    ~locked_host() {
        auto& table = locks(); std::lock_guard lock(table.mutex);
        unlock_pages(table); release_storage();
    }
};
}
std::unique_ptr<host_allocation> allocate_locked_host(size_t bytes) { return std::make_unique<locked_host>(nullptr, bytes, true); }
std::unique_ptr<host_allocation> register_locked_host(void* pointer, size_t bytes) { return std::make_unique<locked_host>(pointer, bytes, false); }
}
