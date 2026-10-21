// Compiler managed storage has registration lifetime, including facade epochs.
// Destroy the native library before its private binding context.
struct registered_managed_image {
    std::string image;
    std::vector<std::string> names;
    std::vector<const void*> slots;
    std::vector<size_t> sizes;
    std::shared_ptr<context> owner;
    std::shared_ptr<device_library> native;
    std::map<std::string, std::shared_ptr<buffer>> storage;
    CUlibrary facade = nullptr;
    uint64_t epoch = 0;
    bool ready = false;
    bool retired = false;
};
struct library_entry {
    std::shared_ptr<device_library> native;
    std::shared_ptr<registered_managed_image> registration;
    std::string image;
    std::vector<CUjit_option> options;
    std::vector<void*> values;
    std::vector<kernel_descriptor> descriptors;
    std::map<std::string, CUkernel> kernels;
    std::map<std::string, global_symbol> managed;
    std::unordered_map<uint64_t, CUmodule> modules;
    std::unordered_map<int, std::shared_ptr<context>> queries;
    std::unordered_set<std::string> unavailable_parameters;
    bool registered = false;
    bool retiring = false;
    std::unordered_set<uint64_t> resetting;
};
struct library_kernel_entry {
    CUlibrary library;
    std::string name;
    std::map<int, std::map<CUfunction_attribute, int>> attributes;
    std::map<int, CUfunc_cache> caches;
};
