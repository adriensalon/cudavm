#pragma once
#include "backend.hpp"
#include <charconv>
#include <numeric>
#include <set>
#include <sstream>
#include <iomanip>

namespace cudavm {
// One provider owns each native loader. Published ordinals refer to endpoints,
// while physical indices are assigned before either cudavm environment filter.
struct device_identity {
    std::optional<pci_address> pci;
    std::optional<device_luid_info> luid;
    std::optional<std::array<char, 16>> uuid;
};
inline bool same_physical_device(const device_identity& a, const device_identity& b) {
    if (a.pci && b.pci) return a.pci == b.pci;
    if (a.luid && b.luid) return a.luid->bytes == b.luid->bytes && a.luid->node_mask == b.luid->node_mask;
    return a.uuid && b.uuid && a.uuid == b.uuid;
}
inline const char* backend_label(cudavmBackend kind) {
    switch (kind) {
    case cudavmBackendCuda: return "CUDA";
    case cudavmBackendVulkan: return "Vulkan";
    case cudavmBackendDirectX: return "DirectX";
    case cudavmBackendOpenCL: return "OpenCL";
    case cudavmBackendOpenGL: return "OpenGL";
    default: return "Unknown";
    }
}
inline const char* inventory_backend_name(cudavmBackend kind) {
    switch (kind) {
    case cudavmBackendCuda: return "cuda";
    case cudavmBackendVulkan: return "vulkan";
    case cudavmBackendDirectX: return "directx";
    case cudavmBackendOpenCL: return "opencl";
    case cudavmBackendOpenGL: return "opengl";
    default: return "invalid";
    }
}
inline std::vector<std::string_view> filter_items(const std::string& value, const char* variable) {
    std::vector<std::string_view> result;
    if (value.empty()) return result;
    std::string_view rest(value);
    for (;;) {
        auto comma = rest.find(','); auto item = rest.substr(0, comma);
        require(!item.empty(), CUDA_ERROR_INVALID_VALUE, std::string(variable) + " contains an empty selector");
        result.push_back(item);
        if (comma == std::string_view::npos) break;
        rest.remove_prefix(comma + 1);
    }
    return result;
}
struct device_inventory : backend {
    struct endpoint {
        device_info info;
        backend* provider;
        int ordinal;
        device_identity identity;
    };
    std::vector<std::unique_ptr<backend>> providers;
    std::vector<endpoint> devices;
    std::string discovery_diagnostic;
    std::vector<device_info> discover() override {
        std::vector<device_info> result;
        for (const auto& device : devices) result.push_back(device.info);
        return result;
    }
    const endpoint& at(int device) const { return devices.at(size_t(device)); }
    cudavmBackend device_kind(int device) const override { return at(device).info.kind; }
    int native_device(int device) const override { return at(device).ordinal; }
    std::unique_ptr<session> open(int device) override { auto& d = at(device); return d.provider->open(d.ordinal); }
    std::unique_ptr<session> open(int device, unsigned flags) override { auto& d = at(device); return d.provider->open(d.ordinal, flags); }
    std::array<char, 16> device_uuid(int device, bool instance) override { auto& d = at(device); return d.provider->device_uuid(d.ordinal, instance); }
    device_luid_info device_luid(int device) override { auto& d = at(device); return d.provider->device_luid(d.ordinal); }
    pci_address device_pci(int device) override { auto& d = at(device); return d.provider->device_pci(d.ordinal); }
    int attribute(int device, CUdevice_attribute which) override { auto& d = at(device); return d.provider->attribute(d.ordinal, which); }
    shader_resource_limits shader_limits(int device) override { auto& d = at(device); return d.provider->shader_limits(d.ordinal); }
    std::string link_registered_images(std::span<const std::string_view> images, int device) override {
        auto& d = at(device); return d.provider->link_registered_images(images, d.ordinal);
    }
    backend* cuda_provider() const {
        for (auto& d : devices) if (d.info.kind == cudavmBackendCuda) return d.provider;
        return nullptr;
    }
    CUmoduleLoadingMode loading_mode() override {
        if (auto* native = cuda_provider()) return native->loading_mode();
        return backend::loading_mode();
    }
    std::shared_ptr<device_library> load_library(const void* image, const char* path, CUjit_option* options, void** values,
            unsigned count, CUlibraryOption* library_options, void** library_values, unsigned library_count) override {
        if (auto* native = cuda_provider()) return native->load_library(image, path, options, values, count, library_options, library_values, library_count);
        return {};
    }
};
template<class F> auto optional_identity(F query) -> std::optional<decltype(query())> {
    try { return query(); }
    catch (const failure& error) {
        if (error.code == CUDA_ERROR_NOT_SUPPORTED || error.code == CUDA_ERROR_INVALID_VALUE) return {};
        throw;
    }
}
template<class Factory> std::unique_ptr<device_inventory> discover_devices(std::span<const cudavmBackend> candidates,
        const std::optional<std::string>& backends, const std::optional<std::string>& physical_devices, Factory factory) {
    std::set<cudavmBackend> allowed(candidates.begin(), candidates.end());
    if (backends) {
        allowed.clear();
        for (auto item : filter_items(*backends, "CUDAVM_BACKENDS")) {
            auto found = std::find_if(candidates.begin(), candidates.end(), [&](auto kind) { return item == inventory_backend_name(kind); });
            require(found != candidates.end(), CUDA_ERROR_INVALID_VALUE, "CUDAVM_BACKENDS names an unknown or unbuilt backend: " + std::string(item));
            allowed.insert(*found);
        }
    }
    auto result = std::make_unique<device_inventory>();
    for (auto kind : candidates) {
        try {
            auto provider = factory(kind); auto discovered = provider->discover();
            require(!discovered.empty(), CUDA_ERROR_NO_DEVICE, "no GPU devices found");
            std::vector<device_inventory::endpoint> endpoints;
            for (size_t i = 0; i < discovered.size(); ++i) {
                device_identity identity;
                identity.pci = optional_identity([&] { return provider->device_pci(int(i)); });
                identity.luid = optional_identity([&] { return provider->device_luid(int(i)); });
                identity.uuid = optional_identity([&] { return provider->device_uuid(int(i), true); });
                if (identity.uuid && std::all_of(identity.uuid->begin(), identity.uuid->end(), [](char c) { return c == 0; })) identity.uuid.reset();
                if (identity.luid && (!identity.luid->node_mask || std::all_of(identity.luid->bytes.begin(), identity.luid->bytes.end(), [](char c) { return c == 0; }))) identity.luid.reset();
                auto info = discovered[i]; info.kind = kind; info.name += " (" + std::string(backend_label(kind)) + ")";
                endpoints.push_back({std::move(info), provider.get(), int(i), identity});
            }
            result->providers.push_back(std::move(provider));
            for (auto& endpoint : endpoints) result->devices.push_back(std::move(endpoint));
        } catch (const failure& error) {
            if (error.code == CUDA_ERROR_OUT_OF_MEMORY) throw;
            if (!result->discovery_diagnostic.empty()) result->discovery_diagnostic += "; ";
            result->discovery_diagnostic += std::string(inventory_backend_name(kind)) + ": " + error.what();
        }
    }
    auto& devices = result->devices;
    require(!devices.empty() || (backends && backends->empty()) || (physical_devices && physical_devices->empty()),
            CUDA_ERROR_NO_DEVICE, "no backend discovered a GPU (" + result->discovery_diagnostic + ")");
    std::vector<size_t> parents(devices.size()); std::iota(parents.begin(), parents.end(), size_t(0));
    auto root = [&](size_t i) { while (parents[i] != i) i = parents[i]; return i; };
    for (size_t i = 0; i < devices.size(); ++i) for (size_t j = 0; j < i; ++j)
        if (same_physical_device(devices[i].identity, devices[j].identity)) parents[root(i)] = root(j);
    // PCI is the most useful machine-local sort key. Unknown identities retain
    // discovery order and are never guessed to refer to the same adapter.
    std::vector<size_t> groups;
    for (size_t i = 0; i < devices.size(); ++i) if (root(i) == i) groups.push_back(i);
    auto pci_key = [&](size_t group) {
        std::string key;
        for (size_t i = 0; i < devices.size(); ++i) if (root(i) == group && devices[i].identity.pci) {
            auto p = *devices[i].identity.pci; std::ostringstream text;
            text << std::hex << std::setfill('0') << std::setw(4) << p.domain << ':' << std::setw(2) << p.bus << ':' << std::setw(2) << p.device << '.' << p.function;
            if (key.empty() || text.str() < key) key = text.str();
        }
        return key;
    };
    std::stable_sort(groups.begin(), groups.end(), [&](size_t a, size_t b) {
        auto ka = pci_key(a), kb = pci_key(b);
        if (ka.empty() != kb.empty()) return !ka.empty();
        return ka < kb;
    });
    for (size_t i = 0; i < devices.size(); ++i)
        devices[i].info.physical_device = int(std::find(groups.begin(), groups.end(), root(i)) - groups.begin());
    std::set<int> selected;
    if (physical_devices) for (auto item : filter_items(*physical_devices, "CUDAVM_DEVICES")) {
        unsigned value = 0; auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        require(parsed.ec == std::errc{} && parsed.ptr == item.data() + item.size() && value < groups.size(),
                CUDA_ERROR_INVALID_VALUE, "CUDAVM_DEVICES contains an invalid physical GPU index: " + std::string(item));
        selected.insert(int(value));
    }
    std::erase_if(devices, [&](auto& d) { return !allowed.contains(d.info.kind) || (physical_devices && !selected.contains(d.info.physical_device)); });
    // Retain native CUDA order (including CUDA_DEVICE_ORDER). Shader endpoints
    // use physical order within the fixed backend priority.
    for (auto kind : candidates) if (kind != cudavmBackendCuda) {
        auto first = std::find_if(devices.begin(), devices.end(), [&](auto& d) { return d.info.kind == kind; });
        auto last = std::find_if(first, devices.end(), [&](auto& d) { return d.info.kind != kind; });
        std::stable_sort(first, last, [](auto& a, auto& b) { return a.info.physical_device < b.info.physical_device; });
    }
    return result;
}
}
