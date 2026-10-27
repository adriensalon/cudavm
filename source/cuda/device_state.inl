int provider_device_attribute(int device, CUdevice_attribute which) {
    if (device_kind(device) != cudavmBackendCuda && (which == CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID ||
        which == CU_DEVICE_ATTRIBUTE_PCI_BUS_ID || which == CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)) {
        try {
            auto pci = global().provider->device_pci(device);
            return int(which == CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID ? pci.domain : which == CU_DEVICE_ATTRIBUTE_PCI_BUS_ID ? pci.bus : pci.device);
        } catch (const failure& error) { if (error.code != CUDA_ERROR_NOT_SUPPORTED) throw; return 0; }
    }
    return global().provider->attribute(device, which);
}
pci_address parse_pci_address(const char* text) {
    require(text && *text, CUDA_ERROR_INVALID_VALUE, "missing PCI bus identifier");
    std::string_view value(text); pci_address result;
    auto number = [&](char delimiter, unsigned& output, bool last = false) {
        auto end = last ? value.size() : value.find(delimiter);
        require(end != std::string_view::npos && end, CUDA_ERROR_INVALID_VALUE, "invalid PCI bus identifier");
        auto parsed = std::from_chars(value.data(), value.data() + end, output, 16);
        require(parsed.ec == std::errc{} && parsed.ptr == value.data() + end, CUDA_ERROR_INVALID_VALUE, "invalid PCI bus component");
        value.remove_prefix(end + (last ? 0 : 1));
    };
    if (std::count(value.begin(), value.end(), ':') == 2) number(':', result.domain);
    else require(value.find('.') != std::string_view::npos, CUDA_ERROR_INVALID_VALUE, "short PCI identifiers require a function");
    number(':', result.bus);
    if (value.find('.') != std::string_view::npos) { number('.', result.device); number(0, result.function, true); }
    else number(0, result.device, true);
    require(value.empty() && result.domain <= 0xffff && result.bus <= 0xff && result.device <= 0x1f && result.function <= 7,
            CUDA_ERROR_INVALID_VALUE, "PCI bus component exceeds its range"); return result;
}
int device_from_pci(const char* text) {
    auto requested = parse_pci_address(text);
    if (auto native = global().provider->device_from_pci(text)) { valid_device(*native); return *native; }
    bool unsupported = false;
    for (size_t i = 0; i < global().devices.size(); ++i) {
        try { if (global().provider->device_pci(int(i)) == requested) return int(i); }
        catch (const failure& error) { if (error.code != CUDA_ERROR_NOT_SUPPORTED) throw; unsupported = true; }
    }
    require(!unsupported, CUDA_ERROR_NOT_SUPPORTED, "PCI bus information is unavailable on one or more devices");
    throw failure(CUDA_ERROR_INVALID_DEVICE, "PCI bus identifier does not name a selected device");
}
CUdevprop legacy_device_properties(int device) {
    auto get = [&](CUdevice_attribute attribute) { return global().provider->attribute(device, attribute); };
    CUdevprop result{}; result.maxThreadsPerBlock = get(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    for (int i = 0; i < 3; ++i) {
        result.maxThreadsDim[i] = get(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X + i));
        result.maxGridSize[i] = get(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i));
    }
    result.sharedMemPerBlock = get(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    result.totalConstantMemory = get(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY); result.SIMDWidth = get(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    result.memPitch = get(CU_DEVICE_ATTRIBUTE_MAX_PITCH); result.regsPerBlock = get(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    result.clockRate = get(CU_DEVICE_ATTRIBUTE_CLOCK_RATE); result.textureAlign = get(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT); return result;
}
