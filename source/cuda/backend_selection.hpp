#pragma once
#include "backend.hpp"

namespace cudavm {
// The factory is injectable so failure, cleanup and selection order can be
// verified independently of the drivers installed on the test machine.
template<class Factory>
backend_selection discover_backend(cudavmBackend requested,
                                    std::span<const cudavmBackend> candidates, Factory factory) {
    auto discover = [&](cudavmBackend kind) {
        auto provider = factory(kind);
        auto devices = provider->discover();
        require(!devices.empty(), CUDA_ERROR_NO_DEVICE, "no GPU devices found");
        return backend_selection{kind, std::move(provider), std::move(devices)};
    };
    if (requested != cudavmBackendAuto) {
        require(std::find(candidates.begin(), candidates.end(), requested) != candidates.end(),
                CUDA_ERROR_NOT_SUPPORTED, "requested backend is disabled by this build or program policy");
        return discover(requested);
    }
    std::string reasons;
    for (auto kind : candidates) {
        try { return discover(kind); }
        catch (const failure& error) {
            // Allocation failures are not evidence that a different API is usable.
            if (error.code == CUDA_ERROR_OUT_OF_MEMORY) throw;
            if (!reasons.empty()) reasons += "; ";
            reasons += std::string(cudavmBackendName(kind)) + ": " + error.what();
        }
    }
    throw failure(CUDA_ERROR_NO_DEVICE, "no enabled backend found a GPU (" + reasons + ")");
}
}
