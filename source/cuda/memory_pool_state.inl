#pragma once
// Pool bookkeeping belongs to the Driver registry. Workers retain native
// buffers and completion records without reading or changing these fields.
struct memory_pool {
    CUmemoryPool handle = nullptr;
    CUmemPoolProps props{};
    bool builtin = false, destroyed = false;
    int reuse_events = 1, reuse_opportunistic = 1, reuse_internal = 1;
    uint64_t threshold = 0, used = 0, used_high = 0;
    std::shared_ptr<pool_backing_stats> backing = std::make_shared<pool_backing_stats>();
    struct cached_block {
        std::shared_ptr<context> owner;
        std::shared_ptr<buffer> native;
        CUdeviceptr address;
        unsigned long long stream_id;
        completion free;
    };
    std::vector<cached_block> cache;
    struct backing_block {
        std::shared_ptr<buffer> native;
        std::vector<std::weak_ptr<buffer>> slices;
    };
    std::vector<backing_block> blocks;
};
