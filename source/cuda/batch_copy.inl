struct batch_copy_entry {
    memory_box destination, source;
    size_t bytes;
    bool snapshot;
    std::vector<std::byte> storage;
};

void async_copy_batch(CUdeviceptr* destinations, CUdeviceptr* sources, size_t* sizes, size_t count,
                      CUmemcpyAttributes* attributes, size_t* indices, size_t attribute_count, CUstream stream, bool single = false) {
    auto owner = current();
    require(single || (stream && stream != CU_STREAM_LEGACY), CUDA_ERROR_INVALID_VALUE, "batch copies require a nonlegacy stream");
    auto target = stream_lookup(stream, owner);
    capture_check_stream(target);
    if (target->capture) {
        target->capture->invalidated = true;
        throw failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "batch copy capture is not implemented");
    }
    require(!count || (destinations && sources && sizes), CUDA_ERROR_INVALID_VALUE, "missing batch copy arrays");
    require(attribute_count <= count && (!count || attribute_count) &&
            (!attribute_count || (attributes && indices)), CUDA_ERROR_INVALID_VALUE, "missing batch copy attributes");
    for (size_t i = 0; i < attribute_count; ++i) {
        require((i ? indices[i] > indices[i - 1] : indices[i] == 0) && indices[i] < count,
                CUDA_ERROR_INVALID_VALUE, "invalid batch attribute partition");
        require(attributes[i].srcAccessOrder >= CU_MEMCPY_SRC_ACCESS_ORDER_STREAM &&
                attributes[i].srcAccessOrder <= CU_MEMCPY_SRC_ACCESS_ORDER_ANY &&
                !(attributes[i].flags & ~CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE),
                CUDA_ERROR_INVALID_VALUE, "invalid batch copy attributes");
    }
    std::vector<batch_copy_entry> copies;
    copies.reserve(count);
    std::vector<std::shared_ptr<host_entry>> hosts;
    auto retain_host = [&](const std::shared_ptr<host_entry>& host) {
        if (host && std::find(hosts.begin(), hosts.end(), host) == hosts.end()) hosts.push_back(host);
    };
    for (size_t i = 0, attribute = 0; i < count; ++i) {
        if (attribute + 1 < attribute_count && indices[attribute + 1] == i) ++attribute;
        auto bytes = sizes[i];
        auto source = memory_endpoint(CU_MEMORYTYPE_UNIFIED, nullptr, sources[i], bytes, 0, 0, 0, 0, bytes, 1, 1, owner, false);
        auto destination = memory_endpoint(CU_MEMORYTYPE_UNIFIED, nullptr, destinations[i], bytes, 0, 0, 0, 0, bytes, 1, 1, owner, false);
        if (bytes) memory_disjoint(destination, source, bytes);
        retain_host(destination.host); retain_host(source.host);
        copies.push_back({destination, source, bytes, attributes[attribute].srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL, {}});
    }
    // Preflight every endpoint before accessing ephemeral sources or publishing
    // work. The immutable command owns both descriptors and early source bytes.
    for (auto& copy : copies) if (copy.snapshot && copy.bytes) copy.storage.resize(copy.bytes);
    for (auto& copy : copies) if (copy.snapshot && copy.bytes) {
        auto staging = memory_endpoint(CU_MEMORYTYPE_HOST, copy.storage.data(), 0, copy.bytes, 0, 0, 0, 0, copy.bytes, 1, 1, owner, false);
        execute_copy_boxes(staging, copy.source, copy.bytes);
        copy.source = std::move(staging);
    }
    for (auto& host : hosts) {
        std::erase_if(host->work, [](auto& item) { return item.expired(); });
        host->work.reserve(host->work.size() + 1);
    }
    auto queue = stream_native(owner, target);
    auto done = submit_stream(owner, stream, {}, {}, 0, [copies = std::move(copies), queue] {
        auto group = std::make_shared<device_group>();
        group->tasks.reserve(copies.size());
        for (auto& copy : copies) group->tasks.push_back(enqueue_copy_boxes(copy.destination, copy.source, copy.bytes, queue.get()));
        return group;
    });
    for (auto& host : hosts) host_track(host, done);
}
