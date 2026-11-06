void decompression_ready(size_t* error_index) {
    try { driver_ready(); }
    catch (...) { if (error_index) *error_index = std::numeric_limits<size_t>::max(); throw; }
}
void async_decompress(CUmemDecompressParams* input, size_t count, unsigned flags, size_t* error_index, CUstream stream) {
    size_t index = std::numeric_limits<size_t>::max();
    try {
        require(!flags && (!count || input) && count <= std::numeric_limits<size_t>::max() / sizeof(CUmemDecompressParams),
                CUDA_ERROR_INVALID_VALUE, "invalid decompression batch flags or descriptor array");
        auto target = stream_lookup(stream); auto owner = target->owner.lock(); capture_check_stream(target);
        if (target->capture) {
            target->capture->invalidated = true;
            throw failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "decompression capture is not implemented");
        }
        std::vector<CUmemDecompressParams> records;
        if (count) records.assign(input, input + count);
        std::vector<std::shared_ptr<buffer>> storage;
        storage.reserve(count * 3);
        for (index = 0; index < count; ++index) {
            const auto& record = records[index];
            require(std::all_of(std::begin(record.padding), std::end(record.padding), [](auto byte) { return byte == 0; }) &&
                    (record.algo == CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE || record.algo == CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY || record.algo == CU_MEM_DECOMPRESS_ALGORITHM_LZ4),
                    CUDA_ERROR_INVALID_VALUE, "invalid decompression algorithm or reserved padding");
            require(record.src && record.dst && record.dstActBytes && record.srcNumBytes &&
                    reinterpret_cast<uintptr_t>(record.dstActBytes) % alignof(cuuint32_t) == 0,
                    CUDA_ERROR_INVALID_VALUE, "invalid decompression endpoint or actual-size alignment");
            storage.push_back(resolve(reinterpret_cast<uintptr_t>(record.src), record.srcNumBytes, owner).value->native);
            // dstNumBytes is only a scheduling hint. The caller must provide
            // enough destination storage for the stream's actual output.
            storage.push_back(resolve(reinterpret_cast<uintptr_t>(record.dst), 1, owner).value->native);
            storage.push_back(resolve(reinterpret_cast<uintptr_t>(record.dstActBytes), sizeof(cuuint32_t), owner).value->native);
        }
        index = std::numeric_limits<size_t>::max();
        const auto mask = global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK);
        require(mask != 0, CUDA_ERROR_NOT_SUPPORTED, "device has no advertised hardware memory decompression");
        const auto maximum = global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH);
        for (index = 0; index < count; ++index) {
            require(mask & records[index].algo, CUDA_ERROR_NOT_SUPPORTED, "decompression algorithm is unavailable on this device");
            require(maximum > 0 && records[index].srcNumBytes <= size_t(maximum), CUDA_ERROR_INVALID_VALUE, "compressed input exceeds device limit");
            for (size_t endpoint = 0; endpoint < 3; ++endpoint)
                require(storage[index * 3 + endpoint]->storage().hw_decompress_capable(), CUDA_ERROR_INVALID_VALUE, "allocation is not hardware-decompression capable");
        }
        index = std::numeric_limits<size_t>::max(); auto queue = stream_native(owner, target);
        submit_stream(owner, stream, {}, {}, 0, [owner, records = std::move(records), storage = std::move(storage), queue] {
            return owner->native->enqueue_decompress(records, queue.get());
        }, false, {}, queue);
    } catch (...) { if (error_index) *error_index = index; throw; }
}
