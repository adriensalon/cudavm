// Descriptions are copied and validated before any work is submitted. Workers
// retain native resources and never inspect the Driver registry.
bool memop_wait(CUstreamBatchMemOpType type) {
    return type == CU_STREAM_MEM_OP_WAIT_VALUE_32 || type == CU_STREAM_MEM_OP_WAIT_VALUE_64;
}
bool memop_write(CUstreamBatchMemOpType type) {
    return type == CU_STREAM_MEM_OP_WRITE_VALUE_32 || type == CU_STREAM_MEM_OP_WRITE_VALUE_64;
}
size_t memop_width(CUstreamBatchMemOpType type) {
    return type == CU_STREAM_MEM_OP_WAIT_VALUE_64 || type == CU_STREAM_MEM_OP_WRITE_VALUE_64 ? 8 : 4;
}
CUdeviceptr memop_address(const CUstreamBatchMemOpParams& op) {
    return memop_wait(op.operation) ? op.waitValue.address : memop_write(op.operation) ? op.writeValue.address : 0;
}
std::vector<CUstreamBatchMemOpParams> memop_parameters(unsigned count, const CUstreamBatchMemOpParams* params, unsigned flags) {
    require(!flags && count < 256 && (!count || params), CUDA_ERROR_INVALID_VALUE, "invalid batch memory operation array or flags");
    std::vector<CUstreamBatchMemOpParams> result(count);
    for (unsigned i = 0; i < count; ++i) {
        auto& op = params[i]; auto& copy = result[i]; copy.operation = op.operation;
        if (memop_wait(op.operation)) {
            require(!(op.waitValue.flags & ~(3u | CU_STREAM_WAIT_VALUE_FLUSH)), CUDA_ERROR_INVALID_VALUE, "invalid memory wait flags");
            require(!(op.waitValue.flags & CU_STREAM_WAIT_VALUE_FLUSH), CUDA_ERROR_NOT_SUPPORTED, "remote write flushing is unavailable");
            copy.waitValue.address = op.waitValue.address; copy.waitValue.flags = op.waitValue.flags;
            if (memop_width(op.operation) == 8) copy.waitValue.value64 = op.waitValue.value64;
            else copy.waitValue.value = op.waitValue.value;
        } else if (memop_write(op.operation)) {
            require(!(op.writeValue.flags & ~CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER), CUDA_ERROR_INVALID_VALUE, "invalid memory write flags");
            copy.writeValue.address = op.writeValue.address; copy.writeValue.flags = op.writeValue.flags;
            if (memop_width(op.operation) == 8) copy.writeValue.value64 = op.writeValue.value64;
            else copy.writeValue.value = op.writeValue.value;
        } else if (op.operation == CU_STREAM_MEM_OP_BARRIER) {
            require(op.memoryBarrier.flags <= CU_STREAM_MEMORY_BARRIER_TYPE_GPU, CUDA_ERROR_INVALID_VALUE, "invalid memory barrier scope");
            copy.memoryBarrier.flags = op.memoryBarrier.flags;
        } else if (op.operation == CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES) {
            require(!op.flushRemoteWrites.flags, CUDA_ERROR_INVALID_VALUE, "invalid remote flush flags");
            throw failure(CUDA_ERROR_NOT_SUPPORTED, "remote write flushing is unavailable");
        } else if (op.operation == CU_STREAM_MEM_OP_ATOMIC_REDUCTION) {
            throw failure(CUDA_ERROR_NOT_SUPPORTED, "batch atomic reductions are unavailable");
        } else throw failure(CUDA_ERROR_INVALID_VALUE, "invalid batch memory operation type");
        if (auto address = memop_address(copy))
            require(address % memop_width(copy.operation) == 0, CUDA_ERROR_INVALID_VALUE, "unaligned memory operation address");
    }
    return result;
}
graph_node graph_memops(const CUDA_BATCH_MEM_OP_NODE_PARAMS& p) {
    require(p.ctx, CUDA_ERROR_INVALID_VALUE, "batch memory operation nodes require an explicit context");
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP; node.owner = lookup(p.ctx);
    node.memops = memop_parameters(p.count, p.paramArray, p.flags); return node;
}
// Only addresses/values and wait predicates can change in an executable.
bool memops_compatible(const graph_node& a, const graph_node& b) {
    if (a.memops.size() != b.memops.size()) return false;
    for (size_t i = 0; i < a.memops.size(); ++i) {
        auto& x = a.memops[i]; auto& y = b.memops[i];
        if (x.operation != y.operation) return false;
        if (memop_wait(x.operation)) {
            if ((x.waitValue.flags & ~3u) != (y.waitValue.flags & ~3u)) return false;
        } else if (memop_write(x.operation)) {
            if (x.writeValue.flags != y.writeValue.flags) return false;
        } else if (x.memoryBarrier.flags != y.memoryBarrier.flags) return false;
    }
    return true;
}
struct memop_binding {
    CUstreamBatchMemOpParams params;
    std::shared_ptr<buffer> storage;
    size_t offset = 0;
};
struct memop_task : device_task {
    std::shared_ptr<context> owner;
    std::vector<memop_binding> operations;
    std::shared_ptr<native_queue> queue;
    uint64_t observed = 0;
    size_t position = 0;
    device_work pending; // Retire transfers before freeing their source/destination.
    bool yields_to_exclusive() const override { return !pending; }
    bool query() override {
        if (position == operations.size()) return true;
        auto& op = operations[position]; auto& p = op.params;
        if (pending) { if (!pending->query()) return false; pending.reset(); }
        else if (p.operation == CU_STREAM_MEM_OP_BARRIER) {
            // Previous stream and batch operations have fully completed. Their
            // native transfer/completion barriers already publish device writes.
            std::atomic_thread_fence(std::memory_order_seq_cst); ++position; return position == operations.size();
        } else {
            std::lock_guard lock(owner->execution);
            auto bytes = memop_width(p.operation);
            bool foreign_pool = owner->command_domain && op.storage->pool_session && op.storage->pool_session.get() != owner->native.get();
            if (memop_wait(p.operation)) {
                observed = 0;
                if (foreign_pool) owner->native->read(*op.storage, op.offset, {reinterpret_cast<std::byte*>(&observed), bytes});
                else pending = owner->native->enqueue_read(*op.storage, op.offset, {reinterpret_cast<std::byte*>(&observed), bytes}, queue.get());
            } else {
                if (foreign_pool) owner->native->write(*op.storage, op.offset, {reinterpret_cast<const std::byte*>(&p.writeValue.value64), bytes});
                else pending = owner->native->enqueue_write(*op.storage, op.offset, {reinterpret_cast<const std::byte*>(&p.writeValue.value64), bytes}, queue.get());
            }
            if (pending) return false;
        }
        if (memop_wait(p.operation)) {
            bool wide = memop_width(p.operation) == 8;
            auto mask = wide ? UINT64_MAX : uint64_t(UINT32_MAX);
            auto expected = wide ? p.waitValue.value64 : p.waitValue.value;
            bool satisfied = false;
            switch (p.waitValue.flags & 3u) {
            case CU_STREAM_WAIT_VALUE_GEQ: satisfied = (((observed - expected) & mask) & (wide ? (uint64_t(1) << 63) : (uint64_t(1) << 31))) == 0; break;
            case CU_STREAM_WAIT_VALUE_EQ: satisfied = observed == expected; break;
            case CU_STREAM_WAIT_VALUE_AND: satisfied = (observed & expected) != 0; break;
            case CU_STREAM_WAIT_VALUE_NOR: satisfied = (~(observed | expected) & mask) != 0; break;
            }
            if (!satisfied) return false;
        }
        ++position; return position == operations.size();
    }
};
std::shared_ptr<memop_task> prepare_memops(const graph_node& node, std::vector<std::shared_ptr<host_entry>>& hosts) {
    auto task = std::make_shared<memop_task>(); task->owner = node.owner.lock();
    require(task->owner && !task->owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "memory operation context was destroyed");
    task->operations.reserve(node.memops.size());
    for (auto& op : node.memops) {
        memop_binding binding{op, {}, 0};
        if (memop_wait(op.operation) || memop_write(op.operation)) {
            auto region = resolve(memop_address(op), memop_width(op.operation), task->owner);
            binding.storage = region.value->native; binding.offset = region.offset;
            if (auto host = region.value->host.lock()) hosts.push_back(host);
        }
        task->operations.push_back(std::move(binding));
    }
    return task;
}
void stream_memops(CUstream stream, unsigned count, const CUstreamBatchMemOpParams* params, unsigned flags) {
    auto owner = current(); auto target = stream_lookup(stream, owner); capture_check_stream(target);
    graph_memory_scope scope(bool(target->capture));
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP; node.owner = owner; node.memops = memop_parameters(count, params, flags);
    std::vector<std::shared_ptr<host_entry>> hosts;
    auto task = prepare_memops(node, hosts);
    for (auto& host : hosts) node.host_regions.push_back(host);
    if (capture_node(owner, stream, std::move(node))) return;
    task->queue = stream_native(owner, target);
    for (auto& host : hosts) { std::erase_if(host->work, [](auto& work) { return work.expired(); }); host->work.reserve(host->work.size() + hosts.size()); }
    auto done = submit_stream(owner, stream, {}, {}, 0, [task] { return task; });
    for (auto& host : hosts) host_track(host, done);
}
void stream_memop_value(CUstream stream, CUdeviceptr address, uint64_t value, unsigned flags, CUstreamBatchMemOpType operation) {
    CUstreamBatchMemOpParams p{}; p.operation = operation;
    if (memop_wait(operation)) { p.waitValue.address = address; p.waitValue.value64 = value; p.waitValue.flags = flags; }
    else { p.writeValue.address = address; p.writeValue.value64 = value; p.writeValue.flags = flags; }
    stream_memops(stream, 1, &p, 0);
}
