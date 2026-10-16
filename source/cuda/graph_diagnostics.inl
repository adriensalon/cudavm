std::string graph_dot_escape(const std::string& text) {
    std::string result;
    for (unsigned char byte : text) {
        if (byte == '\\' || byte == '"') { result += '\\'; result += static_cast<char>(byte); }
        else if (byte == '\n') result += "\\n";
        else if (byte == '\r') result += "\\r";
        else if (byte < 32 || byte == 127) result += '?';
        else result += static_cast<char>(byte);
    }
    return result;
}
template<class T> std::string graph_dot_handle(T value) {
    std::ostringstream out; out << "0x" << std::hex;
    if constexpr (std::is_pointer_v<T>) out << reinterpret_cast<uintptr_t>(value);
    else out << value;
    return out.str();
}
const char* graph_dot_type(CUgraphNodeType type) {
    switch (type) {
    case CU_GRAPH_NODE_TYPE_EMPTY: return "Empty";
    case CU_GRAPH_NODE_TYPE_KERNEL: return "Kernel";
    case CU_GRAPH_NODE_TYPE_MEMCPY: return "Memcpy";
    case CU_GRAPH_NODE_TYPE_MEMSET: return "Memset";
    case CU_GRAPH_NODE_TYPE_HOST: return "Host";
    case CU_GRAPH_NODE_TYPE_GRAPH: return "ChildGraph";
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return "EventRecord";
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return "EventWait";
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: return "ExtSemaphoreSignal";
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: return "ExtSemaphoreWait";
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC: return "MemAlloc";
    case CU_GRAPH_NODE_TYPE_MEM_FREE: return "MemFree";
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: return "BatchMemOp";
    default: return "Unsupported";
    }
}
const char* graph_dot_memory(CUmemorytype type) {
    switch (type) {
    case CU_MEMORYTYPE_HOST: return "Host";
    case CU_MEMORYTYPE_DEVICE: return "Device";
    case CU_MEMORYTYPE_ARRAY: return "Array";
    case CU_MEMORYTYPE_UNIFIED: return "Unified";
    default: return "Unknown";
    }
}
std::string graph_dot_node_label(const graph_node& node, unsigned flags) {
    std::ostringstream label; label << "nodeId=" << node.local_id << "\n" << graph_dot_type(node.type);
    bool runtime = (flags & CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES) != 0;
    auto enabled = [&](unsigned flag) { return (flags & (CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE | flag)) != 0; };
    if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES)) label << "\nnode=" << graph_dot_handle(node.handle);
    switch (node.type) {
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP:
        label << "\noperations=" << node.memops.size();
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE)) for (auto& op : node.memops)
            label << "\nop=" << op.operation << " address=" << graph_dot_handle(memop_address(op));
        break;
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC:
        label << "\naddress=" << graph_dot_handle(node.memory->address) << " bytes=" << node.memory->params.bytesize; break;
    case CU_GRAPH_NODE_TYPE_MEM_FREE:
        label << "\naddress=" << graph_dot_handle(node.memory->address); break;
    case CU_GRAPH_NODE_TYPE_KERNEL:
        label << "\n" << node.kernel_name;
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES) && !enabled(CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS)) label << "\nfunc=" << graph_dot_handle(node.function);
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS)) {
            label << "\n" << (runtime ? "cudaKernelNodeParams" : "CUDA_KERNEL_NODE_PARAMS")
                  << "\nfunc=" << graph_dot_handle(node.function)
                  << "\ngrid=" << node.shape.grid[0] << ',' << node.shape.grid[1] << ',' << node.shape.grid[2]
                  << " block=" << node.shape.block[0] << ',' << node.shape.block[1] << ',' << node.shape.block[2]
                  << " sharedMemBytes=" << node.shape.shared_bytes << "\narguments=";
            for (auto byte : node.arguments) label << std::hex << std::setfill('0') << std::setw(2) << std::to_integer<unsigned>(byte);
            label << std::dec << "\nctx=" << graph_dot_handle(node.parameter_context);
        }
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES)) {
            auto owner = node.owner.lock(); label << "\npriority=" << (owner ? stream_priority(node.priority, owner) : node.priority);
        }
        break;
    case CU_GRAPH_NODE_TYPE_MEMCPY:
        label << "\n" << graph_dot_memory(node.copy.srcMemoryType) << " -> " << graph_dot_memory(node.copy.dstMemoryType);
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS)) {
            auto& p = node.copy;
            label << "\n" << (runtime ? "cudaMemcpy3DParms" : "CUDA_MEMCPY3D")
                  << "\nextent=" << p.WidthInBytes << ',' << p.Height << ',' << p.Depth
                  << "\nsrc=" << graph_dot_handle(p.srcMemoryType == CU_MEMORYTYPE_HOST ? reinterpret_cast<uintptr_t>(p.srcHost) : p.srcDevice)
                  << " pitch=" << p.srcPitch << " height=" << p.srcHeight
                  << " offset=" << p.srcXInBytes << ',' << p.srcY << ',' << p.srcZ
                  << "\ndst=" << graph_dot_handle(p.dstMemoryType == CU_MEMORYTYPE_HOST ? reinterpret_cast<uintptr_t>(p.dstHost) : p.dstDevice)
                  << " pitch=" << p.dstPitch << " height=" << p.dstHeight
                  << " offset=" << p.dstXInBytes << ',' << p.dstY << ',' << p.dstZ;
            if (node.peer_copy) {
                auto src = node.copy_source.lock(), dst = node.copy_destination.lock();
                label << "\npeer srcCtx=" << graph_dot_handle(src ? src->handle : nullptr)
                      << " dstCtx=" << graph_dot_handle(dst ? dst->handle : nullptr);
            } else label << "\nctx=" << graph_dot_handle(node.parameter_context);
        }
        break;
    case CU_GRAPH_NODE_TYPE_MEMSET:
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS)) {
            auto& p = node.fill;
            label << "\n" << (runtime ? "cudaMemsetParams" : "CUDA_MEMSET_NODE_PARAMS")
                  << "\ndst=" << graph_dot_handle(p.dst) << " pitch=" << p.pitch << " value=" << p.value
                  << " elementSize=" << p.elementSize << " width=" << p.width << " height=" << p.height;
        }
        break;
    case CU_GRAPH_NODE_TYPE_HOST:
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS))
            label << "\n" << (runtime ? "cudaHostNodeParams" : "CUDA_HOST_NODE_PARAMS")
                  << "\nfn=" << graph_dot_handle(node.host.fn) << " userData=" << graph_dot_handle(node.host.userData)
                  << " ctx=" << graph_dot_handle(node.host_context);
        break;
    case CU_GRAPH_NODE_TYPE_GRAPH:
        label << "\nchildGraphId=" << graph_lookup(node.child).id; break;
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT:
        if (enabled(CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS)) label << "\nevent=" << graph_dot_handle(node.event);
        break;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL:
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: {
        bool signal = node.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL;
        if (enabled(signal ? CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS : CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS)) {
            label << "\n" << (signal ? (runtime ? "cudaExternalSemaphoreSignalNodeParams" : "CUDA_EXT_SEM_SIGNAL_NODE_PARAMS")
                                    : (runtime ? "cudaExternalSemaphoreWaitNodeParams" : "CUDA_EXT_SEM_WAIT_NODE_PARAMS"))
                  << "\nnumExtSems=" << node.semaphores.size();
            for (size_t i = 0; i < node.semaphores.size(); ++i) {
                auto value = signal ? node.signals[i].params.fence.value : node.waits[i].params.fence.value;
                auto operation_flags = signal ? node.signals[i].flags : node.waits[i].flags;
                label << "\nsem[" << i << "]=" << graph_dot_handle(node.semaphores[i]) << " fence=" << value << " flags=" << operation_flags;
            }
        }
        break;
    }
    default: break;
    }
    return label.str();
}
std::string graph_dot(CUgraph handle, unsigned flags) {
    require(!(flags & ~0xffffu), CUDA_ERROR_INVALID_VALUE, "unknown graph DOT flags");
    graph_lookup(handle);
    std::ostringstream out; out << "digraph cudavm {\n";
    std::function<void(CUgraph)> append = [&](CUgraph graph) {
        auto& value = graph_lookup(graph);
        out << "subgraph cluster_graph_" << value.id << " {\nlabel=\"graphId=" << value.id;
        std::vector<unsigned> executables;
        for (auto& [exec_handle, executable] : global().graph_execs) if (executable.source_id == value.id) executables.push_back(executable.id);
        std::sort(executables.begin(), executables.end());
        for (auto id : executables) out << "\\ngraphExecId=" << id;
        out << "\";\n";
        auto name = [&](CUgraphNode node) { return "n_" + std::to_string(value.id) + "_" + std::to_string(graph_node_lookup(node).local_id); };
        for (auto& node : value.nodes) {
            out << name(node.handle) << " [label=\"" << graph_dot_escape(graph_dot_node_label(node, flags)) << "\"];\n";
            if (node.child) append(node.child);
        }
        for (auto& node : value.nodes) for (size_t i = 0; i < node.dependencies.size(); ++i) {
            out << name(node.dependencies[i]) << " -> " << name(node.handle);
            if (flags & (CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE | CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO)) {
                const auto& edge = node.dependency_edges[i];
                out << " [label=\"edge=" << i << " from_port=" << unsigned(edge.from_port) << " to_port=" << unsigned(edge.to_port) << " type=" << unsigned(edge.type) << "\"]";
            }
            out << ";\n";
        }
        out << "}\n";
    };
    append(handle); out << "}\n"; return out.str();
}
void graph_dot_print(CUgraph graph, const char* path, unsigned flags) {
    require(path && *path, CUDA_ERROR_INVALID_VALUE, "null or empty graph DOT path");
    auto description = graph_dot(graph, flags);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    require(file.is_open(), CUDA_ERROR_OPERATING_SYSTEM, "could not open graph DOT output");
    file.write(description.data(), static_cast<std::streamsize>(description.size())); file.close();
    require(!file.fail(), CUDA_ERROR_OPERATING_SYSTEM, "could not write graph DOT output");
}
