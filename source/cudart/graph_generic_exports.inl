#include <algorithm>
namespace {
template<class Member> void runtime_graph_union_validate(const cudaGraphNodeParams& p, const Member&) {
    require(std::all_of(std::begin(p.reserved0), std::end(p.reserved0), [](auto v) { return v == 0; }) && !p.reserved2,
            CUDA_ERROR_INVALID_VALUE, "nonzero generic graph reserved fields");
    auto* bytes = reinterpret_cast<const unsigned char*>(&p.reserved1);
    require(std::all_of(bytes + sizeof(Member), bytes + sizeof(p.reserved1), [](auto byte) { return byte == 0; }),
            CUDA_ERROR_INVALID_VALUE, "nonzero unused graph union bytes");
}
struct runtime_generic_node {
    CUgraphNodeParams params{};
    std::vector<CUexternalSemaphore> handles;
    std::vector<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> signals;
    std::vector<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> waits;
    std::vector<CUmemAccessDesc> access;
    explicit runtime_generic_node(const cudaGraphNodeParams& p, CUcontext owner) {
        params.type = static_cast<CUgraphNodeType>(p.type);
        switch (p.type) {
        case cudaGraphNodeTypeEmpty:
            runtime_graph_union_validate(p, char{});
            require(!p.reserved1[0], CUDA_ERROR_INVALID_VALUE, "nonzero empty node union"); break;
        case cudaGraphNodeTypeMemAlloc: {
            runtime_graph_union_validate(p, p.alloc); runtime_graph_allocation converted(p.alloc);
            access = std::move(converted.access); converted.params.accessDescs = access.empty() ? nullptr : access.data();
            std::memcpy(&params.alloc, &converted.params, sizeof(converted.params)); break;
        }
        case cudaGraphNodeTypeMemFree:
            runtime_graph_union_validate(p, p.free); params.free.dptr = reinterpret_cast<uintptr_t>(p.free.dptr); break;
        case cudaGraphNodeTypeKernel: {
            runtime_graph_union_validate(p, p.kernel);
            require(!p.kernel.ctx, CUDA_ERROR_NOT_SUPPORTED, "Runtime execution context handles are unavailable");
            CUDA_KERNEL_NODE_PARAMS kernel{};
            if (p.kernel.functionType == cudaKernelFunctionTypeKernel) {
                kernel.kern = graph_cast<CUkernel>(p.kernel.kern); kernel.ctx = owner;
                kernel.gridDimX = p.kernel.gridDim.x; kernel.gridDimY = p.kernel.gridDim.y; kernel.gridDimZ = p.kernel.gridDim.z;
                kernel.blockDimX = p.kernel.blockDim.x; kernel.blockDimY = p.kernel.blockDim.y; kernel.blockDimZ = p.kernel.blockDim.z;
                kernel.sharedMemBytes = p.kernel.sharedMemBytes; kernel.kernelParams = p.kernel.kernelParams; kernel.extra = p.kernel.extra;
            } else if (p.kernel.functionType == cudaKernelFunctionTypeFunction) {
                kernel.func = graph_cast<CUfunction>(p.kernel.cuFunc);
                kernel.gridDimX = p.kernel.gridDim.x; kernel.gridDimY = p.kernel.gridDim.y; kernel.gridDimZ = p.kernel.gridDim.z;
                kernel.blockDimX = p.kernel.blockDim.x; kernel.blockDimY = p.kernel.blockDim.y; kernel.blockDimZ = p.kernel.blockDim.z;
                kernel.sharedMemBytes = p.kernel.sharedMemBytes; kernel.kernelParams = p.kernel.kernelParams; kernel.extra = p.kernel.extra;
            } else {
                require(p.kernel.functionType == cudaKernelFunctionTypeUnspecified || p.kernel.functionType == cudaKernelFunctionTypeDeviceEntry,
                        p.kernel.functionType == cudaKernelFunctionTypeKernel ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE, "unsupported Runtime kernel function type");
                cudaKernelNodeParams legacy{p.kernel.func, p.kernel.gridDim, p.kernel.blockDim, p.kernel.sharedMemBytes, p.kernel.kernelParams, p.kernel.extra};
                kernel = runtime_graph_kernel(legacy, owner);
            }
            std::memcpy(&params.kernel, &kernel, sizeof(kernel)); break;
        }
        case cudaGraphNodeTypeMemcpy:
            runtime_graph_union_validate(p, p.memcpy);
            require(!p.memcpy.ctx, CUDA_ERROR_NOT_SUPPORTED, "Runtime execution context handles are unavailable");
            params.memcpy.flags = p.memcpy.flags; params.memcpy.reserved = p.memcpy.reserved;
            params.memcpy.copyCtx = owner; params.memcpy.copyParams = runtime_copy_3d(p.memcpy.copyParams); break;
        case cudaGraphNodeTypeMemset:
            runtime_graph_union_validate(p, p.memset);
            require(!p.memset.ctx, CUDA_ERROR_NOT_SUPPORTED, "Runtime execution context handles are unavailable");
            params.memset.dst = reinterpret_cast<uintptr_t>(p.memset.dst); params.memset.pitch = p.memset.pitch;
            params.memset.value = p.memset.value; params.memset.elementSize = p.memset.elementSize;
            params.memset.width = p.memset.width; params.memset.height = p.memset.height; params.memset.ctx = owner; break;
        case cudaGraphNodeTypeHost:
            runtime_graph_union_validate(p, p.host);
            require(!p.host.ctx, CUDA_ERROR_NOT_SUPPORTED, "Runtime execution context handles are unavailable");
            params.host.fn = p.host.fn; params.host.userData = p.host.userData; params.host.syncMode = p.host.syncMode; break;
        case cudaGraphNodeTypeGraph:
            runtime_graph_union_validate(p, p.graph);
            params.graph.graph = graph_cast<CUgraph>(p.graph.graph);
            params.graph.ownership = static_cast<CUgraphChildGraphNodeOwnership>(p.graph.ownership); break;
        case cudaGraphNodeTypeEventRecord:
            runtime_graph_union_validate(p, p.eventRecord); params.eventRecord.event = graph_cast<CUevent>(p.eventRecord.event); break;
        case cudaGraphNodeTypeWaitEvent:
            runtime_graph_union_validate(p, p.eventWait); params.eventWait.event = graph_cast<CUevent>(p.eventWait.event); break;
        case cudaGraphNodeTypeExtSemaphoreSignal: {
            runtime_graph_union_validate(p, p.extSemSignal);
            require(!p.extSemSignal.ctx, CUDA_ERROR_NOT_SUPPORTED, "Runtime execution context handles are unavailable");
            runtime_semaphore_node<CUDA_EXT_SEM_SIGNAL_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> arrays(p.extSemSignal);
            handles = std::move(arrays.handles); signals = std::move(arrays.params);
            params.extSemSignal.extSemArray = handles.data(); params.extSemSignal.paramsArray = signals.data();
            params.extSemSignal.numExtSems = static_cast<unsigned>(handles.size()); params.extSemSignal.ctx = owner; break;
        }
        case cudaGraphNodeTypeExtSemaphoreWait: {
            runtime_graph_union_validate(p, p.extSemWait);
            require(!p.extSemWait.ctx, CUDA_ERROR_NOT_SUPPORTED, "Runtime execution context handles are unavailable");
            runtime_semaphore_node<CUDA_EXT_SEM_WAIT_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> arrays(p.extSemWait);
            handles = std::move(arrays.handles); waits = std::move(arrays.params);
            params.extSemWait.extSemArray = handles.data(); params.extSemWait.paramsArray = waits.data();
            params.extSemWait.numExtSems = static_cast<unsigned>(handles.size()); params.extSemWait.ctx = owner; break;
        }
        default:
            require(p.type >= cudaGraphNodeTypeKernel && p.type <= cudaGraphNodeTypeConditional, CUDA_ERROR_INVALID_VALUE, "invalid generic graph node type");
            throw failure(CUDA_ERROR_NOT_SUPPORTED, "generic Runtime graph node type is not implemented");
        }
    }
};
}
RUNTIME_API(cudaGraphAddNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* data, size_t count, cudaGraphNodeParams* params),
    require(output && params, CUDA_ERROR_INVALID_VALUE, "null generic graph node arguments"); runtime_generic_node converted(*params, runtime_current());
    auto nodes = graph_dependencies(dependencies, count); auto edges = graph_edge_data(data, count); CUgraphNode node{};
    check(cuGraphAddNode(&node, graph_cast<CUgraph>(graph), nodes.data(), edges.data(), count, &converted.params));
    if (params->type == cudaGraphNodeTypeMemAlloc) params->alloc.dptr = reinterpret_cast<void*>(uintptr_t(converted.params.alloc.dptr));
    *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphNodeSetParams, (cudaGraphNode_t node, cudaGraphNodeParams* params),
    require(params, CUDA_ERROR_INVALID_VALUE, "null generic graph node parameters"); runtime_generic_node converted(*params, runtime_current());
    check(cuGraphNodeSetParams(graph_cast<CUgraphNode>(node), &converted.params)))
RUNTIME_API(cudaGraphExecNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, cudaGraphNodeParams* params),
    require(params, CUDA_ERROR_INVALID_VALUE, "null generic graph node parameters"); runtime_generic_node converted(*params, runtime_current());
    check(cuGraphExecNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &converted.params)))
RUNTIME_API(cudaGraphExecGetFlags, (cudaGraphExec_t handle, unsigned long long* flags),
    require(flags, CUDA_ERROR_INVALID_VALUE, "null executable graph flags output"); runtime_current(); cuuint64_t value{};
    check(cuGraphExecGetFlags(graph_cast<CUgraphExec>(handle), &value)); *flags = value)
