RUNTIME_API(cudaGraphAddChildGraphNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, cudaGraph_t child),
    require(output, CUDA_ERROR_INVALID_VALUE, "null child node output"); runtime_current(); auto nodes = graph_dependencies(dependencies, count); CUgraphNode node{};
    check(cuGraphAddChildGraphNode(&node, graph_cast<CUgraph>(graph), nodes.data(), count, graph_cast<CUgraph>(child))); *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphChildGraphNodeGetGraph, (cudaGraphNode_t node, cudaGraph_t* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null child graph output"); runtime_current(); CUgraph child{};
    check(cuGraphChildGraphNodeGetGraph(graph_cast<CUgraphNode>(node), &child)); *output = graph_cast<cudaGraph_t>(child))
RUNTIME_API(cudaGraphExecChildGraphNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, cudaGraph_t child),
    runtime_current(); check(cuGraphExecChildGraphNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), graph_cast<CUgraph>(child))))
RUNTIME_API(cudaGraphExecUpdate, (cudaGraphExec_t handle, cudaGraph_t graph, cudaGraphExecUpdateResultInfo* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph update result"); runtime_current(); CUgraphExecUpdateResultInfo info{};
    auto result = cuGraphExecUpdate(graph_cast<CUgraphExec>(handle), graph_cast<CUgraph>(graph), &info);
    output->result = static_cast<cudaGraphExecUpdateResult>(info.result); output->errorNode = graph_cast<cudaGraphNode_t>(info.errorNode); output->errorFromNode = graph_cast<cudaGraphNode_t>(info.errorFromNode); check(result))
RUNTIME_API(cudaGraphAddEventRecordNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, cudaEvent_t event),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph event node output"); runtime_current(); auto nodes = graph_dependencies(dependencies, count); CUgraphNode node{};
    check(cuGraphAddEventRecordNode(&node, graph_cast<CUgraph>(graph), nodes.data(), count, graph_cast<CUevent>(event))); *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphEventRecordNodeGetEvent, (cudaGraphNode_t node, cudaEvent_t* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph event output"); runtime_current(); CUevent event{};
    check(cuGraphEventRecordNodeGetEvent(graph_cast<CUgraphNode>(node), &event)); *output = graph_cast<cudaEvent_t>(event))
RUNTIME_API(cudaGraphEventRecordNodeSetEvent, (cudaGraphNode_t node, cudaEvent_t event), runtime_current(); check(cuGraphEventRecordNodeSetEvent(graph_cast<CUgraphNode>(node), graph_cast<CUevent>(event))))
RUNTIME_API(cudaGraphExecEventRecordNodeSetEvent, (cudaGraphExec_t handle, cudaGraphNode_t node, cudaEvent_t event),
    runtime_current(); check(cuGraphExecEventRecordNodeSetEvent(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), graph_cast<CUevent>(event))))
RUNTIME_API(cudaGraphAddEventWaitNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, cudaEvent_t event),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph event node output"); runtime_current(); auto nodes = graph_dependencies(dependencies, count); CUgraphNode node{};
    check(cuGraphAddEventWaitNode(&node, graph_cast<CUgraph>(graph), nodes.data(), count, graph_cast<CUevent>(event))); *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphEventWaitNodeGetEvent, (cudaGraphNode_t node, cudaEvent_t* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph event output"); runtime_current(); CUevent event{};
    check(cuGraphEventWaitNodeGetEvent(graph_cast<CUgraphNode>(node), &event)); *output = graph_cast<cudaEvent_t>(event))
RUNTIME_API(cudaGraphEventWaitNodeSetEvent, (cudaGraphNode_t node, cudaEvent_t event), runtime_current(); check(cuGraphEventWaitNodeSetEvent(graph_cast<CUgraphNode>(node), graph_cast<CUevent>(event))))
RUNTIME_API(cudaGraphExecEventWaitNodeSetEvent, (cudaGraphExec_t handle, cudaGraphNode_t node, cudaEvent_t event),
    runtime_current(); check(cuGraphExecEventWaitNodeSetEvent(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), graph_cast<CUevent>(event))))
