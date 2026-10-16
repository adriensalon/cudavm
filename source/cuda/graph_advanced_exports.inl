DRIVER_API(cuGraphAddChildGraphNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, CUgraph child),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null child graph node output"); *output = graph_add_child(graph, dependencies, count, child))
DRIVER_API(cuGraphChildGraphNodeGetGraph, (CUgraphNode node, CUgraph* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null child graph output"); auto& value = graph_node_lookup(node); require(value.type == CU_GRAPH_NODE_TYPE_GRAPH, CUDA_ERROR_INVALID_VALUE, "not a child graph node"); *output = value.child)
DRIVER_API(cuGraphExecChildGraphNodeSetParams, (CUgraphExec handle, CUgraphNode node, CUgraph child), driver_ready(); graph_exec_child_replace(handle, node, child))
DRIVER_API(cuGraphExecUpdate, (CUgraphExec handle, CUgraph graph, CUgraphExecUpdateResultInfo* info), driver_ready(); graph_update(handle, graph, info))
#undef cuGraphExecUpdate
DRIVER_API(cuGraphExecUpdate, (CUgraphExec handle, CUgraph graph, CUgraphNode* error, CUgraphExecUpdateResult* result),
    driver_ready(); require(result, CUDA_ERROR_INVALID_VALUE, "null graph update result"); CUgraphExecUpdateResultInfo info{};
    try { graph_update(handle, graph, &info); } catch (...) { if (error) *error = info.errorNode; *result = info.result; throw; }
    if (error) *error = info.errorNode; *result = info.result)
DRIVER_API(cuGraphAddEventRecordNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, CUevent event),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null event node output"); *output = graph_add(graph, dependencies, count, graph_event(event, CU_GRAPH_NODE_TYPE_EVENT_RECORD)))
DRIVER_API(cuGraphEventRecordNodeGetEvent, (CUgraphNode node, CUevent* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph event output"); auto& value = graph_node_lookup(node); require(value.type == CU_GRAPH_NODE_TYPE_EVENT_RECORD, CUDA_ERROR_INVALID_VALUE, "wrong graph event node type"); *output = value.event)
DRIVER_API(cuGraphEventRecordNodeSetEvent, (CUgraphNode node, CUevent event), driver_ready(); graph_replace(graph_node_lookup(node), graph_event(event, CU_GRAPH_NODE_TYPE_EVENT_RECORD)))
DRIVER_API(cuGraphExecEventRecordNodeSetEvent, (CUgraphExec handle, CUgraphNode node, CUevent event), driver_ready(); graph_exec_replace(handle, node, graph_event(event, CU_GRAPH_NODE_TYPE_EVENT_RECORD)))
DRIVER_API(cuGraphAddEventWaitNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, CUevent event),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null event node output"); *output = graph_add(graph, dependencies, count, graph_event(event, CU_GRAPH_NODE_TYPE_WAIT_EVENT)))
DRIVER_API(cuGraphEventWaitNodeGetEvent, (CUgraphNode node, CUevent* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph event output"); auto& value = graph_node_lookup(node); require(value.type == CU_GRAPH_NODE_TYPE_WAIT_EVENT, CUDA_ERROR_INVALID_VALUE, "wrong graph event node type"); *output = value.event)
DRIVER_API(cuGraphEventWaitNodeSetEvent, (CUgraphNode node, CUevent event), driver_ready(); graph_replace(graph_node_lookup(node), graph_event(event, CU_GRAPH_NODE_TYPE_WAIT_EVENT)))
DRIVER_API(cuGraphExecEventWaitNodeSetEvent, (CUgraphExec handle, CUgraphNode node, CUevent event), driver_ready(); graph_exec_replace(handle, node, graph_event(event, CU_GRAPH_NODE_TYPE_WAIT_EVENT)))
