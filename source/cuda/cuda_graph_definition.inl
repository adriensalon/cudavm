// Included beside the CUDA queue and context implementations.
struct cuda_graph_definition;
struct cuda_graph_executable : native_graph_executable {
    std::shared_ptr<cuda_context> context;
    std::shared_ptr<native_graph_definition> definition;
    CUgraphExec value = nullptr;
    cuda_graph_executable(std::shared_ptr<cuda_context> owner, std::shared_ptr<native_graph_definition> graph)
        : context(std::move(owner)), definition(std::move(graph)) {}
    ~cuda_graph_executable() {
        if (value && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->driver.get<decltype(&cuGraphExecDestroy)>("cuGraphExecDestroy")(value);
            CUcontext previous{}; context->api->pop(&previous);
        }
    }
    device_work enqueue(native_queue* queue) override {
        return cuda_enqueue(context, queue, [&](CUstream stream) {
            check(context->api->driver.get<decltype(&cuGraphLaunch)>("cuGraphLaunch")(value, stream), "cuGraphLaunch device state");
        });
    }
    device_work upload(native_queue* queue) override {
        return cuda_enqueue(context, queue, [&](CUstream stream) {
            check(context->api->driver.get<decltype(&cuGraphUpload)>("cuGraphUpload")(value, stream), "cuGraphUpload device state");
        });
    }
    void set_params(CUgraphNode node, CUgraphNodeParams& params) override;
    void set_enabled(CUgraphNode node, bool enabled) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuGraphNodeSetEnabled)>("cuGraphNodeSetEnabled")(value, node, enabled), "cuGraphNodeSetEnabled device state");
    }
};
// Context fields always refer to the real CUDA context at this boundary.
CUgraphNodeParams cuda_graph_parameters(CUgraphNodeParams params, CUcontext context) {
    switch (params.type) {
    case CU_GRAPH_NODE_TYPE_KERNEL: params.kernel.ctx = context; break;
    case CU_GRAPH_NODE_TYPE_MEMCPY: params.memcpy.copyCtx = context; break;
    case CU_GRAPH_NODE_TYPE_MEMSET: params.memset.ctx = context; break;
    case CU_GRAPH_NODE_TYPE_CONDITIONAL: params.conditional.ctx = context; break;
    case CU_GRAPH_NODE_TYPE_HOST: params.host.ctx = context; break;
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: params.memOp.ctx = context; break;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: params.extSemSignal.ctx = context; break;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: params.extSemWait.ctx = context; break;
    default: break;
    }
    return params;
}
void cuda_graph_executable::set_params(CUgraphNode node, CUgraphNodeParams& params) {
    cuda_context_guard guard(*context); auto converted = cuda_graph_parameters(params, context->value);
    check(context->api->driver.get<decltype(&cuGraphExecNodeSetParams)>("cuGraphExecNodeSetParams")(value, node, &converted), "cuGraphExecNodeSetParams device state");
}
struct cuda_graph_definition : native_graph_definition, std::enable_shared_from_this<cuda_graph_definition> {
    std::shared_ptr<cuda_context> context;
    std::shared_ptr<native_graph_definition> parent;
    CUgraph value = nullptr;
    explicit cuda_graph_definition(std::shared_ptr<cuda_context> owner) : context(std::move(owner)) {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuGraphCreate)>("cuGraphCreate")(&value, 0), "cuGraphCreate device state");
    }
    cuda_graph_definition(std::shared_ptr<cuda_context> owner, CUgraph graph, std::shared_ptr<native_graph_definition> root = {})
        : context(std::move(owner)), parent(std::move(root)), value(graph) {}
    ~cuda_graph_definition() {
        if (!parent && value && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->driver.get<decltype(&cuGraphDestroy)>("cuGraphDestroy")(value);
            CUcontext previous{}; context->api->pop(&previous);
        }
    }
    CUgraph handle() const override { return value; }
    CUgraphNode add(CUgraphNodeParams& params, std::span<const CUgraphNode> dependencies, std::span<const CUgraphEdgeData> data) override {
        cuda_context_guard guard(*context); auto converted = cuda_graph_parameters(params, context->value); CUgraphNode node{};
        auto& driver = context->api->driver;
        // v2 adds edge data; v3 changes child ownership and other 13.4 fields.
        auto add = reinterpret_cast<decltype(&cuGraphAddNode)>(driver.find("cuGraphAddNode_v3"));
        if (!add) add = driver.get<decltype(add)>("cuGraphAddNode_v2");
        check(add(&node, value, dependencies.data(), data.data(), dependencies.size(), &converted), "cuGraphAddNode device state");
        if (params.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) params.conditional.phGraph_out = converted.conditional.phGraph_out;
        return node;
    }
    void remove(CUgraphNode node) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuGraphDestroyNode)>("cuGraphDestroyNode")(node), "cuGraphDestroyNode device state");
    }
    void edges(std::span<const CUgraphNode> from, std::span<const CUgraphNode> to, std::span<const CUgraphEdgeData> data, bool adding) override {
        cuda_context_guard guard(*context); auto& driver = context->api->driver;
        auto change = adding ? driver.get<decltype(&cuGraphAddDependencies)>("cuGraphAddDependencies_v2") :
            driver.get<decltype(&cuGraphRemoveDependencies)>("cuGraphRemoveDependencies_v2");
        check(change(value, from.data(), to.data(), data.data(), from.size()), "native graph dependency edit");
    }
    void set_params(CUgraphNode node, CUgraphNodeParams& params) override {
        cuda_context_guard guard(*context); auto converted = cuda_graph_parameters(params, context->value);
        check(context->api->driver.get<decltype(&cuGraphNodeSetParams)>("cuGraphNodeSetParams")(node, &converted), "cuGraphNodeSetParams device state");
    }
    CUkernelNodeAttrValue attribute(CUgraphNode node, CUkernelNodeAttrID id, const CUkernelNodeAttrValue* input = nullptr) override {
        cuda_context_guard guard(*context); auto& driver = context->api->driver;
        if (input) check(driver.get<decltype(&cuGraphKernelNodeSetAttribute)>("cuGraphKernelNodeSetAttribute")(node, id, input), "cuGraphKernelNodeSetAttribute device state");
        CUkernelNodeAttrValue result{};
        check(driver.get<decltype(&cuGraphKernelNodeGetAttribute)>("cuGraphKernelNodeGetAttribute")(node, id, &result), "cuGraphKernelNodeGetAttribute device state");
        return result;
    }
    CUgraphConditionalHandle condition(unsigned initial, unsigned flags) override {
        cuda_context_guard guard(*context); CUgraphConditionalHandle result{};
        check(context->api->driver.get<decltype(&cuGraphConditionalHandleCreate)>("cuGraphConditionalHandleCreate")(&result, value, context->value, initial, flags), "cuGraphConditionalHandleCreate");
        return result;
    }
    void copy_attributes(CUgraphNode destination, CUgraphNode source) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuGraphKernelNodeCopyAttributes)>("cuGraphKernelNodeCopyAttributes")(destination, source), "cuGraphKernelNodeCopyAttributes device state");
    }
    std::shared_ptr<native_graph_definition> body(CUgraph graph) override {
        return std::make_shared<cuda_graph_definition>(context, graph, shared_from_this());
    }
    std::shared_ptr<native_graph_definition> child(CUgraphNode node) override {
        cuda_context_guard guard(*context); CUgraph graph{};
        check(context->api->driver.get<decltype(&cuGraphChildGraphNodeGetGraph)>("cuGraphChildGraphNodeGetGraph")(node, &graph), "cuGraphChildGraphNodeGetGraph device state");
        return body(graph);
    }
    std::shared_ptr<native_graph_definition> clone() override {
        cuda_context_guard guard(*context); auto result = std::make_shared<cuda_graph_definition>(context, CUgraph{});
        check(context->api->driver.get<decltype(&cuGraphClone)>("cuGraphClone")(&result->value, value), "cuGraphClone device state");
        return result;
    }
    CUgraphNode find_in_clone(CUgraphNode original) override {
        cuda_context_guard guard(*context); CUgraphNode result{};
        check(context->api->driver.get<decltype(&cuGraphNodeFindInClone)>("cuGraphNodeFindInClone")(&result, original, value), "cuGraphNodeFindInClone device state");
        return result;
    }
    std::shared_ptr<native_graph_executable> instantiate(unsigned long long flags) override {
        cuda_context_guard guard(*context); auto result = std::make_shared<cuda_graph_executable>(context, shared_from_this());
        check(context->api->driver.get<decltype(&cuGraphInstantiateWithFlags)>("cuGraphInstantiateWithFlags")(&result->value, value, flags), "cuGraphInstantiate device state");
        return result;
    }
};
