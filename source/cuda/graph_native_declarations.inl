std::shared_ptr<native_graph_definition> graph_native_materialize(CUgraph, const std::shared_ptr<context>&);
void graph_native_add(CUgraph, graph_node&);
void graph_native_replace(const graph_node&, graph_node&);
bool graph_has_device_nodes(const std::vector<graph_node>&);
bool graph_has_conditionals(const std::vector<graph_node>&);
bool graph_native_launch(graph_exec&, const std::shared_ptr<context>&, const std::shared_ptr<stream_state>&, CUstream, bool);
void graph_native_instantiate(graph_exec&, CUgraph);
void graph_native_update(graph_exec&, CUgraph, std::vector<graph_node>&);
void graph_native_bind_clone(CUgraph, CUgraph, const std::shared_ptr<native_graph_definition>&);
