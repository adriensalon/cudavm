#pragma once
#include "backend.hpp"
#include "command_queue.hpp"

namespace cudavm {
// Keys describe resolved resources, not just CUDA addresses: a freed allocation
// can be replaced at the same address. The retained commands prevent identity
// reuse while an entry is cached or an accepted replay is still in flight.
using native_graph_key = std::vector<uint64_t>;
struct native_graph_region {
    size_t end = 0;
    command_domain* domain = nullptr;
    bool exclusive = false;
    std::vector<native_graph_key> keys;
    std::vector<native_graph_node> nodes;
    // Destroy native descriptors before releasing their recorded resources.
    std::shared_ptr<native_graph> graph;
};
using native_graph_cache = std::map<size_t, std::shared_ptr<native_graph_region>>;

// Contract contiguous, topologically ordered device regions. Host/event work,
// foreign contexts, disabled nodes and memory transitions remain boundaries.
// All incoming dependencies gate the region; every outgoing dependency waits
// for its native join. Dependency ports inside the region remain native edges.
template<class Compile> void lower_graph_regions(std::vector<command_queue::submission>& replay,
        const std::vector<native_graph_command>& recorders, Compile compile, std::shared_ptr<native_queue> queue,
        native_graph_cache* cache = nullptr, std::span<const native_graph_key> keys = {},
        std::span<const std::vector<CUgraphEdgeData>> edges = {}) {
    require(recorders.size() == replay.size(), CUDA_ERROR_INVALID_VALUE, "missing native graph recorders");
    require(!cache || keys.size() == recorders.size(), CUDA_ERROR_INVALID_VALUE, "missing native graph resource keys");
    require(edges.empty() || edges.size() == replay.size(), CUDA_ERROR_INVALID_VALUE, "missing native graph edges");
    for (size_t i = 0; i < edges.size(); ++i)
        require(edges[i].size() == replay[i].dependencies.size(), CUDA_ERROR_INVALID_VALUE, "missing native graph dependency metadata");
    std::map<command_completion*, completion> replacements;
    std::vector<command_queue::submission> lowered;
    native_graph_cache next_cache;
    lowered.reserve(replay.size());
    for (size_t first = 0; first < replay.size();) {
        size_t end = first + 1;
        if (recorders[first]) while (end < replay.size() && recorders[end] &&
                replay[end].domain == replay[first].domain && replay[end].exclusive == replay[first].exclusive) ++end;
        if (end > first + 1) {
            std::map<command_completion*, size_t> indices;
            for (size_t i = first; i < end; ++i) indices.emplace(replay[i].done.get(), i - first);
            std::vector<native_graph_node> nodes;
            std::vector<completion> incoming;
            int priority = replay[first].priority;
            for (size_t i = first; i < end; ++i) {
                native_graph_node node{recorders[i], {}, replay[i].priority};
                priority = std::min(priority, node.priority);
                for (size_t j = 0; j < replay[i].dependencies.size(); ++j) {
                    auto& dependency = replay[i].dependencies[j];
                    if (auto found = indices.find(dependency.get()); found != indices.end()) {
                        require(found->second < i - first, CUDA_ERROR_INVALID_VALUE, "native graph is not topologically ordered");
                        node.dependencies.push_back({found->second, edges.empty() ? CUgraphEdgeData{} : edges[i][j]});
                    } else if (std::find(incoming.begin(), incoming.end(), dependency) == incoming.end()) incoming.push_back(dependency);
                }
                nodes.push_back(std::move(node));
            }
            std::shared_ptr<native_graph_region> region;
            if (cache) if (auto found = cache->find(first); found != cache->end()) {
                auto& entry = found->second;
                if (entry->end == end &&
                    entry->domain == replay[first].domain.get() && entry->exclusive == replay[first].exclusive &&
                    std::equal(entry->keys.begin(), entry->keys.end(), keys.begin() + first, keys.begin() + end) &&
                    std::equal(entry->nodes.begin(), entry->nodes.end(), nodes.begin(), nodes.end(), [](const auto& a, const auto& b) {
                        return a.priority == b.priority && a.dependencies == b.dependencies;
                    }))
                    region = entry;
            }
            if (!region) {
                region = std::make_shared<native_graph_region>();
                region->end = end;
                region->domain = replay[first].domain.get(); region->exclusive = replay[first].exclusive;
                if (cache) region->keys.assign(keys.begin() + first, keys.begin() + end);
                region->nodes = std::move(nodes);
                region->graph = compile(region->nodes);
            }
            // An unavailable recording service is also stable for this key.
            // Exceptions leave both the cache and the replay batch unchanged.
            if (cache) next_cache.emplace(first, region);
            if (region->graph) {
                auto work = replay[first];
                work.done = replay[end - 1].done;
                work.dependencies = std::move(incoming);
                work.priority = priority;
                work.action = {};
                work.start = [region, queue] { return region->graph->enqueue(queue.get()); };
                for (size_t i = first; i < end; ++i) replacements[replay[i].done.get()] = work.done;
                lowered.push_back(std::move(work)); first = end; continue;
            }
            for (size_t i = first; i < end; ++i) lowered.push_back(replay[i]);
            first = end; continue;
        }
        lowered.push_back(replay[first]); ++first;
    }
    for (auto& work : lowered) {
        std::vector<completion> dependencies;
        for (auto dependency : work.dependencies) {
            if (auto found = replacements.find(dependency.get()); found != replacements.end()) dependency = found->second;
            if (std::find(dependencies.begin(), dependencies.end(), dependency) == dependencies.end()) dependencies.push_back(std::move(dependency));
        }
        work.dependencies = std::move(dependencies);
    }
    replay.swap(lowered);
    if (cache) cache->swap(next_cache);
}
}
