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
    int priority = 0;
    command_domain* domain = nullptr;
    bool exclusive = false;
    std::vector<native_graph_key> keys;
    std::vector<native_graph_command> commands;
    // Destroy native descriptors before releasing their recorded resources.
    std::shared_ptr<native_graph> graph;
};
using native_graph_cache = std::map<size_t, std::shared_ptr<native_graph_region>>;

// Only contract chains whose intermediate completions have a single consumer.
// Every other dependency remains a scheduler boundary, including events, host
// work, foreign contexts, disabled nodes and graph memory transitions.
template<class Compile> void lower_graph_regions(std::vector<command_queue::submission>& replay,
        const std::vector<native_graph_command>& recorders, Compile compile, std::shared_ptr<native_queue> queue,
        native_graph_cache* cache = nullptr, std::span<const native_graph_key> keys = {}) {
    require(recorders.size() == replay.size(), CUDA_ERROR_INVALID_VALUE, "missing native graph recorders");
    require(!cache || keys.size() == recorders.size(), CUDA_ERROR_INVALID_VALUE, "missing native graph resource keys");
    std::map<command_completion*, size_t> consumers;
    for (auto& work : replay) for (auto& dependency : work.dependencies) ++consumers[dependency.get()];
    std::map<command_completion*, completion> replacements;
    std::vector<command_queue::submission> lowered;
    native_graph_cache next_cache;
    lowered.reserve(replay.size());
    for (size_t first = 0; first < replay.size();) {
        size_t end = first + 1;
        if (recorders[first]) while (end < replay.size() && recorders[end] &&
                replay[end].dependencies.size() == 1 && replay[end].dependencies[0] == replay[end - 1].done &&
                consumers[replay[end - 1].done.get()] == 1 && replay[end].priority == replay[first].priority &&
                replay[end].domain == replay[first].domain && replay[end].exclusive == replay[first].exclusive) ++end;
        if (end > first + 1) {
            std::shared_ptr<native_graph_region> region;
            if (cache) if (auto found = cache->find(first); found != cache->end()) {
                auto& entry = found->second;
                if (entry->end == end && entry->priority == replay[first].priority &&
                    entry->domain == replay[first].domain.get() && entry->exclusive == replay[first].exclusive &&
                    std::equal(entry->keys.begin(), entry->keys.end(), keys.begin() + first, keys.begin() + end))
                    region = entry;
            }
            if (!region) {
                region = std::make_shared<native_graph_region>();
                region->end = end; region->priority = replay[first].priority;
                region->domain = replay[first].domain.get(); region->exclusive = replay[first].exclusive;
                if (cache) region->keys.assign(keys.begin() + first, keys.begin() + end);
                region->commands.assign(recorders.begin() + first, recorders.begin() + end);
                region->graph = compile(region->commands);
            }
            // An unavailable recording service is also stable for this key.
            // Exceptions leave both the cache and the replay batch unchanged.
            if (cache) next_cache.emplace(first, region);
            if (region->graph) {
                auto work = replay[first];
                work.done = replay[end - 1].done;
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
    for (auto& work : lowered) for (auto& dependency : work.dependencies)
        if (auto found = replacements.find(dependency.get()); found != replacements.end()) dependency = found->second;
    replay.swap(lowered);
    if (cache) cache->swap(next_cache);
}
}
