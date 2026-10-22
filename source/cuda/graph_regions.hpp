#pragma once
#include "backend.hpp"
#include "command_queue.hpp"

namespace cudavm {
// Only contract chains whose intermediate completions have a single consumer.
// Every other dependency remains a scheduler boundary, including events, host
// work, foreign contexts, disabled nodes and graph memory transitions.
template<class Compile> void lower_graph_regions(std::vector<command_queue::submission>& replay,
        const std::vector<native_graph_command>& recorders, Compile compile, std::shared_ptr<native_queue> queue) {
    std::map<command_completion*, size_t> consumers;
    for (auto& work : replay) for (auto& dependency : work.dependencies) ++consumers[dependency.get()];
    std::map<command_completion*, completion> replacements;
    std::vector<command_queue::submission> lowered;
    lowered.reserve(replay.size());
    for (size_t first = 0; first < replay.size();) {
        size_t end = first + 1;
        if (recorders[first]) while (end < replay.size() && recorders[end] &&
                replay[end].dependencies.size() == 1 && replay[end].dependencies[0] == replay[end - 1].done &&
                consumers[replay[end - 1].done.get()] == 1 && replay[end].priority == replay[first].priority &&
                replay[end].domain == replay[first].domain && replay[end].exclusive == replay[first].exclusive) ++end;
        if (end > first + 1) {
            std::vector<native_graph_command> commands;
            commands.reserve(end - first);
            for (size_t i = first; i < end; ++i)
                commands.push_back(recorders[i]);
            auto graph = compile(commands);
            if (graph) {
                auto work = replay[first];
                work.done = replay[end - 1].done;
                work.action = {};
                work.start = [graph, commands = std::move(commands), queue] { return graph->enqueue(queue.get()); };
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
}
}
