#pragma once
#include "backend.hpp"
#include "../core/shader_packet.hpp"
#include <unordered_map>

namespace cudavm {
struct shader_staging_range { buffer* native; size_t first, bytes, offset; };
struct shader_memory_plan {
    std::vector<packet_allocation> records;
    std::vector<buffer*> direct;
    std::vector<shader_staging_range> staged;
    size_t arena_bytes = packet_arena_header_bytes;
};
inline shader_memory_plan plan_shader_memory(session& owner, std::span<const allocation_binding> bindings) {
    struct backing_range {
        buffer* native;
        size_t first, end, offset = 0;
        uint32_t binding = 0;
        bool descriptor = false;
        std::vector<std::pair<size_t,size_t>> intervals;
    };
    std::vector<backing_range> ranges;
    std::unordered_map<buffer*, size_t> indices;
    for (auto binding : bindings) {
        auto* native = &binding.native->storage();
        auto first = binding.native->storage_offset(), bytes = binding.native->size;
        require(first <= native->size && bytes <= native->size-first && bytes &&
            binding.address && bytes <= UINT64_MAX-binding.address, CUDA_ERROR_INVALID_VALUE, "invalid shader allocation range");
        auto [at, inserted] = indices.emplace(native, ranges.size());
        if (inserted) ranges.push_back({native, first, first+bytes});
        else { auto& range = ranges[at->second]; range.first=std::min(first,range.first); range.end=std::max(first+bytes,range.end); }
        ranges[at->second].intervals.push_back({first,first+bytes});
        ranges[at->second].descriptor |= binding.image_descriptor;
    }
    // Give the largest reachable backing ranges the scarce native slots. Pool
    // slices and aliases share one slot, or one staged range, in either path.
    std::vector<size_t> candidates;
    for (size_t i=0;i<ranges.size();++i)
        if (!ranges[i].descriptor && owner.can_bind_direct(*ranges[i].native) && ranges[i].end <= packet_offset_mask) candidates.push_back(i);
    std::stable_sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
        return ranges[a].end-ranges[a].first > ranges[b].end-ranges[b].first;
    });
    shader_memory_plan result;
    auto slots = std::min(packet_direct_bindings, owner.direct_binding_limit());
    for (auto i : candidates) {
        if (result.direct.size() == slots) break;
        ranges[i].binding = uint32_t(result.direct.size()+1); result.direct.push_back(ranges[i].native);
    }
    auto capacity = std::min(uint64_t(owner.shader_arena_capacity()), packet_arena_capacity);
    for (auto& range : ranges) if (!range.binding) {
        auto aligned = (uint64_t(result.arena_bytes)+15)&~uint64_t(15);
        auto bytes = range.end-range.first;
        require(aligned <= capacity && bytes <= capacity-aligned, CUDA_ERROR_OUT_OF_MEMORY, "staged shader storage exceeds native resource capacity");
        range.offset = size_t(aligned);
        // Copy only the live interval union. Gaps can belong to independent
        // contexts sharing a pool and must never be written back by this launch.
        std::sort(range.intervals.begin(),range.intervals.end());
        for (auto [first,end] : range.intervals) {
            if (!result.staged.empty() && result.staged.back().native == range.native &&
                first <= result.staged.back().first+result.staged.back().bytes) {
                auto& previous=result.staged.back(); previous.bytes=std::max(end,previous.first+previous.bytes)-previous.first;
            } else result.staged.push_back({range.native,first,end-first,range.offset+(first-range.first)});
        }
        result.arena_bytes = size_t(aligned+bytes);
    }
    result.arena_bytes = size_t((uint64_t(result.arena_bytes)+15)&~uint64_t(15));
    require(result.arena_bytes <= capacity, CUDA_ERROR_OUT_OF_MEMORY, "shader arena exceeds native resource capacity");
    for (auto binding : bindings) {
        const auto& range = ranges[indices.at(&binding.native->storage())];
        auto offset = binding.native->storage_offset();
        result.records.push_back({split_packet_word64(binding.address), binding.native->size,
            range.binding ? uint64_t(offset) : uint64_t(range.offset+(offset-range.first)), range.binding});
    }
    std::sort(result.records.begin(), result.records.end(), [](const auto& a, const auto& b) {
        return join_packet_word64(a.address) < join_packet_word64(b.address);
    });
    for (size_t i=1;i<result.records.size();++i)
        require(join_packet_word64(result.records[i].address)-join_packet_word64(result.records[i-1].address) >= result.records[i-1].size,
            CUDA_ERROR_INVALID_VALUE, "overlapping shader virtual allocation ranges");
    return result;
}
}
