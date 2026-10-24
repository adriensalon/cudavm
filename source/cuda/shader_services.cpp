#include "backend.hpp"
#include "../core/device_services.hpp"
#include <bit>
#include <cstdio>
namespace cudavm {
size_t shader_service_limit(session& owner, CUlimit limit) {
    if (limit == CU_LIMIT_MALLOC_HEAP_SIZE) return owner.device_heap_limit;
    if (limit == CU_LIMIT_PRINTF_FIFO_SIZE) return owner.device_output_limit;
    throw failure(CUDA_ERROR_UNSUPPORTED_LIMIT, "this context resource limit is not implemented on the adapter");
}
void set_shader_service_limit(session& owner, CUlimit limit, size_t value) {
    if (limit == CU_LIMIT_MAX_L2_FETCH_GRANULARITY || limit == CU_LIMIT_PERSISTING_L2_CACHE_SIZE) return;
    require(limit == CU_LIMIT_MALLOC_HEAP_SIZE || limit == CU_LIMIT_PRINTF_FIFO_SIZE, CUDA_ERROR_UNSUPPORTED_LIMIT, "this context resource limit is not implemented on the adapter");
    auto capacity = std::min(owner.shader_arena_capacity(), packet_arena_capacity);
    if (limit == CU_LIMIT_MALLOC_HEAP_SIZE) {
        require(!owner.device_heap, CUDA_ERROR_INVALID_VALUE, "device heap size must be set before loading a module using it");
        require(value <= capacity, CUDA_ERROR_OUT_OF_MEMORY, "device heap exceeds native capacity");
        auto payload = std::max(size_t(16), (value + 15) & ~size_t(15));
        auto chunks = uint64_t(payload / 16), words = (chunks + 31) / 32;
        auto data = (16 + words * 4 + chunks * 4 + 15) & ~uint64_t(15);
        require(data + payload <= capacity, CUDA_ERROR_OUT_OF_MEMORY, "device heap exceeds native capacity");
        owner.device_heap_limit = payload;
    } else {
        require(!owner.device_output_used, CUDA_ERROR_INVALID_VALUE, "printf size must be set before a kernel uses it");
        capacity = std::min(capacity, device_output_max_bytes);
        require(value <= capacity && value <= UINT32_MAX - device_output_record_bytes, CUDA_ERROR_OUT_OF_MEMORY, "device output FIFO exceeds native capacity");
        auto slots = std::max(size_t(1), (value + device_output_record_bytes - 1) / device_output_record_bytes);
        auto bytes = device_output_header_bytes + slots * device_output_record_bytes;
        require(bytes <= capacity, CUDA_ERROR_OUT_OF_MEMORY, "device output FIFO exceeds native capacity");
        owner.device_output_limit = bytes;
    }
}
void prepare_shader_services(session& owner, uint32_t flags) {
    if (!(flags & device_service_heap) || owner.device_heap) return;
    uint64_t chunks = owner.device_heap_limit / 16, words = (chunks + 31) / 32;
    uint64_t data = (16 + words * 4 + chunks * 4 + 15) & ~uint64_t(15);
    require(data + owner.device_heap_limit <= std::min(owner.shader_arena_capacity(), packet_arena_capacity), CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, "device heap exceeds native resource capacity");
    std::shared_ptr<buffer> heap;
    try { heap = owner.allocate(size_t(data + owner.device_heap_limit)); }
    catch (const failure&) { throw failure(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, "cannot allocate context device heap"); }
    std::vector<std::byte> metadata(size_t(data), std::byte{});
    uint32_t header[]{uint32_t(chunks), uint32_t(words), uint32_t(data), 0};
    std::memcpy(metadata.data(), header, sizeof(header)); owner.write(*heap, 0, metadata);
    owner.device_heap = std::move(heap);
}
namespace {
uint32_t number(std::span<const std::byte> bytes, size_t at) {
    require(at <= bytes.size() && bytes.size() - at >= 4, CUDA_ERROR_INVALID_IMAGE, "truncated device output record");
    uint32_t result; std::memcpy(&result, bytes.data() + at, 4); return result;
}
std::string_view text(std::span<const std::byte> bytes, size_t at) {
    require(at < bytes.size(), CUDA_ERROR_INVALID_IMAGE, "invalid device output string");
    auto* first = reinterpret_cast<const char*>(bytes.data() + at);
    auto* end = static_cast<const char*>(std::memchr(first, 0, bytes.size() - at));
    require(end, CUDA_ERROR_INVALID_IMAGE, "unterminated device output string"); return {first, size_t(end - first)};
}
template<class T> void print_value(std::string& output, const std::string& format, T value, const std::vector<int>& stars) {
    auto print = [&](char* data, size_t bytes) {
        if (stars.empty()) return std::snprintf(data, bytes, format.c_str(), value);
        if (stars.size() == 1) return std::snprintf(data, bytes, format.c_str(), stars[0], value);
        return std::snprintf(data, bytes, format.c_str(), stars[0], stars[1], value);
    };
    auto size = print(nullptr, 0);
    require(size >= 0, CUDA_ERROR_INVALID_VALUE, "device printf formatting failed");
    std::string result(size_t(size) + 1, '\0'); print(result.data(), result.size()); output.append(result.data(), size_t(size));
}
std::string format_record(std::span<const std::byte> record) {
    auto format = text(record, number(record, 40)); unsigned count = number(record, 12), argument = 0;
    require(count <= 32, CUDA_ERROR_INVALID_IMAGE, "invalid device printf argument count");
    std::string output;
    for (size_t at = 0; at < format.size();) {
        if (format[at] != '%') { output += format[at++]; continue; }
        auto first = at++; if (at < format.size() && format[at] == '%') { output += '%'; ++at; continue; }
        while (at < format.size() && std::string_view("-+ #0").find(format[at]) != std::string_view::npos) ++at;
        std::vector<int> stars;
        auto star = [&] {
            if (argument < count) stars.push_back(int(number(record, 64 + argument++ * 16 + 8)));
        };
        if (at < format.size() && format[at] == '*') { star(); ++at; }
        else while (at < format.size() && format[at] >= '0' && format[at] <= '9') ++at;
        if (at < format.size() && format[at] == '.') {
            ++at; if (at < format.size() && format[at] == '*') { star(); ++at; }
            else while (at < format.size() && format[at] >= '0' && format[at] <= '9') ++at;
        }
        auto length = at;
        if (at < format.size() && (format[at] == 'h' || format[at] == 'l')) { char c = format[at++]; if (at < format.size() && format[at] == c) ++at; }
        auto conversion = at;
        if (at >= format.size()) { output.append(format.substr(first)); break; }
        char type = format[at++];
        if (std::string_view("cdiouxXpeEfFgGaAs").find(type) == std::string_view::npos || argument >= count) {
            output.append(format.substr(first, at - first)); continue;
        }
        auto base = 64 + argument++ * 16; auto code = number(record, base);
        uint64_t bits = number(record, base + 8) | (uint64_t(number(record, base + 12)) << 32);
        bool integer = std::string_view("cdiouxX").find(type) != std::string_view::npos;
        require((integer && code >= 1 && code <= 4) || (type == 's' && code == 6) || (type == 'p' && code == 7) ||
            (!integer && type != 's' && type != 'p' && code == 5), CUDA_ERROR_INVALID_IMAGE, "device printf format and argument types differ");
        std::string spec(format.substr(first, length - first));
        if (code == 3 || code == 4) spec += "ll";
        else if ((code == 1 || code == 2) && format.substr(length, conversion - length) != "l") spec.append(format.substr(length, conversion - length));
        spec += type;
        switch (code) {
        case 1: case 2:
            if (type == 'd' || type == 'i' || type == 'c') print_value(output, spec, int32_t(bits), stars);
            else print_value(output, spec, uint32_t(bits), stars);
            break;
        case 3: case 4:
            if (type == 'd' || type == 'i') print_value(output, spec, static_cast<long long>(bits), stars);
            else print_value(output, spec, static_cast<unsigned long long>(bits), stars);
            break;
        case 5: print_value(output, spec, std::bit_cast<double>(bits), stars); break;
        case 6: print_value(output, spec, text(record, number(record, base + 4)).data(), stars); break;
        case 7: print_value(output, spec, reinterpret_cast<void*>(uintptr_t(bits)), stars); break;
        default: throw failure(CUDA_ERROR_INVALID_IMAGE, "invalid device printf argument type");
        }
    }
    return output;
}
}
void complete_shader_services(session& owner, buffer* fifo, buffer* arena, uint32_t flags) {
    uint32_t asserted = 0;
    if (flags & device_service_assert) owner.read(*arena, device_assert_fault_word * 4, std::as_writable_bytes(std::span(&asserted, 1)));
    // Publish the sticky fault even if an output record was overwritten or a
    // host formatting error occurs. A fresh context after reset owns fresh state.
    if (asserted) owner.device_fault.store(CUDA_ERROR_ASSERT);
    if (fifo) {
        std::vector<std::byte> bytes(fifo->size); owner.read(*fifo, 0, bytes);
        if (std::getenv("CUDAVM_TRACE_NATIVE")) std::fprintf(stderr, "device output tickets=%u slots=%u\n", number(bytes, 0), number(bytes, 4));
        struct record_view { uint32_t ticket; std::span<const std::byte> bytes; };
        std::vector<record_view> records;
        for (size_t at = device_output_header_bytes; at + device_output_record_bytes <= bytes.size(); at += device_output_record_bytes) {
            auto record = std::span<const std::byte>(bytes).subspan(at, device_output_record_bytes);
            if (number(record, 4)) records.push_back({number(record, 48), record});
        }
        std::sort(records.begin(), records.end(), [](auto a, auto b) { return int32_t(a.ticket - b.ticket) < 0; });
        for (auto item : records) {
            auto record = item.bytes;
            if (number(record, 4) == 1) {
                auto output = format_record(record); std::fwrite(output.data(), 1, output.size(), stdout);
            } else if (number(record, 4) == 2) {
                auto message = text(record, number(record, 40)), file = text(record, number(record, 44)), function = text(record, number(record, 56));
                std::fprintf(stderr, "%s:%u:%s: block: [%u,%u,%u], thread: [%u,%u,%u] Assertion `%s` failed.\n",
                    file.data(), number(record, 12), function.data(), number(record, 16), number(record, 20), number(record, 24),
                    number(record, 28), number(record, 32), number(record, 36), message.data());
            }
        }
        std::fflush(stdout); std::fflush(stderr);
    }
    require(!asserted, CUDA_ERROR_ASSERT, "device assertion requires context reset");
}
}
