#pragma once
namespace cudavm {
inline constexpr const char device_services_source[] = R"cudavm(
#define DEV __attribute__((device))
using size_t = __SIZE_TYPE__;
using u64 = unsigned long long;
extern "C" DEV void* __cudavm_service_buffer(unsigned);
extern "C" DEV void __cudavm_service_assert();
extern "C" DEV unsigned __cudavm_service_id(unsigned);
namespace cudavm {
DEV unsigned cas(unsigned* p, unsigned old, unsigned value) { return __sync_val_compare_and_swap(p, old, value); }
DEV unsigned get(unsigned* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
DEV void put(unsigned* p, unsigned value) { __atomic_store_n(p, value, __ATOMIC_RELAXED); }
DEV unsigned* output_pointer(unsigned* pointer) {
    return reinterpret_cast<unsigned*>(reinterpret_cast<u64>(__cudavm_service_buffer(1)) | (reinterpret_cast<u64>(pointer) & 0x7fffffffull));
}
// Every acquisition either succeeds or rolls back; no invocation waits while
// owning a lock. A bitmap bit represents sixteen payload bytes. Length metadata
// is separate, so free from another thread needs no user-memory header access.
DEV void* allocate(size_t bytes, size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) return nullptr;
    alignment = alignment < 16 ? 16 : alignment;
    auto* heap = static_cast<unsigned*>(__cudavm_service_buffer(0));
    unsigned chunks = heap[0], bitmap_words = heap[1], data = heap[2];
    if (bytes > u64(chunks) * 16 || alignment > u64(chunks) * 16) return nullptr;
    unsigned need = unsigned((bytes + 15) / 16); if (!need) need = 1;
    auto* bitmap = heap + 4; auto* lengths = bitmap + bitmap_words;
    unsigned step = unsigned(alignment / 16);
    for (unsigned start = 0; start <= chunks - need;) {
        // Align the actual data address, not just its index in the bitmap.
        u64 address = reinterpret_cast<u64>(heap) + data + u64(start) * 16;
        unsigned adjustment = unsigned((0ull - address) & (alignment - 1)) / 16;
        if (adjustment > chunks - need - start) break;
        start += adjustment;
        address += u64(adjustment) * 16;
        unsigned end = start + need, word = start / 32, last = (end - 1) / 32, claimed = word;
        bool success = true;
        for (; claimed <= last; ++claimed) {
            unsigned low = claimed == word ? start % 32 : 0, high = claimed == last ? (end - 1) % 32 + 1 : 32;
            unsigned mask = (~0u >> (32 - (high - low))) << low;
            unsigned previous = get(bitmap + claimed);
            for (;;) {
                if (previous & mask) { success = false; break; }
                unsigned observed = cas(bitmap + claimed, previous, previous | mask);
                if (observed == previous) break;
                previous = observed;
            }
            if (!success) break;
        }
        if (success) { put(lengths + start, need); return reinterpret_cast<void*>(address); }
        for (unsigned undo = word; undo < claimed; ++undo) {
            unsigned low = undo == word ? start % 32 : 0, high = undo == last ? (end - 1) % 32 + 1 : 32;
            unsigned mask = (~0u >> (32 - (high - low))) << low;
            __atomic_fetch_and(bitmap + undo, ~mask, __ATOMIC_RELEASE);
        }
        // Skip occupied bits in the failed word. Retrying only higher addresses
        // makes contention finite without a subgroup-dependent spin lock.
        unsigned occupied = get(bitmap + claimed), next = start + step;
        if (occupied) {
            unsigned first = unsigned(__builtin_ctz(occupied));
            unsigned conflict = claimed * 32 + first;
            if (conflict >= start && conflict < end) next = conflict + 1;
        }
        start = next;
    }
    return nullptr;
}
extern "C" DEV void* malloc(size_t bytes) { return allocate(bytes, 16); }
extern "C" DEV void* __nv_aligned_device_malloc(size_t bytes, size_t alignment) { return allocate(bytes, alignment); }
extern "C" DEV void* __nv_aligned_device_malloc_impl(size_t bytes, size_t alignment) { return allocate(bytes, alignment); }
extern "C" DEV void free(void* pointer) {
    if (!pointer) return;
    auto* heap = static_cast<unsigned*>(__cudavm_service_buffer(0));
    u64 offset = reinterpret_cast<u64>(pointer) - reinterpret_cast<u64>(heap) - heap[2];
    if (offset % 16 || offset / 16 >= heap[0]) return;
    auto* bitmap = heap + 4; auto* lengths = bitmap + heap[1]; unsigned start = unsigned(offset / 16);
    unsigned need = get(lengths + start); if (!need || need > heap[0] - start) return;
    put(lengths + start, 0);
    unsigned end = start + need, last = (end - 1) / 32;
    for (unsigned word = start / 32; word <= last; ++word) {
        unsigned low = word == start / 32 ? start % 32 : 0, high = word == last ? (end - 1) % 32 + 1 : 32;
        unsigned mask = (~0u >> (32 - (high - low))) << low;
        __atomic_fetch_and(bitmap + word, ~mask, __ATOMIC_RELEASE);
    }
}
// Fixed records provide independent publication. A busy slot can occur only
// after the bounded circular FIFO has wrapped; never wait on another lane.
DEV unsigned* record(unsigned kind) {
    auto* fifo = static_cast<unsigned*>(__cudavm_service_buffer(1));
    unsigned ticket = __atomic_fetch_add(fifo, 1u, __ATOMIC_RELAXED), slots = fifo[1];
    if (!slots) return nullptr;
    auto* out = output_pointer(fifo + 4 + (ticket % slots) * 1024);
    if (cas(out, 0, 1)) return nullptr;
    out[1] = 0; out[12] = ticket; out[13] = kind; return out;
}
DEV unsigned text(unsigned* out, unsigned& at, const char* input) {
    unsigned begin = at; auto* bytes = reinterpret_cast<char*>(out);
    if (!input) input = "(null)";
    do { if (at >= 4096) return 0; bytes[at++ & 4095] = *input; } while (*input++);
    return begin;
}
DEV void publish(unsigned* out, unsigned at, unsigned kind) {
    out[2] = at; out[1] = kind; __atomic_store_n(out, 0u, __ATOMIC_RELEASE);
}
extern "C" DEV int __cudavm_service_literal(const char* format) {
    if (auto* claimed = record(1)) {
        auto* out = output_pointer(claimed);
        unsigned at = 576; out[3] = 0; out[10] = text(out, at, format);
        bool valid = out[10] != 0; publish(out, at, valid ? 1 : 0); return valid ? 0 : -2;
    }
    return 0;
}
extern "C" DEV u64 __cudavm_service_scan(const char* format) {
    unsigned count = 0, strings = 0; bool conversion = false;
    for (auto* p = format; *p; ++p) {
        unsigned type = static_cast<unsigned char>(*p);
        if (!conversion) { conversion = type == '%'; continue; }
        if (type == '%') { conversion = false; continue; }
        if (type == 'h' || type == 'l' || type == '-' || type == '+' || type == ' ' || type == '#' || type == '.' || (type >= '0' && type <= '9')) continue;
        bool star = type == '*';
        unsigned lower = type | 32, index = lower - 'a', bit = 1u << (index & 31);
        bool real = index < 26 && (bit & 0x71u);
        bool integer = index < 26 && (type == lower || type == 'X') && (bit & 0x90410cu);
        if (!star) conversion = false;
        if ((star || real || integer || type == 's' || type == 'p') && count < 32) {
            if (type == 's') strings |= 1u << count;
            ++count;
        }
    }
    return (u64(strings) << 32) | count;
}
extern "C" DEV int vprintf(const char* format, const char* arguments) {
    if (!format) return -1;
    unsigned* claimed = record(1); bool owned = claimed != nullptr;
    unsigned* out = output_pointer(claimed); unsigned at = 576, count = 0, offset = 0;
    bool valid = true, conversion = false; unsigned length = 0;
    if (owned) { out[10] = text(out, at, format); valid = out[10] != 0; }
    for (auto* p = format; *p; ++p) {
        unsigned type = static_cast<unsigned char>(*p);
        if (!conversion) { conversion = type == '%'; length = 0; continue; }
        if (type == '%') { conversion = false; continue; }
        if (type == 'l') { ++length; continue; }
        if (type == 'h' || type == '-' || type == '+' || type == ' ' || type == '#' || type == '.' || (type >= '0' && type <= '9')) continue;
        bool star = type == '*';
        unsigned lower = type | 32, index = lower - 'a', bit = 1u << (index & 31);
        bool real = index < 26 && (bit & ((1u << 0) | (1u << 4) | (1u << 5) | (1u << 6)));
        bool integer = index < 26 && (type == lower || type == 'X') &&
            (bit & ((1u << 2) | (1u << 3) | (1u << 8) | (1u << 14) | (1u << 20) | (1u << 23)));
        if (!star) conversion = false;
        if (!star && !real && !integer && type != 's' && type != 'p') continue;
        if (count == 32) continue;
        bool signed_value = type == 'd' || type == 'i' || type == 'c';
        bool wide = length > 1 || (length == 1 && sizeof(long) == 8);
        unsigned code = star ? 1 : real ? 5 : type == 's' ? 6 : type == 'p' ? 7 : wide ? (signed_value ? 3 : 4) : (signed_value ? 1 : 2);
        unsigned bytes = code >= 3 ? 8 : 4; offset = (offset + bytes - 1) & ~(bytes - 1);
        if (!arguments) { valid = false; break; }
        u64 bits = bytes == 8 ? *reinterpret_cast<const u64*>(arguments + offset) : *reinterpret_cast<const unsigned*>(arguments + offset);
        offset += bytes;
        if (owned && valid) {
            auto* arg = out + 16 + (count & 31) * 4; arg[0] = code; arg[1] = 0; arg[2] = unsigned(bits); arg[3] = unsigned(bits >> 32);
            if (code == 6) { arg[1] = text(out, at, reinterpret_cast<const char*>(bits)); valid = arg[1] != 0; }
        }
        ++count;
    }
    if (owned) { out[3] = count; publish(out, at, valid ? 1 : 0); }
    return valid ? int(count) : -2;
}
extern "C" DEV void __assertfail(const char* message, const char* file, unsigned line, const char* function, size_t character_size) {
    if (auto* claimed = record(2)) {
        auto* out = output_pointer(claimed);
        unsigned at = 64; out[3] = line;
        for (unsigned i = 0; i < 6; ++i) out[4 + i] = __cudavm_service_id(i);
        out[10] = text(out, at, message); out[11] = text(out, at, file); out[14] = text(out, at, function);
        publish(out, at, out[10] && out[11] && out[14] && character_size == 1 ? 2 : 0);
    }
    __cudavm_service_assert();
    __builtin_unreachable();
}
}
)cudavm";
}
