#pragma once
#ifdef CUDAVM_TEST_NVRTC
#include <nvrtc.h>
#endif
#ifdef CUDAVM_TEST_NVJITLINK
#include <nvJitLink.h>
#endif
#ifndef CUDAVM_TEST_HOST_ONLY
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#ifdef CUDAVM_TEST_REGISTRATION
#include <cuda_runtime_api.h>
// Private compiler ABI declarations belong to the registration fixtures.
extern "C" {
void** CUDARTAPI __cudaRegisterFatBinary(void*);
void CUDARTAPI __cudaRegisterFatBinaryEnd(void**);
void CUDARTAPI __cudaUnregisterFatBinary(void**);
void CUDARTAPI __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
void CUDARTAPI __cudaRegisterVar(void**, char*, char*, const char*, int, size_t, int, int);
void CUDARTAPI __cudaRegisterManagedVar(void**, void**, char*, const char*, int, size_t, int, int);
char CUDARTAPI __cudaInitModule(void**);
}
#endif

namespace cudavm {
void require(bool condition, const std::string& message);
// Host/compiler contracts share reporting without GPU initialization or Driver
// references. HOST_ONLY registration selects this subset of the common runner.
int run_host_test(int argc, char** argv, void (*execute)());

#if defined(CUDAVM_TEST_NVRTC) || defined(CUDAVM_TEST_NVJITLINK)
template<class Size, class Get> std::vector<unsigned char> compiler_fixture_bytes(Size query, Get retrieve) {
    struct { size_t before = 0x5917ab3d; size_t value{}; size_t after = 0x93bd175a; } size;
    query(&size.value);
    require(size.before == 0x5917ab3d && size.after == 0x93bd175a, "compiler binary size output guards");
    require(size.value > 0 && size.value < 64 * 1024 * 1024, "compiler fixture binary extent is nonempty and bounded");
    std::vector<unsigned char> bytes(size.value + 32, 0x97);
    retrieve(bytes.data() + 16);
    for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= size.value + 16)
        require(bytes[i] == 0x97, "compiler binary retrieval exact extent guards");
    return {bytes.begin() + 16, bytes.end() - 16};
}
#endif
// Compiler helpers own handles and exact output extents. Each slice retains
// its sources, options and semantic oracle. CMake enables only linked APIs.
#ifdef CUDAVM_TEST_NVRTC
struct compiler_fixture_program {
    nvrtcProgram value{};
    compiler_fixture_program(const std::string& source, const char* name);
    ~compiler_fixture_program();
    compiler_fixture_program(const compiler_fixture_program&) = delete;
    compiler_fixture_program& operator=(const compiler_fixture_program&) = delete;
    static void check(nvrtcResult result);
    nvrtcResult compile(const std::vector<std::string>& options = {}) const;
    std::string text(bool log = false) const;
    std::string success(const std::vector<std::string>& options = {}) const;
    std::vector<unsigned char> lto() const;
};
#endif
#ifdef CUDAVM_TEST_NVJITLINK
struct linker_fixture_program {
    nvJitLinkHandle value{};
    explicit linker_fixture_program(const std::vector<std::string>& options = {"-arch=sm_75", "-lto", "-O0"});
    ~linker_fixture_program();
    linker_fixture_program(const linker_fixture_program&) = delete;
    linker_fixture_program& operator=(const linker_fixture_program&) = delete;
    void check(nvJitLinkResult status, const char* operation) const;
    void add(const std::vector<unsigned char>& image, const char* name, nvJitLinkInputType kind = NVJITLINK_INPUT_LTOIR) const;
    void complete() const;
    std::vector<unsigned char> image() const;
};
#endif

struct test_child_result { int status; std::string output; };
// Run the same executable with <backend> <case> --worker and inherited stdout
// capture. The worker has a bounded deadline and propagates skip/failure status.
test_child_result run_test_child(const char* executable, const char* backend, const char* case_name,
                                 bool capture_stderr = false);

// A fresh copy of this executable receives <backend> <case> --peer <read>
// <write> <partner>. Control bytes and real OS handles use separate operations;
// GPU payloads never pass through this channel. One thread owns each endpoint.
class test_peer {
    struct state;
    std::unique_ptr<state> storage;
    explicit test_peer(std::unique_ptr<state> value);
public:
    test_peer(const char* executable, const char* backend, const char* case_name);
    static test_peer connect(std::string_view read, std::string_view write, std::string_view partner);
    ~test_peer();
    test_peer(test_peer&&) noexcept;
    test_peer(const test_peer&) = delete;
    void send(std::span<const std::byte> bytes);
    void receive(std::span<std::byte> bytes);
    void send_handle(uintptr_t handle);
    uintptr_t receive_handle();
    int wait();
};

struct fixture_data {
    std::string_view source, ptx, nvvm;
    std::span<const unsigned char> cubin;
    unsigned architecture = 75;
    unsigned minimum_ptx = 0;
    std::string_view architecture_suffix;
    unsigned source_standard = 17;
};
inline std::string test_architecture(const fixture_data& fixtures) {
    return std::to_string(fixtures.architecture) + std::string(fixtures.architecture_suffix);
}

// Independent exact dyadic arithmetic for numerical fixtures. No host floating
// operations, compiler internals, CUDA helpers or shader implementations are used.
// Limbs are little endian; addition and multiplication retain every exact bit.
struct numerical_integer {
    std::vector<uint32_t> words;
    explicit numerical_integer(uint64_t value = 0) {
        if (value) words.push_back(uint32_t(value));
        if (value >> 32) words.push_back(uint32_t(value >> 32));
    }
    void trim() { while (!words.empty() && !words.back()) words.pop_back(); }
    unsigned width() const { return words.empty() ? 0 : unsigned((words.size() - 1) * 32 + 32 - std::countl_zero(words.back())); }
    bool bit(unsigned position) const { return position / 32 < words.size() && ((words[position / 32] >> (position % 32)) & 1); }
    bool below(unsigned position) const {
        const auto whole = std::min(size_t(position / 32), words.size());
        for (size_t i = 0; i < whole; ++i) if (words[i]) return true;
        return whole < words.size() && position % 32 && (words[whole] & ((uint32_t(1) << (position % 32)) - 1));
    }
    uint64_t window(unsigned position) const {
        uint64_t result = 0;
        for (unsigned i = 0; i < 64; ++i) if (bit(position + i)) result |= uint64_t(1) << i;
        return result;
    }
    numerical_integer shifted(unsigned shift) const {
        numerical_integer result;
        if (words.empty()) return result;
        result.words.resize(words.size() + shift / 32 + 1);
        for (size_t i = 0; i < words.size(); ++i) {
            const uint64_t value = uint64_t(words[i]) << (shift % 32);
            result.words[i + shift / 32] |= uint32_t(value);
            result.words[i + shift / 32 + 1] |= uint32_t(value >> 32);
        }
        result.trim(); return result;
    }
    int compare(const numerical_integer& other) const {
        if (words.size() != other.words.size()) return words.size() < other.words.size() ? -1 : 1;
        for (size_t i = words.size(); i; --i) if (words[i - 1] != other.words[i - 1]) return words[i - 1] < other.words[i - 1] ? -1 : 1;
        return 0;
    }
    numerical_integer plus(const numerical_integer& other) const {
        numerical_integer result; result.words.resize(std::max(words.size(), other.words.size()) + 1);
        uint64_t carry = 0;
        for (size_t i = 0; i < result.words.size(); ++i) {
            carry += (i < words.size() ? uint64_t(words[i]) : 0) + (i < other.words.size() ? uint64_t(other.words[i]) : 0);
            result.words[i] = uint32_t(carry); carry >>= 32;
        }
        result.trim(); return result;
    }
    numerical_integer minus(const numerical_integer& other) const {
        numerical_integer result; result.words.resize(words.size());
        uint64_t borrow = 0;
        for (size_t i = 0; i < words.size(); ++i) {
            const uint64_t right = (i < other.words.size() ? uint64_t(other.words[i]) : 0) + borrow;
            result.words[i] = uint32_t(uint64_t(words[i]) - right); borrow = uint64_t(words[i]) < right;
        }
        result.trim(); return result;
    }
    numerical_integer times(const numerical_integer& other) const {
        numerical_integer result; result.words.resize(words.size() + other.words.size());
        for (size_t i = 0; i < words.size(); ++i) {
            uint64_t carry = 0;
            for (size_t j = 0; j < other.words.size(); ++j) {
                const uint64_t value = uint64_t(words[i]) * other.words[j] + result.words[i + j] + carry;
                result.words[i + j] = uint32_t(value); carry = value >> 32;
            }
            result.words[i + other.words.size()] = uint32_t(carry);
        }
        result.trim(); return result;
    }
};
struct numerical_dyadic {
    numerical_integer magnitude;
    int scale = 0;
    bool negative = false;
};
// Rounding indices match the public suffix order rn, rz, rd, ru.
inline numerical_dyadic numerical_sum(numerical_dyadic a, numerical_dyadic b, unsigned rounding) {
    const int scale = std::min(a.scale, b.scale);
    auto left = a.magnitude.shifted(unsigned(a.scale - scale)), right = b.magnitude.shifted(unsigned(b.scale - scale));
    if (a.negative == b.negative) return {left.plus(right), scale, a.negative};
    const auto order = left.compare(right);
    if (!order) return {numerical_integer{}, scale, rounding == 2};
    return order > 0 ? numerical_dyadic{left.minus(right), scale, a.negative} : numerical_dyadic{right.minus(left), scale, b.negative};
}
template<unsigned Fraction, unsigned Exponent> struct numerical_format {
    static constexpr int bias = (1 << (Exponent - 1)) - 1;
    static constexpr uint64_t sign = uint64_t(1) << (Fraction + Exponent);
    static constexpr uint64_t infinity = ((uint64_t(1) << Exponent) - 1) << Fraction;
    static constexpr uint64_t nan = infinity | (uint64_t(1) << (Fraction - 1));
    static bool is_nan(uint64_t bits) { return (bits & (sign - 1)) > infinity; }
    static bool is_infinite(uint64_t bits) { return (bits & (sign - 1)) == infinity; }
    static bool is_zero(uint64_t bits) { return !(bits & (sign - 1)); }
    static numerical_dyadic decode(uint64_t bits) {
        const auto exponent = unsigned((bits & infinity) >> Fraction);
        const uint64_t fraction = bits & ((uint64_t(1) << Fraction) - 1);
        return {numerical_integer(fraction | (exponent ? uint64_t(1) << Fraction : 0)),
            (exponent ? int(exponent) - bias : 1 - bias) - int(Fraction), (bits & sign) != 0};
    }
    static uint64_t overflow(bool negative, unsigned rounding) {
        const bool up = !rounding || (rounding == 2 && negative) || (rounding == 3 && !negative);
        return (negative ? sign : 0) | (up ? infinity : infinity - 1);
    }
    static uint64_t encode(const numerical_dyadic& value, unsigned rounding) {
        const auto width = value.magnitude.width();
        const auto signed_bits = value.negative ? sign : 0;
        if (!width) return signed_bits;
        const int highest = int(width) - 1 + value.scale;
        if (highest > bias) return overflow(value.negative, rounding);
        int unit = std::max(highest - int(Fraction), 1 - bias - int(Fraction));
        const int shift = unit - value.scale;
        uint64_t retained;
        bool remainder = false, halfway = false, lower = false;
        if (shift > 0) {
            retained = value.magnitude.window(unsigned(shift));
            halfway = value.magnitude.bit(unsigned(shift - 1)); lower = value.magnitude.below(unsigned(shift - 1));
            remainder = halfway || lower;
        } else retained = value.magnitude.window(0) << unsigned(-shift);
        const bool increment = rounding == 0 ? halfway && (lower || (retained & 1))
            : remainder && ((rounding == 2 && value.negative) || (rounding == 3 && !value.negative));
        retained += increment;
        if (retained == (uint64_t(1) << (Fraction + 1))) { retained >>= 1; ++unit; }
        if (retained < (uint64_t(1) << Fraction)) return signed_bits | retained;
        const int exponent = unit + int(Fraction) + bias;
        if (exponent >= (1 << Exponent) - 1) return overflow(value.negative, rounding);
        return signed_bits | (uint64_t(exponent) << Fraction) | (retained - (uint64_t(1) << Fraction));
    }
    static uint64_t add(uint64_t a, uint64_t b, unsigned rounding) {
        if (is_nan(a) || is_nan(b)) return nan;
        if (is_infinite(a) || is_infinite(b)) return is_infinite(a) && is_infinite(b) && ((a ^ b) & sign) ? nan : is_infinite(a) ? a : b;
        return encode(numerical_sum(decode(a), decode(b), rounding), rounding);
    }
    static uint64_t multiply(uint64_t a, uint64_t b, unsigned rounding) {
        if (is_nan(a) || is_nan(b)) return nan;
        if (is_infinite(a) || is_infinite(b)) return is_zero(a) || is_zero(b) ? nan : ((a ^ b) & sign) | infinity;
        auto left = decode(a), right = decode(b);
        return encode({left.magnitude.times(right.magnitude), left.scale + right.scale, left.negative != right.negative}, rounding);
    }
    static uint64_t fused(uint64_t a, uint64_t b, uint64_t c, unsigned rounding) {
        if (is_nan(a) || is_nan(b) || is_nan(c)) return nan;
        if (is_infinite(a) || is_infinite(b)) {
            if (is_zero(a) || is_zero(b)) return nan;
            const auto product = ((a ^ b) & sign) | infinity;
            return is_infinite(c) && ((product ^ c) & sign) ? nan : product;
        }
        if (is_infinite(c)) return c;
        const auto left = decode(a), right = decode(b);
        return encode(numerical_sum({left.magnitude.times(right.magnitude), left.scale + right.scale,
            left.negative != right.negative}, decode(c), rounding), rounding);
    }
};
using numerical_binary32 = numerical_format<23, 8>;
using numerical_binary64 = numerical_format<52, 11>;

// Compare exact nonnegative dyadics; elementary references deliberately use
// bracketing and integer products rather than a device root/division algorithm.
inline int elementary_compare(const numerical_dyadic& a, const numerical_dyadic& b) {
    const int scale = std::min(a.scale, b.scale);
    return a.magnitude.shifted(unsigned(a.scale - scale)).compare(b.magnitude.shifted(unsigned(b.scale - scale)));
}
template<class Format> uint64_t elementary_integral(uint64_t bits, unsigned mode) {
    if (Format::is_nan(bits)) return Format::nan;
    if (Format::is_infinite(bits)) return bits;
    auto value = Format::decode(bits);
    if (value.scale >= 0 || !value.magnitude.width()) return bits;
    const auto shift = unsigned(-value.scale);
    auto retained = value.magnitude.window(shift);
    const bool halfway = value.magnitude.bit(shift - 1), lower = value.magnitude.below(shift - 1);
    const bool increment = mode == 0 ? halfway && (lower || (retained & 1))
        : mode == 4 ? halfway : (halfway || lower) && ((mode == 2 && value.negative) || (mode == 3 && !value.negative));
    return Format::encode({numerical_integer(retained + increment), 0, value.negative}, 0);
}
template<class Format> std::array<uint64_t, 6> elementary_rounding(uint64_t bits) {
    return {elementary_integral<Format>(bits, 2), elementary_integral<Format>(bits, 3),
        elementary_integral<Format>(bits, 1), elementary_integral<Format>(bits, 4),
        elementary_integral<Format>(bits, 0), elementary_integral<Format>(bits, 0)};
}
// Find the two adjacent destination values enclosing sqrt(x) or 1/x. Compare
// the exact squared midpoint or midpoint product to choose nearest-even.
// The conceptual value after maximum finite is 2^(bias+1), for overflow ties.
template<class Format> uint64_t elementary_root_value(const numerical_dyadic& source, bool reciprocal = false) {
    const numerical_dyadic one{numerical_integer(1), 0, false};
    auto compare = [&](const numerical_dyadic& candidate) {
        numerical_dyadic product{candidate.magnitude.times(reciprocal ? source.magnitude : candidate.magnitude),
            candidate.scale + (reciprocal ? source.scale : candidate.scale), false};
        return elementary_compare(product, reciprocal ? one : source);
    };
    uint64_t low = 0, high = Format::infinity;
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (compare(Format::decode(middle)) <= 0) low = middle;
        else high = middle;
    }
    const auto lower = Format::decode(low);
    if (compare(lower) == 0) return low;
    const auto upper = high == Format::infinity ? numerical_dyadic{numerical_integer(1), Format::bias + 1, false}
        : Format::decode(high);
    auto midpoint = numerical_sum(lower, upper, 0); --midpoint.scale;
    const auto order = compare(midpoint);
    return order < 0 || (order == 0 && (low & 1)) ? high : low;
}
template<class Format> uint64_t elementary_root(uint64_t bits, bool reciprocal) {
    if (Format::is_nan(bits)) return Format::nan;
    const auto magnitude = bits & (Format::sign - 1), sign = bits & Format::sign;
    if (reciprocal) {
        if (!magnitude) return sign | Format::infinity;
        if (Format::is_infinite(bits)) return sign;
    } else {
        if (!magnitude) return bits;
        if (sign) return Format::nan;
        if (Format::is_infinite(bits)) return bits;
    }
    return sign | elementary_root_value<Format>(Format::decode(magnitude), reciprocal);
}
// Bracket 1/sqrt(x) using exact candidate^2*x products. This avoids rounding
// an intermediate square root or reciprocal through any other format.
template<class Format> uint64_t elementary_rsqrt(uint64_t bits) {
    if (Format::is_nan(bits)) return Format::nan;
    if (Format::is_zero(bits)) return (bits & Format::sign) | Format::infinity;
    if (bits & Format::sign) return Format::nan;
    if (Format::is_infinite(bits)) return 0;
    const auto source = Format::decode(bits);
    const numerical_dyadic one{numerical_integer(1), 0, false};
    auto compare = [&](const numerical_dyadic& candidate) {
        return elementary_compare({candidate.magnitude.times(candidate.magnitude).times(source.magnitude),
            candidate.scale * 2 + source.scale, false}, one);
    };
    uint64_t low = 0, high = Format::infinity;
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (compare(Format::decode(middle)) <= 0) low = middle;
        else high = middle;
    }
    const auto lower = Format::decode(low);
    if (compare(lower) == 0) return low;
    auto midpoint = numerical_sum(lower, Format::decode(high), 0); --midpoint.scale;
    const auto order = compare(midpoint);
    return order < 0 || (order == 0 && (low & 1)) ? high : low;
}
// Exact rational division oracle: compare destination values and their
// midpoints after multiplication by the positive divisor, without division
// through another floating-point format or a device implementation helper.
template<class Format> uint64_t numerical_divide(uint64_t a, uint64_t b) {
    const auto sign = (a ^ b) & Format::sign;
    if (Format::is_nan(a) || Format::is_nan(b) || (Format::is_zero(a) && Format::is_zero(b))
        || (Format::is_infinite(a) && Format::is_infinite(b))) return Format::nan;
    if (Format::is_infinite(a) || Format::is_zero(b)) return sign | Format::infinity;
    if (Format::is_zero(a) || Format::is_infinite(b)) return sign;
    const auto numerator = Format::decode(a & (Format::sign - 1)), divisor = Format::decode(b & (Format::sign - 1));
    auto compare = [&](const numerical_dyadic& candidate) {
        return elementary_compare({candidate.magnitude.times(divisor.magnitude), candidate.scale + divisor.scale, false}, numerator);
    };
    uint64_t low = 0, high = Format::infinity;
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (compare(Format::decode(middle)) <= 0) low = middle;
        else high = middle;
    }
    const auto lower = Format::decode(low);
    if (compare(lower) == 0) return sign | low;
    const auto upper = high == Format::infinity ? numerical_dyadic{numerical_integer(1), Format::bias + 1, false}
        : Format::decode(high);
    auto midpoint = numerical_sum(lower, upper, 0); --midpoint.scale;
    const auto order = compare(midpoint);
    return sign | (order < 0 || (order == 0 && (low & 1)) ? high : low);
}
// Packed 16-bit numerical slices own scalar semantics and apply them to each
// lane separately. Packing happens only after all expected scalar bits exist.
template<size_t Fields, class Oracle>
std::array<uint32_t, Fields> numerical_packed16_rows(const std::array<uint32_t, 3>& input, Oracle oracle) {
    std::array<uint32_t, Fields> result{};
    for (unsigned shift : {0u, 16u}) {
        std::array<uint32_t, 3> lane{};
        for (size_t i = 0; i < input.size(); ++i) lane[i] = (input[i] >> shift) & 0xffffu;
        const auto expected = oracle(lane);
        static_assert(std::tuple_size_v<decltype(expected)> == Fields);
        for (size_t i = 0; i < Fields; ++i) result[i] |= uint32_t(expected[i]) << shift;
    }
    return result;
}
template<class Word, unsigned Fraction, unsigned Exponent>
std::vector<std::array<Word, 3>> elementary_inputs() {
    using format = numerical_format<Fraction, Exponent>;
    std::vector<std::array<Word, 3>> result;
    const uint64_t fraction_mask = (uint64_t(1) << Fraction) - 1;
    for (uint64_t exponent = 0; exponent < (uint64_t(1) << Exponent); ++exponent)
        for (uint64_t fraction : {uint64_t(0), uint64_t(1), uint64_t(2), fraction_mask / 2, fraction_mask / 2 + 1, fraction_mask - 1, fraction_mask})
            for (uint64_t sign : {uint64_t(0), format::sign}) result.push_back({Word(sign | (exponent << Fraction) | fraction), 0, 0});
    // Neighboring encodings of exact half-integers distinguish ties-even from
    // ties-away and preserve the sign when the result rounds to zero.
    for (unsigned integer = 0; integer < 129; ++integer) {
        const auto halfway = format::encode({numerical_integer(integer * 2 + 1), -1, false}, 0);
        for (auto value : {halfway - 1, halfway, halfway + 1})
            for (auto sign : {uint64_t(0), format::sign}) result.push_back({Word(value | sign), 0, 0});
    }
    uint64_t state = 0x751c3a9b692df4e1ull;
    for (unsigned i = 0; i < 4097; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        result.push_back({Word(state), 0, 0});
    }
    return result;
}

inline numerical_dyadic numerical_integer_value(uint64_t bits, unsigned width, bool signed_value) {
    const uint64_t mask = width == 64 ? UINT64_MAX : (uint64_t(1) << width) - 1;
    bits &= mask;
    const bool negative = signed_value && ((bits >> (width - 1)) & 1);
    return {numerical_integer(negative ? (0 - bits) & mask : bits), 0, negative};
}
// Inputs for an asserted floating-to-integer result are finite and lie inside
// the original integer domain. This conservative rule avoids depending on an
// unspecified out-of-range conversion, even when rounding might return zero.
template<class Format> unsigned numerical_integer_domains(uint64_t bits) {
    const auto magnitude = bits & (Format::sign - 1);
    if (magnitude >= Format::infinity) return 0;
    const bool negative = (bits & Format::sign) != 0;
    const auto value = Format::decode(bits);
    unsigned mask = 0;
    for (unsigned group = 0; group < 4; ++group) {
        const unsigned width = group < 2 ? 32 : 64;
        const bool sign = !(group & 1);
        const uint64_t limit = sign ? (uint64_t(1) << (width - 1)) - unsigned(!negative)
            : width == 64 ? UINT64_MAX : (uint64_t(1) << width) - 1;
        const auto order = value.scale < 0 ? value.magnitude.compare(numerical_integer(limit).shifted(unsigned(-value.scale)))
            : value.magnitude.shifted(unsigned(value.scale)).compare(numerical_integer(limit));
        const bool valid = order <= 0 && (sign || !negative || !magnitude);
        if (valid) mask |= 1u << group;
    }
    return mask;
}
template<class Format> uint64_t numerical_to_integer(uint64_t bits, unsigned rounding) {
    const auto value = Format::decode(bits);
    uint64_t magnitude;
    bool halfway = false, lower = false;
    if (value.scale < 0) {
        const auto shift = unsigned(-value.scale);
        magnitude = value.magnitude.window(shift);
        halfway = value.magnitude.bit(shift - 1); lower = value.magnitude.below(shift - 1);
    } else magnitude = value.magnitude.window(0) << unsigned(value.scale);
    const bool increment = !rounding ? halfway && (lower || (magnitude & 1))
        : (halfway || lower) && ((rounding == 2 && value.negative) || (rounding == 3 && !value.negative));
    magnitude += increment;
    return value.negative ? 0 - magnitude : magnitude;
}
inline std::vector<std::array<uint64_t, 3>> numerical_integer_inputs() {
    std::vector<std::array<uint64_t, 3>> inputs;
    for (unsigned bit = 0; bit < 64; ++bit) for (uint64_t distance = 0; distance < 5; ++distance) {
        const uint64_t power = uint64_t(1) << bit;
        for (auto value : {power - distance, power + distance}) {
            inputs.push_back({value, 0, 0}); inputs.push_back({0 - value, 0, 0});
        }
    }
    uint64_t state = 0x5917c3ab8d6f2031ull;
    for (unsigned i = 0; i < 4097; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        inputs.push_back({state, 0, 0});
    }
    return inputs;
}
template<class Format, unsigned Fraction, unsigned Exponent>
std::vector<std::array<uint64_t, 3>> numerical_real_conversion_inputs() {
    std::vector<std::array<uint64_t, 3>> inputs;
    auto append = [&](uint64_t bits) { inputs.push_back({bits, numerical_integer_domains<Format>(bits), 0}); };
    for (unsigned exponent = 0; exponent < (1u << Exponent); ++exponent)
        for (uint64_t fraction : {uint64_t(0), uint64_t(1), (uint64_t(1) << (Fraction - 1)) - 1,
            uint64_t(1) << (Fraction - 1), (uint64_t(1) << Fraction) - 2, (uint64_t(1) << Fraction) - 1})
            for (uint64_t sign : {uint64_t(0), Format::sign}) append(sign | (uint64_t(exponent) << Fraction) | fraction);
    for (unsigned integer = 0; integer < 513; ++integer) {
        const auto halfway = Format::encode({numerical_integer(integer * 2 + 1), -1, false}, 0);
        for (auto bits : {halfway - 1, halfway, halfway + 1}) { append(bits); append(bits | Format::sign); }
    }
    return inputs;
}

template<class Word> inline constexpr Word numerical_output_guard = Word(0x9b3d715ac6e8294full);

#ifndef CUDAVM_TEST_HOST_ONLY
// Only an absent declared device prerequisite is unavailable evidence. API or
// execution failures on an applicable device must still fail the test.
void require_capability(bool available, const std::string& reason);
// Query native service availability independently of the pinned facade version.
std::string native_cuda_reference_library();
int native_cuda_version();
void check_driver(CUresult result, const char* operation);
#define TEST_CUDA(call) ::cudavm::check_driver((call), #call)

#ifdef CUDAVM_TEST_REGISTRATION
struct registration_wrapper { uint32_t magic = 0x466243b1, version = 1; const void* image{}; const void* auxiliary{}; };
// Independently encode one uncompressed, 64-bit NVIDIA PTX fatbinary member.
std::vector<unsigned char> registration_ptx_image(std::string_view source, unsigned major, unsigned minor, unsigned architecture);
#endif

class device_buffer {
public:
    CUdeviceptr value{};
    explicit device_buffer(size_t size) { TEST_CUDA(cuMemAlloc(&value, size)); }
    ~device_buffer() { if (value) cuMemFree(value); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
};

class loaded_program {
    CUmodule module{};
    CUlibrary library{};
public:
    loaded_program(const void* image, bool use_library);
    ~loaded_program();
    loaded_program(const loaded_program&) = delete;
    loaded_program& operator=(const loaded_program&) = delete;
    CUfunction function(const char* name) const;
    bool contains_function(const char* name) const;
    // Contextless kernel routes apply to library loads; module loads return null.
    CUkernel library_kernel(const char* name) const;
    CUdeviceptr global(const char* name, size_t expected_size) const;
};

// Format-independent Driver API contracts need a device, but no shader compiler
// or executable-input probe. A boundary slice that owns executable input may
// supply an independent probe; slice failures never trigger device fallback.
int run_test(int argc, char** argv, void (*execute)(), unsigned context_flags = 0,
             void (*probe)() = nullptr);
// Compiler/linker boundary slices can reuse the same independent execution
// prerequisite as embedded-input slices, without adding unrelated fixtures.
void probe_execution();
// Fatal-device witnesses own a disposable CTest process. Report and exit without
// synchronizing or destroying a faulted context, even when an assertion fails.
[[noreturn]] void run_isolated_fault_test(int argc, char** argv, void (*execute)(),
                                        void (*probe)() = nullptr);

// Execution slices reuse one oracle unchanged by every importer and loader.
// Context flags declare prerequisites such as host mapping for the slice itself.
int run_test(int argc, char** argv, const fixture_data& fixtures,
             void (*execute)(const loaded_program&), unsigned load_repetitions = 1,
             unsigned context_flags = 0);
void expect_words(std::span<const uint32_t> actual, std::span<const uint32_t> expected,
                  const std::string& context);

// Shared storage/launch checks; the owning slice supplies its independent rows
// and the only permitted equivalences for numerical results.
template<class Word, size_t Fields, size_t InputFields, class Oracle, class Equivalent>
void execute_numerical_rows(const loaded_program& program, const char* kernel,
    const std::vector<std::array<Word, InputFields>>& values, Oracle oracle, Equivalent equivalent) {
    constexpr unsigned guards = 12;
    const Word guard = numerical_output_guard<Word>;
    const auto capacity = unsigned(values.size());
    std::vector<Word> input(size_t(capacity) * InputFields + guards * 2, guard), preserved(input.size());
    std::vector<Word> initial((size_t(capacity) + 3) * Fields + guards * 2, guard), complete(initial), actual(initial.size());
    for (unsigned i = 0; i < capacity; ++i) {
        std::copy(values[i].begin(), values[i].end(), input.begin() + guards + size_t(i) * InputFields);
        const auto row = oracle(values[i]);
        static_assert(std::tuple_size_v<decltype(row)> == Fields);
        std::copy(row.begin(), row.end(), complete.begin() + guards + size_t(i) * Fields);
    }
    device_buffer source(input.size() * sizeof(Word)), output(initial.size() * sizeof(Word));
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * sizeof(Word)));
    auto from = source.value + guards * sizeof(Word), to = output.value + guards * sizeof(Word);
    for (unsigned threads : {33u, 128u}) for (unsigned count : {0u, 1u, 37u, capacity}) {
        TEST_CUDA(cuMemcpyHtoD(output.value, initial.data(), initial.size() * sizeof(Word)));
        void* arguments[]{&to, &from, &count};
        TEST_CUDA(cuLaunchKernel(program.function(kernel), (capacity + threads - 1) / threads, 1, 1,
            threads, 1, 1, 0, nullptr, arguments, nullptr));
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * sizeof(Word)));
        for (size_t i = 0; i < actual.size(); ++i) {
            const bool active = i >= guards && i < guards + size_t(count) * Fields;
            const auto expected = active ? complete[i] : guard;
            bool matches = actual[i] == expected;
            if (active) {
                if constexpr (std::is_invocable_r_v<bool, Equivalent, Word, Word, size_t>)
                    matches = equivalent(actual[i], expected, (i - guards) % Fields);
                else matches = equivalent(actual[i], expected);
            }
            if (matches) continue;
            const auto position = active ? " record=" + std::to_string((i - guards) / Fields)
                + " field=" + std::to_string((i - guards) % Fields) : " guard=" + std::to_string(i);
            throw std::runtime_error(std::string(kernel) + position + " threads=" + std::to_string(threads)
                + " count=" + std::to_string(count) + " got=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected));
        }
        TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * sizeof(Word)));
        require(preserved == input, std::string(kernel) + " preserves every source byte and guard");
    }
}

inline void virtual_runtime(cudaError_t result) {
    require(result == cudaSuccess, std::string("virtual memory Runtime: ") + cudaGetErrorString(result));
}
inline CUmemAllocationProp virtual_memory_properties() {
    CUmemAllocationProp properties{}; properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    TEST_CUDA(cuCtxGetDevice(&properties.location.id));
    return properties;
}
inline size_t virtual_memory_granularity(const CUmemAllocationProp& properties) {
    size_t result{}; TEST_CUDA(cuMemGetAllocationGranularity(&result, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    require(result && result <= std::numeric_limits<size_t>::max() / 16, "bounded virtual memory granularity");
    return result;
}
struct virtual_physical_memory {
    CUmemGenericAllocationHandle value{};
    size_t bytes;
    virtual_physical_memory(size_t size, const CUmemAllocationProp& properties) : bytes(size) {
        struct { uint64_t before = 0x19375928abcdef01; CUmemGenericAllocationHandle value{}; uint64_t after = 0xfedcba0129183745; } output;
        TEST_CUDA(cuMemCreate(&output.value, bytes, &properties, 0));
        value = output.value;
        require(value && output.before == 0x19375928abcdef01 && output.after == 0xfedcba0129183745, "physical allocation output guards");
    }
    ~virtual_physical_memory() { if (value) cuMemRelease(value); }
    virtual_physical_memory(const virtual_physical_memory&) = delete;
    void close() { TEST_CUDA(cuMemRelease(value)); value = 0; }
};
struct virtual_memory_reservation {
    CUdeviceptr value{};
    size_t bytes;
    explicit virtual_memory_reservation(size_t size) : bytes(size) { TEST_CUDA(cuMemAddressReserve(&value, bytes, 0, 0, 0)); }
    ~virtual_memory_reservation() { if (value) cuMemAddressFree(value, bytes); }
    virtual_memory_reservation(const virtual_memory_reservation&) = delete;
};
struct virtual_memory_mapping {
    CUdeviceptr value{};
    size_t bytes;
    virtual_memory_mapping(CUdeviceptr pointer, size_t size, CUmemGenericAllocationHandle physical) : value(pointer), bytes(size) {
        TEST_CUDA(cuMemMap(value, bytes, 0, physical, 0));
    }
    ~virtual_memory_mapping() { if (value) cuMemUnmap(value, bytes); }
    virtual_memory_mapping(const virtual_memory_mapping&) = delete;
    void grant(CUmemLocation location, CUmemAccess_flags flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE) const {
        CUmemAccessDesc access{location, flags}; TEST_CUDA(cuMemSetAccess(value, bytes, &access, 1));
    }
    void close() { TEST_CUDA(cuMemUnmap(value, bytes)); value = 0; }
};
inline std::vector<unsigned char> virtual_memory_pattern(size_t bytes, unsigned seed) {
    std::vector<unsigned char> values(bytes);
    for (size_t i = 0; i < bytes; ++i) values[i] = static_cast<unsigned char>((i * 73 + seed * 37) ^ (i >> 7));
    return values;
}
inline void virtual_memory_write(CUdeviceptr pointer, std::span<const unsigned char> values, bool runtime = false) {
    if (runtime) virtual_runtime(cudaMemcpy(reinterpret_cast<void*>(uintptr_t(pointer)), values.data(), values.size(), cudaMemcpyHostToDevice));
    else TEST_CUDA(cuMemcpyHtoD(pointer, values.data(), values.size()));
}
inline void virtual_memory_expect(CUdeviceptr pointer, std::span<const unsigned char> expected, const char* label, bool runtime = false) {
    std::vector<unsigned char> values(expected.size() + 32, 0xa7);
    if (runtime) virtual_runtime(cudaMemcpy(values.data() + 16, reinterpret_cast<void*>(uintptr_t(pointer)), expected.size(), cudaMemcpyDeviceToHost));
    else TEST_CUDA(cuMemcpyDtoH(values.data() + 16, pointer, expected.size()));
    for (size_t i = 0; i < values.size(); ++i) {
        const auto wanted = i < 16 || i >= expected.size() + 16 ? 0xa7 : expected[i - 16];
        if (values[i] != wanted) require(false, std::string(label) + " byte " + std::to_string(i));
    }
}
inline void require_virtual_memory() {
    CUdevice device{}; int supported{};
    TEST_CUDA(cuCtxGetDevice(&device));
    TEST_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));
    require_capability(supported != 0, "virtual memory requires advertised VMM support");
}

inline void image_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("image Runtime: ") + cudaGetErrorString(status));
}
// Shared allocation/descriptor mechanics; each slice constructs its own pixels,
// coordinates and expected results independently of the device instructions.
struct image_test_storage {
    CUarray array{};
    CUdeviceptr pointer{};
    CUresourcetype type;
    CUarray_format format;
    unsigned width, height, channels, bits;
    cudaChannelFormatKind kind;
    size_t pitch;
    std::vector<unsigned char> bytes;
    image_test_storage(CUresourcetype resource_type, CUarray_format component_format,
        cudaChannelFormatKind component_kind, unsigned component_bits, unsigned component_count,
        unsigned columns, unsigned rows, bool runtime, unsigned array_flags = 0)
        : type(resource_type), format(component_format), width(columns), height(rows), channels(component_count),
          bits(component_bits), kind(component_kind), pitch(size_t(width) * channels * bits / 8) {
        if (type == CU_RESOURCE_TYPE_ARRAY) {
            if (runtime) {
                auto channel = channel_descriptor(); cudaArray_t result{};
                image_runtime(cudaMallocArray(&result, &channel, width, height,
                    array_flags & CUDA_ARRAY3D_SURFACE_LDST ? cudaArraySurfaceLoadStore : 0)); array = reinterpret_cast<CUarray>(result);
            } else {
                CUDA_ARRAY3D_DESCRIPTOR descriptor{width, height, 0, format, channels, array_flags};
                TEST_CUDA(cuArray3DCreate(&array, &descriptor));
            }
        } else if (type == CU_RESOURCE_TYPE_PITCH2D) {
            TEST_CUDA(cuMemAllocPitch(&pointer, &pitch, pitch, height, 16));
        } else TEST_CUDA(cuMemAlloc(&pointer, pitch * height));
        bytes.assign(pitch * height, 0xa7);
    }
    ~image_test_storage() { if (array) cuArrayDestroy(array); if (pointer) cuMemFree(pointer); }
    image_test_storage(const image_test_storage&) = delete;
    cudaChannelFormatDesc channel_descriptor() const {
        return {int(bits), channels >= 2 ? int(bits) : 0, channels == 4 ? int(bits) : 0, channels == 4 ? int(bits) : 0, kind};
    }
    CUDA_RESOURCE_DESC driver_descriptor() const {
        CUDA_RESOURCE_DESC result{}; result.resType = type;
        if (array) result.res.array.hArray = array;
        else if (type == CU_RESOURCE_TYPE_LINEAR) result.res.linear = {pointer, format, channels, bytes.size()};
        else result.res.pitch2D = {pointer, format, channels, width, height, pitch};
        return result;
    }
    cudaResourceDesc runtime_descriptor() const {
        cudaResourceDesc result{}; result.resType = static_cast<cudaResourceType>(type);
        if (array) result.res.array.array = reinterpret_cast<cudaArray_t>(array);
        else if (type == CU_RESOURCE_TYPE_LINEAR) result.res.linear = {reinterpret_cast<void*>(uintptr_t(pointer)), channel_descriptor(), bytes.size()};
        else result.res.pitch2D = {reinterpret_cast<void*>(uintptr_t(pointer)), channel_descriptor(), width, height, pitch};
        return result;
    }
    void transfer(std::vector<unsigned char>& host, bool upload) const {
        if (pointer) {
            if (upload) TEST_CUDA(cuMemcpyHtoD(pointer, host.data(), host.size()));
            else TEST_CUDA(cuMemcpyDtoH(host.data(), pointer, host.size()));
        } else {
            CUDA_MEMCPY2D copy{}; copy.WidthInBytes = pitch; copy.Height = height;
            copy.srcMemoryType = upload ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_ARRAY;
            copy.dstMemoryType = upload ? CU_MEMORYTYPE_ARRAY : CU_MEMORYTYPE_HOST;
            copy.srcHost = host.data(); copy.dstHost = host.data(); copy.srcArray = array; copy.dstArray = array;
            copy.srcPitch = copy.dstPitch = pitch; TEST_CUDA(cuMemcpy2D(&copy));
        }
    }
    void upload() { transfer(bytes, true); }
    void verify() const {
        std::vector<unsigned char> actual(bytes.size(), 0x63); transfer(actual, false);
        require(actual == bytes, "image reads preserve every backing byte and pitched padding");
    }
};
struct image_test_texture {
    CUtexObject value{};
    image_test_texture(const image_test_storage& storage, bool runtime, CUaddress_mode address = CU_TR_ADDRESS_MODE_CLAMP,
        CUfilter_mode filter = CU_TR_FILTER_MODE_POINT, bool normalized_coordinates = false, bool normalized_read = false,
        std::array<float, 4> border = {}) {
        if (runtime) {
            auto resource = storage.runtime_descriptor(); cudaTextureDesc sampler{};
            for (auto& mode : sampler.addressMode) mode = static_cast<cudaTextureAddressMode>(address);
            sampler.filterMode = static_cast<cudaTextureFilterMode>(filter);
            sampler.readMode = normalized_read ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
            sampler.normalizedCoords = normalized_coordinates; sampler.maxAnisotropy = 1;
            std::copy(border.begin(), border.end(), sampler.borderColor);
            cudaTextureObject_t result{}; image_runtime(cudaCreateTextureObject(&result, &resource, &sampler, nullptr)); value = result;
        } else {
            auto resource = storage.driver_descriptor(); CUDA_TEXTURE_DESC sampler{};
            for (auto& mode : sampler.addressMode) mode = address;
            sampler.filterMode = filter; sampler.maxAnisotropy = 1;
            sampler.flags = (normalized_coordinates ? CU_TRSF_NORMALIZED_COORDINATES : 0) | (normalized_read ? 0 : CU_TRSF_READ_AS_INTEGER);
            std::copy(border.begin(), border.end(), sampler.borderColor);
            TEST_CUDA(cuTexObjectCreate(&value, &resource, &sampler, nullptr));
        }
    }
    ~image_test_texture() { if (value) cuTexObjectDestroy(value); }
    image_test_texture(const image_test_texture&) = delete;
};
#endif
}
