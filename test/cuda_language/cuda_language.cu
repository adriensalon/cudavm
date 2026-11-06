namespace language_fixture {
template<unsigned N> struct constant_sum {
    static constexpr unsigned value = N + constant_sum<N - 1>::value;
};
template<> struct constant_sum<0> { static constexpr unsigned value = 0; };
template<class T> struct selection { static constexpr unsigned value = 3; };
template<class T> struct selection<T*> { static constexpr unsigned value = 17; };
template<> struct selection<unsigned> { static constexpr unsigned value = 29; };

__host__ __device__ constexpr unsigned target_overload(unsigned value) { return value + 31u; }
__device__ unsigned target_overload(unsigned long long value) { return unsigned(value) + 53u; }
__host__ __device__ constexpr unsigned dual_target(unsigned value) {
#if defined(__CUDA_ARCH__)
    return value * 7u + 11u;
#else
    return value * 7u + 0x80000000u;
#endif
}
__device__ unsigned overloaded(unsigned value) { return value ^ 0x193759bdu; }
__device__ unsigned overloaded(int value) { return static_cast<unsigned>(value) + 43u; }
template<class T> __device__ unsigned templated(T value) {
    if constexpr (sizeof(T) == 8) return unsigned(value) ^ unsigned(value >> 32);
    else return unsigned(value) * 13u;
}
template<> __device__ unsigned templated<unsigned>(unsigned value) { return value * 19u; }
__device__ unsigned default_argument(unsigned value, unsigned bias = constant_sum<7>::value) { return value + bias; }
enum class operation : unsigned { add, invert, scale };
struct alignas(16) record {
    unsigned first;
    unsigned second = 37;
    unsigned third = constant_sum<5>::value;
    unsigned fourth = 0;
};
union word { unsigned value; unsigned other; };
static_assert(constant_sum<7>::value == 28, "recursive template constant");
static_assert(selection<unsigned>::value == 29 && selection<unsigned*>::value == 17
    && selection<int>::value == 3, "specialization selection");
static_assert(sizeof(record) == 16 && alignof(record) == 16, "aggregate layout");
}

extern "C" __global__ void language_results(unsigned* output, const unsigned* input, unsigned count) {
    using namespace language_fixture;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = input[i];
    unsigned* row = output + i * 13;
    row[0] = target_overload(value);
    row[1] = dual_target(value);
    row[2] = overloaded(value);
    row[3] = overloaded(int(value & 0x7fffffffu));
    row[4] = templated(value);
    row[5] = templated((static_cast<unsigned long long>(value) << 32) | (value ^ 0xabcdef01u));
    row[6] = selection<decltype(input)>::value + selection<unsigned>::value + selection<int>::value;
    const record aggregate{value};
    row[7] = aggregate.first + aggregate.second + aggregate.third + aggregate.fourth;
    const record zero{};
    row[8] = zero.first + zero.second + zero.third + zero.fourth;
    const word initialized{value ^ 0x76543210u};
    row[9] = initialized.value;
    const auto captured = [value](unsigned other) { return value * 23u + other; };
    row[10] = captured(default_argument(i));
    switch (static_cast<operation>(i % 3)) {
    case operation::add: row[11] = value + 59u; break;
    case operation::invert: row[11] = ~value; break;
    case operation::scale: row[11] = value * 61u; break;
    }
    row[12] = target_overload(static_cast<unsigned long long>(value));
}
