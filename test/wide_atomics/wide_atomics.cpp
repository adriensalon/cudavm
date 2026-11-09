#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
std::array<uint32_t, 4> wide_ticket(uint32_t ticket) {
    return {~ticket, ticket, ticket * 17u, ticket ^ 0x9e3779b9u};
}
void execute_wide_atomics(const loaded_program& program) {
    constexpr unsigned guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    // Independent scalar operations distinguish a comparison of all 128 bits
    // from an implementation that compares only one constituent word.
    for (unsigned changed = 0; changed < 4; ++changed) {
        std::vector<uint32_t> state(guards * 2 + 4, guard), output(guards * 2 + 16, guard), actual(output.size());
        const std::array<uint32_t, 4> initial{0, UINT32_MAX, 0x80000000, 0x12345678}, desired{UINT32_MAX, 0, 0x76543210, 1};
        std::copy(initial.begin(), initial.end(), state.begin() + guards);
        std::vector<uint32_t> operations(guards * 2 + 32, guard);
        std::array<uint32_t, 4> value = initial;
        for (unsigned op = 0; op < 4; ++op) {
            auto compare = value; if (!(op & 1)) compare[changed] ^= 1;
            auto replacement = desired; replacement[(changed + 1) % 4] ^= op;
            std::copy(compare.begin(), compare.end(), operations.begin() + guards + op * 8);
            std::copy(replacement.begin(), replacement.end(), operations.begin() + guards + op * 8 + 4);
            std::copy(value.begin(), value.end(), output.begin() + guards + op * 4);
            if (compare == value) value = replacement;
        }
        device_buffer storage(state.size() * 4), result(output.size() * 4), inputs(operations.size() * 4);
        auto pointer = storage.value + guards * 4, to = result.value + guards * 4, from = inputs.value + guards * 4;
        require(pointer % 16 == 0, "128-bit CAS target alignment");
        std::vector<uint32_t> initialized(output.size(), guard), preserved(operations.size());
        CONFORMANCE_CUDA(cuMemcpyHtoD(storage.value, state.data(), state.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(result.value, initialized.data(), initialized.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(inputs.value, operations.data(), operations.size() * 4));
        unsigned count = 4; void* args[]{&to, &pointer, &from, &count};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("wide_serial"), 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), result.value, actual.size() * 4));
        expect_words(actual, output, "128-bit returned words, compare failure/success and output guards");
        std::copy(value.begin(), value.end(), state.begin() + guards);
        std::vector<uint32_t> actual_state(state.size());
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual_state.data(), storage.value, actual_state.size() * 4));
        expect_words(actual_state, state, "128-bit replacement and target neighbor preservation");
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), inputs.value, preserved.size() * 4));
        expect_words(preserved, operations, "wide CAS operation input preservation");
    }
    for (unsigned threads : {1u, 13u, 32u, 64u}) for (unsigned rounds : {0u, 1u, 4u}) {
        constexpr unsigned blocks = 3;
        unsigned count = blocks * threads * rounds;
        std::vector<uint32_t> state(guards * 2 + 4, guard), actual_state(state.size());
        auto initial = wide_ticket(0); std::copy(initial.begin(), initial.end(), state.begin() + guards);
        std::vector<uint32_t> expected(guards * 2 + (count + 2) * 4, guard), actual(expected.size());
        device_buffer storage(state.size() * 4), result(expected.size() * 4);
        auto pointer = storage.value + guards * 4, to = result.value + guards * 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(storage.value, state.data(), state.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(result.value, expected.data(), expected.size() * 4));
        void* args[]{&to, &pointer, &rounds};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("wide_tickets"), blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), result.value, actual.size() * 4));
        std::vector<uint32_t> tickets;
        for (unsigned row = 0; row < count; ++row) {
            auto ticket = actual[guards + row * 4 + 1]; tickets.push_back(ticket);
            auto record = wide_ticket(ticket);
            std::copy(record.begin(), record.end(), expected.begin() + guards + row * 4);
        }
        expect_words(actual, expected, "every observed atomic packet is internally consistent with complete guards");
        std::sort(tickets.begin(), tickets.end());
        for (unsigned rank = 0; rank < count; ++rank) require(tickets[rank] == rank, "one indivisible successful update per wide CAS ticket");
        auto final = wide_ticket(count); std::copy(final.begin(), final.end(), state.begin() + guards);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual_state.data(), storage.value, actual_state.size() * 4));
        expect_words(actual_state, state, "wide atomic final state and allocation neighbors");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_wide_atomics);
}
