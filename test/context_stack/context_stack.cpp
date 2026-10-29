#include "conformance.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

namespace cudavm {
CUcontext current_stack_context() {
    struct guarded_context {
        uint64_t prefix = 0x0123456789abcdef;
        CUcontext value{};
        uint64_t suffix = 0xfedcba9876543210;
    } output;
    CONFORMANCE_CUDA(cuCtxGetCurrent(&output.value));
    require(output.prefix == 0x0123456789abcdef && output.suffix == 0xfedcba9876543210,
        "current-context output guards");
    return output.value;
}
void expect_stack_context(CUcontext expected, const std::string& operation) {
    require(current_stack_context() == expected, operation + ": current context");
}
void expect_stack_pop(CUcontext expected, CUcontext exposed) {
    CUcontext popped{};
    CONFORMANCE_CUDA(cuCtxPopCurrent(&popped));
    require(popped == expected, "pop returns the removed top context");
    expect_stack_context(exposed, "pop exposes predecessor");
}
struct stack_contexts {
    CUcontext original{};
    CUdevice device{};
    std::vector<CUcontext> owned;
    stack_contexts() {
        original = current_stack_context();
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    }
    ~stack_contexts() {
        for (auto context : owned) cuCtxDestroy(context);
        cuCtxSetCurrent(original);
    }
    CUcontext create() {
        const auto preceding = current_stack_context();
        CUcontext context{};
        CONFORMANCE_CUDA(cuCtxCreate(&context, nullptr, 0, device));
        owned.push_back(context);
        expect_stack_context(context, "creation pushes new context");
        expect_stack_pop(context, preceding);
        return context;
    }
};
void context_stack_transitions() {
    stack_contexts contexts;
    const auto a = contexts.create(), b = contexts.create(), c = contexts.create();
    CONFORMANCE_CUDA(cuCtxPushCurrent(a)); expect_stack_context(a, "push a");
    CONFORMANCE_CUDA(cuCtxPushCurrent(b)); expect_stack_context(b, "nested push b");
    CONFORMANCE_CUDA(cuCtxSetCurrent(c)); expect_stack_context(c, "set replaces b");
    expect_stack_pop(c, a);
    CONFORMANCE_CUDA(cuCtxPushCurrent(a)); expect_stack_context(a, "duplicate push a");
    CONFORMANCE_CUDA(cuCtxSetCurrent(b)); expect_stack_context(b, "replace duplicate top");
    CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr)); expect_stack_context(a, "null set removes one top");
    CONFORMANCE_CUDA(cuCtxPushCurrent(c)); expect_stack_context(c, "push c");
    CONFORMANCE_CUDA(cuCtxPushCurrent(b)); expect_stack_context(b, "nested push b after c");
    expect_stack_pop(b, c);
    CONFORMANCE_CUDA(cuCtxSetCurrent(a)); expect_stack_context(a, "replacement creates duplicate a");
    expect_stack_pop(a, a);
    expect_stack_pop(a, contexts.original);
    expect_stack_context(contexts.original, "balanced stack restores runner");
}
template<class Function> void stack_thread(Function execute) {
    std::exception_ptr error;
    std::jthread worker([&] {
        try { execute(); } catch (...) { error = std::current_exception(); }
    });
    worker.join();
    if (error) std::rethrow_exception(error);
}
void context_stack_empty() {
    stack_contexts contexts;
    const auto a = contexts.create(), b = contexts.create();
    stack_thread([&] {
        expect_stack_context(nullptr, "fresh thread");
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
            expect_stack_context(nullptr, "empty set-null is a no-op");
            CUcontext popped{};
            require(cuCtxPopCurrent(&popped) == CUDA_ERROR_INVALID_CONTEXT, "pop of empty stack");
            CUdevice device{};
            require(cuCtxGetDevice(&device) == CUDA_ERROR_INVALID_CONTEXT, "device query with no current context");
            CONFORMANCE_CUDA(cuCtxSetCurrent(a)); expect_stack_context(a, "set on empty stack");
            CONFORMANCE_CUDA(cuCtxSetCurrent(b)); expect_stack_context(b, "set replaces sole binding");
            expect_stack_pop(b, nullptr);
            CONFORMANCE_CUDA(cuCtxPushCurrent(a)); expect_stack_context(a, "push on empty stack");
            CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr)); expect_stack_context(nullptr, "null set removes sole binding");
        }
    });
    expect_stack_context(contexts.original, "worker empty-stack operations preserve parent");
}
struct stack_thread_gate {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned ready = 0;
    bool released = false;
    void enter() {
        std::unique_lock lock(mutex);
        ++ready; changed.notify_all();
        require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return released; }), "context worker release deadline");
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return ready == 2; }), "context workers entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void context_stack_threading() {
    stack_contexts contexts;
    const auto a = contexts.create(), b = contexts.create();
    stack_thread_gate gate;
    std::array<std::exception_ptr, 2> errors;
    auto work = [&](unsigned index, CUcontext first, CUcontext second) {
        try {
            expect_stack_context(nullptr, "independent worker starts empty");
            CONFORMANCE_CUDA(cuCtxPushCurrent(first));
            gate.enter();
            for (unsigned repeat = 0; repeat < 32; ++repeat) {
                expect_stack_context(first, "worker retains its top binding");
                CONFORMANCE_CUDA(cuCtxPushCurrent(second));
                expect_stack_context(second, "worker nested push");
                CONFORMANCE_CUDA(cuCtxSetCurrent(first));
                expect_stack_pop(first, first);
            }
            expect_stack_pop(first, nullptr);
        } catch (...) { errors[index] = std::current_exception(); }
    };
    std::jthread first([&] { work(0, a, b); }), second([&] { work(1, b, a); });
    struct release_workers { stack_thread_gate& gate; ~release_workers() { gate.release(); } } cleanup{gate};
    gate.await();
    expect_stack_context(contexts.original, "workers preserve parent binding");
    CONFORMANCE_CUDA(cuCtxPushCurrent(b));
    expect_stack_context(b, "parent binds context also used in worker");
    gate.release();
    for (unsigned repeat = 0; repeat < 32; ++repeat) expect_stack_context(b, "parent binding under concurrent worker transitions");
    first.join(); second.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    expect_stack_pop(b, contexts.original);
}
void context_stack_transfer() {
    stack_contexts contexts;
    const auto owned = contexts.create();
    for (unsigned generation = 0; generation < 3; ++generation) {
        CUdeviceptr allocation{};
        std::vector<unsigned char> expected(257), actual(289, 0xd3);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<unsigned char>(i * 73 + generation * 31);
        stack_thread([&] {
            expect_stack_context(nullptr, "receiving thread starts empty");
            CONFORMANCE_CUDA(cuCtxPushCurrent(owned));
            expect_stack_context(owned, "popped context moves to receiving thread");
            CONFORMANCE_CUDA(cuMemAlloc(&allocation, expected.size()));
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation, expected.data(), expected.size()));
            expect_stack_pop(owned, nullptr);
        });
        expect_stack_context(contexts.original, "context transfer preserves parent binding");
        CONFORMANCE_CUDA(cuCtxPushCurrent(owned));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, allocation, expected.size()));
        require(std::equal(expected.begin(), expected.end(), actual.begin() + 16), "transferred context retains every allocation byte");
        for (size_t i = 0; i < actual.size(); ++i)
            if (i < 16 || i >= 16 + expected.size()) require(actual[i] == 0xd3, "transferred-context readback guards");
        CONFORMANCE_CUDA(cuMemFree(allocation));
        expect_stack_pop(owned, contexts.original);
    }
}
void context_stack_validation() {
    stack_contexts contexts;
    const auto owned = contexts.create();
    CONFORMANCE_CUDA(cuCtxPushCurrent(owned));
    require(cuCtxGetCurrent(nullptr) == CUDA_ERROR_INVALID_VALUE, "null current-context output");
    expect_stack_context(owned, "rejected current query preserves binding");
    require(cuCtxGetDevice(nullptr) == CUDA_ERROR_INVALID_VALUE, "null current-device output");
    expect_stack_context(owned, "rejected device query preserves binding");
    require(cuCtxPushCurrent(nullptr) == CUDA_ERROR_INVALID_CONTEXT, "push requires a context");
    expect_stack_context(owned, "rejected push preserves stack");
    expect_stack_pop(owned, contexts.original);
}
std::string_view context_stack_case = "all";
void execute_context_stack() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"transitions", context_stack_transitions}, {"empty", context_stack_empty}, {"threading", context_stack_threading},
        {"transfer", context_stack_transfer}, {"validation", context_stack_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (context_stack_case == "all" || context_stack_case == name) {
        found = true;
        std::cout << "context_stack=" << name << std::endl;
        execute();
    }
    require(found, "unknown context_stack case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::context_stack_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_context_stack);
}
