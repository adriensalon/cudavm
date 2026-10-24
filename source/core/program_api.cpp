#include "program_api.hpp"
#include "program.hpp"
#include "ptx_import.hpp"

namespace cudavm {
namespace {
template<class Function> compiler_status run_compiler_program_api(compiler_result** output, Function function) noexcept {
    if (!output) return compiler_status::invalid;
    *output = nullptr;
    try {
        auto result = std::make_unique<compiler_result>();
        auto status = compiler_status::success;
        try {
            std::lock_guard lock(compiler_mutex());
            function(*result);
        } catch (const compiler_error& error) { status = error.status; result->diagnostics += error.what(); }
        catch (const ptx_import_error& error) {
            status = error.unsupported_version ? compiler_status::unsupported_ptx_version
                : error.architecture_mismatch ? compiler_status::architecture_mismatch
                : error.unsupported ? compiler_status::unsupported : compiler_status::invalid;
            result->diagnostics += error.what();
        } catch (const std::bad_alloc&) { status = compiler_status::out_of_memory; }
        catch (const std::exception& error) { status = compiler_status::invalid; result->diagnostics += error.what(); }
        catch (...) { status = compiler_status::invalid; result->diagnostics += "unknown compiler failure"; }
        if (status != compiler_status::success) { result->output.clear(); result->lowered_names.clear(); }
        *output = result.release();
        return status;
    } catch (const std::bad_alloc&) { return compiler_status::out_of_memory; }
    catch (...) { return compiler_status::invalid; }
}
}
}

using namespace cudavm;
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_import_v1(
        const compiler_input_request* request, compiler_program** program, compiler_result** result) noexcept {
    if (program) *program = nullptr;
    return run_compiler_program_api(result, [&](compiler_result& output) {
        require_compiler_request(request && program, "invalid compiler import request");
        *program = import_compiler_program(*request, output).release();
    });
}
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_link_v1(
        size_t count, const compiler_program* const* inputs, compiler_program** program, compiler_result** result) noexcept {
    if (program) *program = nullptr;
    return run_compiler_program_api(result, [&](compiler_result& output) {
        require_compiler_request(program, "null compiler program output");
        *program = link_compiler_programs(count, inputs, output).release();
    });
}
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_prepare_v1(
        const compiler_program* input, const compiler_pipeline_request* request, compiler_program** program, compiler_result** result) noexcept {
    if (program) *program = nullptr;
    return run_compiler_program_api(result, [&](compiler_result& output) {
        require_compiler_request(input && input->module && request && program, "invalid compiler preparation request");
        auto prepared = prepare_compiler_program(*input, *request);
        output.lowered_names = prepared->lowered_names;
        *program = prepared.release();
    });
}
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_emit_v1(
        const compiler_program* program, const compiler_emit_request* request, compiler_result** result) noexcept {
    return run_compiler_program_api(result, [&](compiler_result& output) {
        require_compiler_request(program && program->module && request, "invalid compiler emission request");
        emit_compiler_program(*program, *request, output);
    });
}
extern "C" CUDAVM_CORE_API void cudavm_core_program_release_v1(compiler_program* program) noexcept { delete program; }
