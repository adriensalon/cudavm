#include "compiler_api.hpp"
#include "nvrtc_frontend.hpp"
using namespace cudavm;
extern "C" CUDAVM_COMPILER_API compiler_status cudavm_compiler_compile_source_v1(
        const compiler_source_request* request, compiler_result** output) noexcept {
    if (!output) return compiler_status::invalid;
    *output = nullptr;
    try {
        auto result = std::make_unique<compiler_result>();
        auto status = compiler_status::success;
        try {
            if (!request || request->abi_version != 1 || !request->source || !request->name ||
                    (request->header_count && !request->headers) || request->relocatable > 1)
                throw std::runtime_error("invalid compiler source request");
            nvrtc_options settings;
            if (request->standard) settings.standard = request->standard;
            if (request->architecture) settings.architecture = request->architecture;
            settings.relocatable = request->relocatable != 0;
            append_compiler_option_strings(request->include_count, request->includes, settings.includes);
            append_compiler_option_strings(request->define_count, request->defines, settings.defines);
            append_compiler_option_strings(request->undefine_count, request->undefines, settings.undefines);
            std::vector<std::pair<std::string, std::string>> headers;
            for (size_t i = 0; i < request->header_count; ++i) {
                const auto& header = request->headers[i];
                if (!header.name || !header.contents) throw std::runtime_error("invalid compiler source header");
                headers.emplace_back(header.name, header.contents);
            }
            std::vector<std::string> expressions;
            append_compiler_option_strings(request->expression_count, request->expressions, expressions);
            auto compiled = compile_nvrtc_source(request->source, request->name, headers, expressions, settings, result->diagnostics);
            result->output = std::move(compiled.ptx);
            for (auto& [expression, name] : compiled.lowered_names)
                result->lowered_names.emplace(expression, std::move(name));
        } catch (const std::bad_alloc&) { status = compiler_status::out_of_memory; }
        catch (const std::exception& error) {
            status = compiler_status::invalid;
            result->diagnostics += std::string("cudavm NVRTC: ") + error.what() + '\n';
        } catch (...) {
            status = compiler_status::invalid;
            result->diagnostics += "cudavm NVRTC: unknown compiler failure\n";
        }
        *output = result.release();
        return status;
    } catch (const std::bad_alloc&) { return compiler_status::out_of_memory; }
    catch (...) { return compiler_status::invalid; }
}
