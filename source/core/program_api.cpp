#include "program.hpp"
#include "nvrtc_frontend.hpp"
#include "ptx_import.hpp"
#include "native_emission.hpp"
#include "device_numeric_ir.hpp"
#include "spirv_emitter.hpp"
#include "dxil_emitter.hpp"
#include "spirv_artifact.hpp"
#include "dxil_artifact.hpp"
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

namespace cudavm {
namespace {
struct compiler_error : std::runtime_error {
    compiler_status status;
    compiler_error(compiler_status value, const char* text) : std::runtime_error(text), status(value) {}
};
void require(bool condition, const char* text) {
    if (!condition) throw compiler_error(compiler_status::invalid, text);
}
std::string program_stage(const llvm::Module& module) {
    auto* stage = module.getNamedMetadata("cudavm.ir.stage");
    if (!stage || stage->getNumOperands() != 1 || stage->getOperand(0)->getNumOperands() != 1) return {};
    auto* value = llvm::dyn_cast<llvm::MDString>(stage->getOperand(0)->getOperand(0));
    return value ? value->getString().str() : std::string();
}
std::string bitcode(const llvm::Module& module) {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    llvm::WriteBitcodeToFile(module, stream);
    return bytes;
}
std::unique_ptr<llvm::Module> read_bitcode(llvm::LLVMContext& context, llvm::StringRef bytes, llvm::StringRef name) {
    auto parsed = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bytes, name), context);
    if (!parsed) throw std::runtime_error("invalid cudavm IR: " + llvm::toString(parsed.takeError()));
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    if (llvm::verifyModule(**parsed, &stream))
        throw compiler_error(compiler_status::invalid, diagnostic.c_str());
    return std::move(*parsed);
}
void copy_program_metadata(const compiler_program& input, compiler_program& output) {
    output.architecture = input.architecture;
    output.lowered_names = input.lowered_names;
}
std::unique_ptr<compiler_program> copy_program(const compiler_program& input) {
    auto output = std::make_unique<compiler_program>();
    output->module = read_bitcode(output->context, bitcode(*input.module), input.module->getModuleIdentifier());
    copy_program_metadata(input, *output);
    return output;
}
template<class Function> compiler_status program_api(compiler_result** output, Function function) noexcept {
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
            status = error.unsupported ? compiler_status::unsupported : compiler_status::invalid;
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

std::unique_ptr<compiler_program> import_compiler_program(const compiler_input_request& request, compiler_result& result) {
    require(request.abi_version == 1, "compiler input requires abi_version 1");
    auto output = std::make_unique<compiler_program>();
    switch (request.kind) {
    case compiler_input_kind::source: {
        auto* source = request.source;
        require(source && source->abi_version == 1 && source->source && source->name && source->relocatable <= 1
            && (!source->header_count || source->headers), "invalid compiler source request");
        nvrtc_options settings;
        if (source->standard) settings.standard = source->standard;
        if (source->architecture) settings.architecture = source->architecture;
        settings.relocatable = source->relocatable != 0;
        append_compiler_option_strings(source->include_count, source->includes, settings.includes);
        append_compiler_option_strings(source->define_count, source->defines, settings.defines);
        append_compiler_option_strings(source->undefine_count, source->undefines, settings.undefines);
        std::vector<std::pair<std::string, std::string>> headers;
        for (size_t i = 0; i < source->header_count; ++i) {
            require(source->headers[i].name && source->headers[i].contents, "invalid compiler source header");
            headers.emplace_back(source->headers[i].name, source->headers[i].contents);
        }
        std::vector<std::string> expressions;
        append_compiler_option_strings(source->expression_count, source->expressions, expressions);
        auto imported = import_cuda_source(output->context, source->source, source->name, headers, expressions, settings, result.diagnostics);
        output->module = std::move(imported.module);
        output->architecture = settings.architecture;
        for (auto& [expression, name] : imported.lowered_names) output->lowered_names.emplace(expression, std::move(name));
        // Source programs remain relocatable until the caller requests closure.
        compiler_options options;
        options.input = source->name; options.bitcode = source->name;
        options.standard = settings.standard; options.architecture = settings.architecture;
        mark_translation_unit(*output->module, options, {});
        break;
    }
    case compiler_input_kind::ptx:
        require(request.data && request.size, "empty PTX input");
        output->module = import_ptx(output->context, {request.data, request.size});
        if (request.name) output->module->setModuleIdentifier(request.name);
        if (auto* target = output->module->getNamedMetadata("cudavm.import.architecture"))
            output->architecture = llvm::cast<llvm::MDString>(target->getOperand(0)->getOperand(0))->getString().str();
        break;
    case compiler_input_kind::cudavm_ir: {
        require(request.data && request.size, "empty cudavm IR input");
        output->module = read_bitcode(output->context, {request.data, request.size}, request.name ? request.name : "input.bc");
        auto stage = program_stage(*output->module);
        require(stage == "clang.cuda.tu" || stage == "clang.cuda.linked" || stage == "cudavm.portable", "unsupported cudavm IR stage");
        if (auto* abi = output->module->getNamedMetadata("cudavm.source.abi"); abi && abi->getNumOperands()) {
            auto* row = abi->getOperand(0);
            require(row->getNumOperands() == 6 && llvm::isa<llvm::MDString>(row->getOperand(2)), "invalid source ABI metadata");
            output->architecture = llvm::cast<llvm::MDString>(row->getOperand(2))->getString().str();
        }
        break;
    }
    case compiler_input_kind::nvvm: throw compiler_error(compiler_status::unsupported, "NVVM import is not implemented");
    case compiler_input_kind::nvltoir: throw compiler_error(compiler_status::unsupported, "NVIDIA LTO-IR import is not implemented");
    case compiler_input_kind::cubin: throw compiler_error(compiler_status::unsupported, "cubin import is not implemented");
    default: throw compiler_error(compiler_status::invalid, "unknown compiler input kind");
    }
    result.lowered_names = output->lowered_names;
    return output;
}

std::unique_ptr<compiler_program> link_compiler_programs(size_t count, const compiler_program* const* programs, compiler_result& result) {
    require(count && programs, "IR linking requires input programs");
    compiler_options options;
    auto output = std::make_unique<compiler_program>();
    for (size_t i = 0; i < count; ++i) {
        require(programs[i] && programs[i]->module, "null IR link input");
        const auto& input = *programs[i];
        if (i == 0) options.architecture = output->architecture = input.architecture;
        require(input.architecture == output->architecture, "incompatible IR link architectures");
        for (const auto& [expression, name] : input.lowered_names) {
            auto [found, added] = output->lowered_names.emplace(expression, name);
            require(added || found->second == name, "conflicting lowered name expressions");
        }
        // Imported PTX has already entered common IR. Give it an archive member
        // identity so the same linker can retain/replay it on a later device link.
        if (program_stage(*input.module) == "cudavm.portable" && !input.module->getNamedMetadata("cudavm.link.archive.v1")) {
            auto member = copy_program(input);
            compiler_options settings;
            settings.input = settings.bitcode = input.module->getModuleIdentifier();
            settings.standard = "c++17"; settings.architecture = input.architecture;
            mark_translation_unit(*member->module, settings, {});
            options.ir_inputs.push_back({bitcode(*member->module), settings.input});
        } else options.ir_inputs.push_back({bitcode(*input.module), input.module->getModuleIdentifier()});
    }
    try { output->module = link_device(output->context, options); }
    catch (...) { result.diagnostics += options.diagnostics; throw; }
    result.diagnostics += options.diagnostics;
    result.lowered_names = output->lowered_names;
    return output;
}

std::unique_ptr<compiler_program> prepare_compiler_program(const compiler_program& input, const compiler_pipeline_request& request) {
    require(request.abi_version == 1 && request.numerical_helpers <= 1 && request.lower_semantics <= 1
        && request.optimization_level <= 3, "invalid compiler pipeline request");
    auto output = copy_program(input);
    if (request.numerical_helpers) link_device_numeric_ir(*output->module);
    if (request.lower_semantics && program_stage(*output->module) != "cudavm.portable") {
        if (program_stage(*output->module) == "clang.cuda.tu") {
            const compiler_program* programs[] = {output.get()};
            compiler_result diagnostics;
            output = link_compiler_programs(1, programs, diagnostics);
        }
        prepare_portable(*output->module);
    }
    if (request.optimization_level) {
        require(program_stage(*output->module) == "cudavm.portable", "optimization requires semantic lowering");
        llvm::LoopAnalysisManager loops;
        llvm::FunctionAnalysisManager functions;
        llvm::CGSCCAnalysisManager calls;
        llvm::ModuleAnalysisManager modules;
        llvm::PassBuilder builder;
        builder.registerModuleAnalyses(modules); builder.registerCGSCCAnalyses(calls);
        builder.registerFunctionAnalyses(functions); builder.registerLoopAnalyses(loops);
        builder.crossRegisterProxies(loops, functions, calls, modules);
        auto level = request.optimization_level == 1 ? llvm::OptimizationLevel::O1
            : request.optimization_level == 2 ? llvm::OptimizationLevel::O2 : llvm::OptimizationLevel::O3;
        builder.buildPerModuleDefaultPipeline(level).run(*output->module, modules);
        require(!llvm::verifyModule(*output->module), "optimization produced invalid IR");
    }
    return output;
}

void emit_compiler_program(const compiler_program& program, const compiler_emit_request& request, compiler_result& result) {
    require(request.abi_version == 1, "compiler emission requires abi_version 1");
    require(uint32_t(request.environment) <= uint32_t(compiler_environment::all), "invalid compiler environment");
    const auto& module = *program.module;
    auto architecture = request.architecture ? request.architecture : program.architecture.c_str();
    auto portable = program_stage(module) == "cudavm.portable";
    result.lowered_names = program.lowered_names;
    switch (request.kind) {
    case compiler_output_kind::bitcode: result.output = bitcode(module); return;
    case compiler_output_kind::llvm_ir: {
        llvm::raw_string_ostream stream(result.output); module.print(stream, nullptr); return;
    }
    case compiler_output_kind::ptx:
        result.output = portable ? emit_native(module, {native_target::nvptx, architecture, false, {}}).bytes
            : emit_cuda_ir(module, architecture);
        return;
    case compiler_output_kind::spirv:
    case compiler_output_kind::dxil: {
        auto symbol = compiler_option_text(request.kernel);
        if (symbol.empty()) {
            for (const auto& function : module)
                if (!function.isDeclaration() && (function.hasFnAttribute("cudavm.kernel") || function.getCallingConv() == llvm::CallingConv::PTX_Kernel)) {
                    require(symbol.empty(), "shader output requires one selected kernel"); symbol = function.getName().str();
                }
        }
        require(!symbol.empty(), "shader output requires a kernel");
        if (request.kind == compiler_output_kind::spirv) {
            require(request.environment == compiler_environment::vulkan || request.environment == compiler_environment::opengl
                || request.environment == compiler_environment::opencl, "SPIR-V output requires a shader environment");
            result.output = emit_native_spirv(*lower_native_spirv(module, symbol, request.environment == compiler_environment::opencl));
        } else {
            require(request.environment == compiler_environment::directx && request.block[0] && request.block[1] && request.block[2], "invalid DXIL output request");
            result.output = emit_native_dxil(*lower_native_dxil(module, symbol, {request.block[0], request.block[1], request.block[2]}));
        }
        return;
    }
    case compiler_output_kind::program_image: break;
    default: throw compiler_error(compiler_status::invalid, "unknown compiler output kind");
    }
    if (request.environment == compiler_environment::all || request.environment == compiler_environment::cuda) {
        if (portable) result.output = emit_native(module, {native_target::nvptx, architecture, true, {}}).bytes;
        else {
            result.output = emit_cuda_ir(module, architecture);
            result.output += emit_native_spirv_payload(module);
            result.output += emit_native_dxil_payload(module);
            result.output += emit_native_module(module);
        }
        return;
    }
    result.output = emit_native_module(module);
    std::vector<shader_artifact> artifacts;
    for (const auto& function : module) if (!function.isDeclaration() && (function.hasFnAttribute("cudavm.kernel") || function.getCallingConv() == llvm::CallingConv::PTX_Kernel)) {
        shader_artifact artifact; artifact.symbol = function.getName().str();
        if (request.environment == compiler_environment::directx) {
            auto lowered = lower_native_dxil(module, artifact.symbol);
            std::vector<dxil_variant> variants;
            for (const auto& shape : dxil_block_shapes) {
                lowered->getFunction("main")->addFnAttr("hlsl.numthreads", std::to_string(shape[0]) + "," + std::to_string(shape[1]) + "," + std::to_string(shape[2]));
                variants.push_back({shape, emit_native_dxil(*lowered)});
            }
            artifact.code = encode_dxil_variants(variants);
        } else if (request.environment == compiler_environment::opencl)
            artifact.opencl_code = emit_native_spirv(*lower_native_spirv(module, artifact.symbol, true));
        else artifact.code = emit_native_spirv(*lower_native_spirv(module, artifact.symbol));
        artifacts.push_back(std::move(artifact));
    }
    result.output += request.environment == compiler_environment::directx ? encode_dxil_artifacts(artifacts) : encode_spirv_artifacts(artifacts);
}
}

using namespace cudavm;
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_import_v1(
        const compiler_input_request* request, compiler_program** program, compiler_result** result) noexcept {
    if (program) *program = nullptr;
    return program_api(result, [&](compiler_result& output) {
        require(request && program, "invalid compiler import request");
        *program = import_compiler_program(*request, output).release();
    });
}
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_link_v1(
        size_t count, const compiler_program* const* inputs, compiler_program** program, compiler_result** result) noexcept {
    if (program) *program = nullptr;
    return program_api(result, [&](compiler_result& output) {
        require(program, "null compiler program output");
        *program = link_compiler_programs(count, inputs, output).release();
    });
}
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_prepare_v1(
        const compiler_program* input, const compiler_pipeline_request* request, compiler_program** program, compiler_result** result) noexcept {
    if (program) *program = nullptr;
    return program_api(result, [&](compiler_result& output) {
        require(input && input->module && request && program, "invalid compiler preparation request");
        auto prepared = prepare_compiler_program(*input, *request);
        output.lowered_names = prepared->lowered_names;
        *program = prepared.release();
    });
}
extern "C" CUDAVM_CORE_API compiler_status cudavm_core_program_emit_v1(
        const compiler_program* program, const compiler_emit_request* request, compiler_result** result) noexcept {
    return program_api(result, [&](compiler_result& output) {
        require(program && program->module && request, "invalid compiler emission request");
        emit_compiler_program(*program, *request, output);
    });
}
extern "C" CUDAVM_CORE_API void cudavm_core_program_release_v1(compiler_program* program) noexcept { delete program; }
