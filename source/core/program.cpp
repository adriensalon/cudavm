#include "program.hpp"
#include "nvrtc_frontend.hpp"
#include "ptx_import.hpp"
#include "native_emission.hpp"
#include "device_numeric_ir.hpp"
#include "spirv_emitter.hpp"
#include "dxil_emitter.hpp"
#include "spirv_artifact.hpp"
#include "dxil_artifact.hpp"
#include "program_image.hpp"
#include "module_descriptor.hpp"
#include <optional>
#include <llvm/Config/llvm-config.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

namespace cudavm {
static_assert(LLVM_VERSION_MAJOR == 22 && LLVM_VERSION_MINOR == 1 && LLVM_VERSION_PATCH == 4,
    "review the portable image IR profile when changing LLVM");
namespace {
bool has_ptx_header(std::string_view input) {
    for (;;) {
        auto first = input.find_first_not_of(" \t\r\n");
        if (first == input.npos) return false;
        input.remove_prefix(first);
        if (input.starts_with("//")) {
            auto end = input.find('\n');
            if (end == input.npos) return false;
            input.remove_prefix(end + 1);
        } else if (input.starts_with("/*")) {
            auto end = input.find("*/", 2);
            if (end == input.npos) return false;
            input.remove_prefix(end + 2);
        } else return input.starts_with(".version");
    }
}
std::string image_metadata(const llvm::Module& module) {
    return llvm::formatv("{0}", llvm::json::Value(describe(module))).str();
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
}

std::string program_input_identity(const llvm::Module& module) {
    auto bytes = bitcode(module);
    auto hash = llvm::SHA256::hash(llvm::ArrayRef(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()));
    return module.getModuleIdentifier() + "." + llvm::toHex(hash, true);
}

std::unique_ptr<compiler_program> import_compiler_program(const compiler_input_request& request, compiler_result& result) {
    require_compiler_request(request.abi_version >= 1 && request.abi_version <= 3, "compiler input requires abi_version 1, 2 or 3");
    const char* architecture = request.abi_version >= 2 ? reinterpret_cast<const compiler_input_request_v2&>(request).architecture : nullptr;
    auto relocatable = request.abi_version == 3 ? reinterpret_cast<const compiler_input_request_v3&>(request).relocatable : 0;
    require_compiler_request(relocatable <= 1, "invalid compiler input relocation mode");
    require_compiler_request(!architecture || request.kind == compiler_input_kind::ptx || request.abi_version == 3,
        "compiler input target override requires PTX or a V3 request");
    auto output = std::make_unique<compiler_program>();
    auto kind = request.kind;
    if (kind == compiler_input_kind::automatic) {
        require_compiler_request(request.data && request.size, "empty automatic compiler input");
        auto bytes = std::string_view(request.data, request.size);
        if (has_program_image(bytes)) kind = compiler_input_kind::program_image;
        else if (has_ptx_header(bytes)) kind = compiler_input_kind::ptx;
        else throw compiler_error(compiler_status::unsupported, "unrecognized compiler input (expected cudavm program image or PTX)");
    }
    if ((kind == compiler_input_kind::cubin || kind == compiler_input_kind::nvltoir) && request.data
        && has_program_image(std::string_view(request.data, request.size))) kind = compiler_input_kind::program_image;
    if (request.abi_version == 3 && request.data && request.size && ((kind == compiler_input_kind::ptx && has_program_image({request.data, request.size}))
        || ((kind == compiler_input_kind::cubin || kind == compiler_input_kind::nvltoir) && has_ptx_header({request.data, request.size}))))
        throw compiler_error(compiler_status::input_type_mismatch, "compiler input bytes do not match the declared input type");
    switch (kind) {
    case compiler_input_kind::source: {
        auto* source = request.source;
        require_compiler_request(source && source->abi_version == 1 && source->source && source->name && source->relocatable <= 1
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
            require_compiler_request(source->headers[i].name && source->headers[i].contents, "invalid compiler source header");
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
        options.input = source->name; options.bitcode = program_input_identity(*output->module);
        options.standard = settings.standard; options.architecture = settings.architecture;
        mark_translation_unit(*output->module, options, {});
        break;
    }
    case compiler_input_kind::ptx:
        require_compiler_request(request.data && request.size, "empty PTX input");
        output->module = import_ptx(output->context, {request.data, request.size}, architecture ? architecture : "", relocatable != 0);
        if (request.name) output->module->setModuleIdentifier(request.name);
        if (auto* target = output->module->getNamedMetadata("cudavm.import.architecture"))
            output->architecture = llvm::cast<llvm::MDString>(target->getOperand(0)->getOperand(0))->getString().str();
        break;
    case compiler_input_kind::program_image:
    case compiler_input_kind::cudavm_ir: {
        require_compiler_request(request.data && request.size, "empty cudavm IR input");
        std::string_view bytes(request.data, request.size);
        std::optional<program_image> image;
        if (has_program_image(bytes)) {
            image = decode_program_image(bytes);
            require_compiler_request(image->producer == program_image_ir_profile, "unsupported program image IR profile");
            bytes = image->ir;
            output->architecture = image->architecture;
            output->lowered_names = image->lowered_names;
        } else require_compiler_request(kind != compiler_input_kind::program_image, "invalid program image magic");
        output->module = read_bitcode(output->context, {bytes.data(), bytes.size()},
            image ? image->module_name.c_str() : request.name ? request.name : "input.bc");
        auto stage = program_stage(*output->module);
        require_compiler_request(stage == "clang.cuda.tu" || stage == "clang.cuda.linked" || stage == "cudavm.portable", "unsupported cudavm IR stage");
        if (auto* abi = output->module->getNamedMetadata("cudavm.source.abi"); abi && abi->getNumOperands()) {
            auto* row = abi->getOperand(0);
            require_compiler_request(row->getNumOperands() == 6 && llvm::isa<llvm::MDString>(row->getOperand(2)), "invalid source ABI metadata");
            auto architecture = llvm::cast<llvm::MDString>(row->getOperand(2))->getString().str();
            require_compiler_request(!image || architecture == image->architecture, "program image architecture mismatch");
            output->architecture = architecture;
        }
        if (image) require_compiler_request(image_metadata(*output->module) == image->metadata, "program image ABI/linkage metadata mismatch");
        break;
    }
    case compiler_input_kind::nvvm: throw compiler_error(compiler_status::unsupported, "NVVM import is not implemented");
    case compiler_input_kind::nvltoir: throw compiler_error(compiler_status::unsupported, "NVIDIA LTO-IR import is not implemented");
    case compiler_input_kind::cubin: throw compiler_error(compiler_status::unsupported, "cubin import is not implemented");
    default: throw compiler_error(compiler_status::invalid, "unknown compiler input kind");
    }
    if (architecture && output->architecture != architecture)
        throw compiler_error(compiler_status::architecture_mismatch, "compiler target must match the retained program's source architecture");
    result.lowered_names = output->lowered_names;
    return output;
}

std::unique_ptr<compiler_program> link_compiler_programs(size_t count, const compiler_program* const* programs, compiler_result& result) {
    require_compiler_request(count && programs, "IR linking requires input programs");
    compiler_options options;
    auto output = std::make_unique<compiler_program>();
    for (size_t i = 0; i < count; ++i) {
        require_compiler_request(programs[i] && programs[i]->module, "null IR link input");
        const auto& input = *programs[i];
        if (i == 0) options.architecture = output->architecture = input.architecture;
        require_compiler_request(input.architecture == output->architecture, "incompatible IR link architectures");
        for (const auto& [expression, name] : input.lowered_names) {
            auto [found, added] = output->lowered_names.emplace(expression, name);
            require_compiler_request(added || found->second == name, "conflicting lowered name expressions");
        }
        // Imported PTX has already entered common IR. Give it an archive member
        // identity so the same linker can retain/replay it on a later device link.
        if (program_stage(*input.module) == "cudavm.portable" && !input.module->getNamedMetadata("cudavm.link.archive.v1")) {
            auto member = copy_program(input);
            compiler_options settings;
            settings.input = input.module->getModuleIdentifier(); settings.bitcode = program_input_identity(*member->module);
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
    require_compiler_request(request.abi_version == 1 && request.numerical_helpers <= 1 && request.lower_semantics <= 1
        && request.optimization_level <= 3, "invalid compiler pipeline request");
    auto output = copy_program(input);
    if (request.numerical_helpers || (request.lower_semantics && output->module->getNamedMetadata("cudavm.import.architecture")))
        link_device_numeric_ir(*output->module);
    if (request.lower_semantics && program_stage(*output->module) != "cudavm.portable") {
        if (program_stage(*output->module) == "clang.cuda.tu") {
            const compiler_program* programs[] = {output.get()};
            compiler_result diagnostics;
            output = link_compiler_programs(1, programs, diagnostics);
        }
        prepare_portable(*output->module);
    }
    if (request.optimization_level) {
        require_compiler_request(program_stage(*output->module) == "cudavm.portable", "optimization requires semantic lowering");
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
        require_compiler_request(!llvm::verifyModule(*output->module), "optimization produced invalid IR");
    }
    return output;
}

void emit_compiler_program(const compiler_program& program, const compiler_emit_request& request, compiler_result& result) {
    require_compiler_request(request.abi_version == 1, "compiler emission requires abi_version 1");
    require_compiler_request(uint32_t(request.environment) <= uint32_t(compiler_environment::none), "invalid compiler environment");
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
                    require_compiler_request(symbol.empty(), "shader output requires one selected kernel"); symbol = function.getName().str();
                }
        }
        require_compiler_request(!symbol.empty(), "shader output requires a kernel");
        if (request.kind == compiler_output_kind::spirv) {
            require_compiler_request(request.environment == compiler_environment::vulkan || request.environment == compiler_environment::opengl
                || request.environment == compiler_environment::opencl, "SPIR-V output requires a shader environment");
            result.output = emit_native_spirv(*lower_native_spirv(module, symbol, request.environment == compiler_environment::opencl));
        } else {
            require_compiler_request(request.environment == compiler_environment::directx && request.block[0] && request.block[1] && request.block[2], "invalid DXIL output request");
            result.output = emit_native_dxil(*lower_native_dxil(module, symbol, {request.block[0], request.block[1], request.block[2]}));
        }
        return;
    }
    case compiler_output_kind::program_image: {
        if (module.getNamedMetadata("cudavm.import.architecture") && !module.getNamedMetadata("cudavm.numeric.helpers")) {
            auto complete = prepare_compiler_program(program, {1, 1, 0, 0});
            emit_compiler_program(*complete, request, result);
            return;
        }
        require_compiler_request(architecture == program.architecture, "program image architecture must match its portable program");
        program_image image;
        image.architecture = program.architecture; image.module_name = module.getModuleIdentifier();
        image.ir = bitcode(module); image.metadata = image_metadata(module); image.lowered_names = program.lowered_names;
        if (request.environment != compiler_environment::none) {
            compiler_emit_request native = request; native.kind = compiler_output_kind::native_image;
            compiler_result artifact;
            emit_compiler_program(program, native, artifact);
            auto hash = program_image_hash(program_image_identity(image));
            unsigned first = request.environment == compiler_environment::all ? 0 : unsigned(request.environment);
            unsigned last = request.environment == compiler_environment::all ? 4 : first;
            auto spirv = decode_spirv_artifacts(artifact.output);
            auto dxil = decode_dxil_artifacts(artifact.output);
            auto entries = decode_module_descriptor(artifact.output).entries.size();
            for (unsigned backend = first; backend <= last; ++backend) {
                bool available = backend == 0 || (backend == 4
                    ? dxil.size() == entries && std::all_of(dxil.begin(), dxil.end(), [](const auto& item) { return !item.code.empty(); })
                    : spirv.size() == entries && std::all_of(spirv.begin(), spirv.end(), [&](const auto& item) { return !(backend == 1 ? item.opencl_code : item.code).empty(); }));
                if (!available) continue;
                auto code = artifact.output;
                if (backend != 0) {
                    code = emit_native_module(module) + emit_data_views(module);
                    if (backend == 4) code += encode_dxil_artifacts(dxil);
                    else {
                        auto selected = spirv;
                        for (auto& entry : selected) {
                            if (backend == 1) { entry.code.clear(); entry.diagnostic.clear(); }
                            else { entry.opencl_code.clear(); entry.opencl_diagnostic.clear(); }
                        }
                        code += encode_spirv_artifacts(selected);
                    }
                }
                image.artifacts.push_back({backend, program_image_native_profile, packet_memory_abi, hash, std::move(code)});
            }
        }
        result.output = encode_program_image(image);
        return;
    }
    case compiler_output_kind::native_image: break;
    default: throw compiler_error(compiler_status::invalid, "unknown compiler output kind");
    }
    require_compiler_request(request.environment != compiler_environment::none, "native image requires an emission environment");
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
    result.output = emit_native_module(module) + emit_data_views(module);
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
