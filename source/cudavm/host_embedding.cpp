#include "compiler.hpp"
#include <filesystem>
#include <set>
#include <stdexcept>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>
namespace cudavm {
namespace {
std::string digest(llvm::StringRef value) {
    llvm::SHA256 hash;
    hash.update(value);
    auto bytes = hash.final();
    constexpr char alphabet[] = "0123456789abcdef";
    std::string result;
    for (auto byte : bytes) { result += alphabet[byte >> 4]; result += alphabet[byte & 15]; }
    return result;
}
std::string literal(llvm::StringRef value) {
    // CUDA kernel symbols are ASCII. JSON and C++ share these escapes.
    return llvm::formatv("{0}", llvm::json::Value(value)).str();
}
}
std::string translation_unit_id(const compiler_options& values) {
    return digest(std::filesystem::absolute(values.bitcode).lexically_normal().generic_string());
}
std::string embed_cuda(const llvm::Module& module, const std::string& code) {
    auto description = describe(module);
    const auto& kernels = *description.getArray("kernels");
    std::string layout = llvm::formatv("{0}", llvm::json::Value(llvm::json::Array(kernels))).str();
    auto identity = digest(code + layout);
    auto* units = module.getNamedMetadata("cudavm.translation.units");
    if (!units || !units->getNumOperands()) throw std::runtime_error("cannot embed a module without translation-unit identities");
    std::set<std::string> identifiers;
    for (auto* row : units->operands()) {
        if (row->getNumOperands() != 2) throw std::runtime_error("invalid translation-unit identity metadata");
        auto id = llvm::cast<llvm::MDString>(row->getOperand(0))->getString().str();
        if (id.size() != 64 || id.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("invalid translation-unit identity");
        identifiers.insert(std::move(id));
    }
    if (auto* libraries = module.getNamedMetadata("cudavm.external.library.v1"))
        for (auto* row : libraries->operands()) {
            if (row->getNumOperands() != 2) throw std::runtime_error("invalid external device library ownership metadata");
            auto* id = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(0));
            auto* format = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(1));
            if (!id || !format || format->getString() != "llvm-bitcode-helper-data-v1" || !identifiers.erase(id->getString().str()))
                throw std::runtime_error("invalid external device library ownership metadata");
        }
    std::string output;
    llvm::raw_string_ostream stream(output);
    stream << "// Generated CUDA AOT module; registration performs no GPU initialization.\n"
        "#include <cudart/cudavm_registration.h>\n";
    for (const auto& id : identifiers)
        stream << "extern \"C\" void __cudavm_register_" << id << "(const cudavmProgram*) noexcept;\n";
    if (auto* registrations = module.getNamedMetadata("cudavm.host.registrations")) {
        for (auto* row : registrations->operands()) {
            if (row->getNumOperands() != 2) throw std::runtime_error("invalid host kernel registration metadata");
            auto* function = module.getFunction(llvm::cast<llvm::MDString>(row->getOperand(1))->getString());
            if (!function || function->isDeclaration() || !function->hasFnAttribute("cudavm.kernel"))
                throw std::runtime_error("host kernel registration escaped the closed device graph");
        }
    }
    if (auto* symbols = module.getNamedMetadata("cudavm.host.symbols")) {
        for (auto* row : symbols->operands()) {
            if (row->getNumOperands() != 2) throw std::runtime_error("invalid host symbol registration metadata");
            auto* variable = module.getGlobalVariable(llvm::cast<llvm::MDString>(row->getOperand(1))->getString(), true);
            if (!variable) {
                auto symbol = llvm::cast<llvm::MDString>(row->getOperand(1))->getString();
                if (auto* views = module.getNamedMetadata("cudavm.data.views.v1"))
                    for (auto* view : views->operands())
                        if (llvm::cast<llvm::MDString>(view->getOperand(0))->getString() == symbol)
                            variable = module.getGlobalVariable(llvm::cast<llvm::MDString>(view->getOperand(1))->getString(), true);
            }
            if (!variable || !variable->hasInitializer() || (variable->getAddressSpace() != 1 && variable->getAddressSpace() != 4))
                throw std::runtime_error("host symbol registration escaped the closed device graph");
        }
    }
    stream << "namespace {\nconst unsigned char image[] = {\n";
    unsigned column = 0;
    for (unsigned char byte : code) {
        stream << static_cast<unsigned>(byte) << ',';
        if (++column == 32) { stream << '\n'; column = 0; }
    }
    stream << "0};\n";
    size_t index = 0;
    for (const auto& value : kernels) {
        const auto& parameters = *value.getAsObject()->getArray("parameters");
        if (!parameters.empty()) {
            stream << "const cudavmKernelParameter parameters" << index << "[] = {\n";
            for (const auto& parameter : parameters) {
                auto* data = parameter.getAsObject();
                stream << '{' << *data->getInteger("size") << ',' << *data->getInteger("alignment") << ',' << *data->getInteger("offset") << "},\n";
            }
            stream << "};\n";
        }
        ++index;
    }
    if (!kernels.empty()) {
        stream << "const cudavmKernelDescriptor kernels[] = {\n";
        index = 0;
        for (const auto& value : kernels) {
            auto* kernel = value.getAsObject();
            auto count = kernel->getArray("parameters")->size();
            stream << '{' << literal(*kernel->getString("symbol")) << ',' << count << ',';
            if (count) stream << "parameters" << index; else stream << "nullptr";
            stream << "},\n"; ++index;
        }
        stream << "};\n";
    }
    stream << "const cudavmProgram program{1,cudavmBackendCuda,\"" << identity
        << "\",image,sizeof(image)," << kernels.size() << ',' << (kernels.empty() ? "nullptr" : "kernels") << "};\n"
        "struct register_program { register_program() noexcept {\n";
    for (const auto& id : identifiers) stream << "::__cudavm_register_" << id << "(&program);\n";
    stream << "} };\n"
        "// CPU registration runs before ordinary user dynamic initializers.\n"
        "#if defined(_MSC_VER)\n#pragma init_seg(lib)\n"
        "register_program registration;\n"
        "#elif defined(__GNUC__)\nregister_program registration __attribute__((init_priority(101)));\n"
        "#else\n#error cudavm requires a supported host initialization-order mechanism\n#endif\n}\n";
    for (const auto& id : identifiers)
        stream << "extern \"C\" const cudavmProgram* __cudavm_program_" << id << "() { return &program; }\n";
    return output;
}
}
