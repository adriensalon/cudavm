#include "compiler.hpp"
#include "embedded_headers.hpp"
#include "host_source.hpp"
#include "data_alias_frontend.hpp"
#include <filesystem>
#include <stdexcept>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <clang/AST/Attr.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Mangle.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Frontend/MultiplexConsumer.h>
#include <clang/Frontend/DependencyOutputOptions.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Verifier.h>
namespace cudavm {
std::vector<std::string> frontend_arguments(const compiler_options& values, bool device) {
    std::vector<std::string> args {
        "cudavm-clang", "--target=" LLVM_HOST_TRIPLE, "-x", "cuda", "-std=" + values.standard,
        "--cuda-gpu-arch=" + values.architecture,
        "-cuid=" + translation_unit_id(values),
        device ? "--cuda-device-only" : "--cuda-host-only",
        "-nocudainc", "-nocudalib", "-resource-dir", values.resources.resource_dir,
        // Clang 22's enum cannot represent 13.4. Select its newest supported
        // core launch protocol explicitly; public declarations remain 13.4.1.
        "-Xclang", "-target-sdk-version=12.8",
        "-include", values.resources.wrapper,
        "-isystem", values.resources.cuda_include, "-isystem", values.resources.cuda_crt_include,
        "-I", values.resources.public_include,
        "-isystem", values.resources.public_include + "/cuda",
        "-isystem", values.resources.public_include + "/cudadevrt",
        "-iquote", std::filesystem::absolute(values.input).parent_path().string(),
        "-Wno-ignored-attributes", "-Wno-ignored-pragmas", "-Wno-macro-redefined",
    };
    for (auto& include : values.includes) { args.emplace_back("-I"); args.push_back(include); }
    for (auto& define : values.defines) args.push_back("-D" + define);
    if (device) {
        args.insert(args.end(), {"-include", values.resources.resource_dir+"/cudavm_device_math.h"});
        args.emplace_back("-O1");
        args.emplace_back("-fgpu-rdc");
        args.emplace_back("-Xclang"); args.emplace_back("-disable-llvm-passes");
        if (!values.depfile.empty()) {
            // Embedded system headers are tracked by the compiler library's
            // build. Only real user headers belong in the consumer depfile.
            args.insert(args.end(), {"-MMD", "-MF", values.depfile, "-MT", values.bitcode});
        }
    }
    args.push_back(std::filesystem::absolute(values.input).string());
    return args;
}
namespace {
struct host_visitor : clang::RecursiveASTVisitor<host_visitor> {
    clang::ASTContext& context;
    host_source& source;
    host_extraction& result;
    std::string unit;
    std::unordered_set<unsigned> rewritten;
    std::unordered_set<std::string> registered;
    std::unordered_set<std::string> registered_symbols;
    std::unique_ptr<clang::MangleContext> device_mangler;
    host_visitor(clang::ASTContext& c, host_source& input, host_extraction& out, const std::string& key) : context(c), source(input), result(out), unit(key) {
        if (!c.getAuxTargetInfo()) throw std::runtime_error("CUDA host extraction requires a device target ABI");
        device_mangler.reset(c.createDeviceMangleContext(*c.getAuxTargetInfo()));
    }
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool shouldTraversePostOrder() const { return true; }
    clang::PrintingPolicy address_policy() {
        clang::PrintingPolicy policy(context.getLangOpts());
        policy.SuppressScope = false;
        policy.FullyQualifiedName = true;
        policy.SuppressUnwrittenScope = true;
        policy.AnonymousTagLocations = false;
        return policy;
    }
    std::string pointer_type(clang::FunctionDecl* function) {
        return context.getPointerType(function->getType()).getAsString(address_policy());
    }
    std::string kernel_symbol(clang::FunctionDecl* function) {
        std::string symbol;
        llvm::raw_string_ostream name(symbol);
        if (device_mangler->shouldMangleDeclName(function))
            device_mangler->mangleName(clang::GlobalDecl(function, clang::KernelReferenceKind::Kernel), name);
        else name << function->getName();
        // Clang externalizes file-local CUDA kernels for relocatable device
        // code. Use the same explicit CUID and postfix as device CodeGen.
        if (context.shouldExternalize(function)) {
            if (context.getCUIDHash().empty()) throw std::runtime_error("missing CUDA compilation-unit identity");
            name << "__intern__" << context.getCUIDHash();
        }
        return symbol;
    }
    void register_kernel(clang::FunctionDecl* function) {
        if (function->getDescribedFunctionTemplate() || function->getType()->isDependentType()) return;
        auto symbol = kernel_symbol(function);
        if (!registered.insert(symbol).second) return;
        std::string host_name;
        llvm::raw_string_ostream stream(host_name);
        function->getNameForDiagnostic(stream, address_policy(), true);
        result.registrations.push_back(llvm::json::Object{{"device_symbol", symbol},
            {"host_name", host_name}, {"host_pointer_type", pointer_type(function)}});
    }
    void register_symbol(clang::VarDecl* variable) {
        if (!variable->isFileVarDecl() || variable->getType()->isDependentType()
            || variable->getDescribedVarTemplate()
            || !(variable->hasAttr<clang::CUDADeviceAttr>() || variable->hasAttr<clang::CUDAConstantAttr>())) return;
        if (!local(variable->getLocation())) return;
        if (variable->hasAttr<clang::HIPManagedAttr>())
            throw std::runtime_error("managed CUDA variable registration is pending: " + variable->getQualifiedNameAsString());
        std::string ir_symbol;
        llvm::raw_string_ostream name(ir_symbol);
        if (device_mangler->shouldMangleDeclName(variable)) device_mangler->mangleName(variable, name);
        else name << variable->getName();
        auto symbol = ir_symbol;
        // Local globals also need stable names in the linked module's metadata.
        if (!variable->hasExternalFormalLinkage())
            symbol += "__static__" + context.getCUIDHash().str();
        if (!registered_symbols.insert(symbol).second) return;
        std::string host_name;
        llvm::raw_string_ostream host(host_name);
        variable->getNameForDiagnostic(host, address_policy(), true);
        result.symbols.push_back(llvm::json::Object{{"device_symbol", symbol}, {"ir_symbol", ir_symbol}, {"host_name", host_name}});
    }
    bool VisitVarDecl(clang::VarDecl* variable) {
        if (local(variable->getLocation()) && variable->hasAttr<clang::AliasAttr>()
            && (variable->hasAttr<clang::CUDADeviceAttr>() || variable->hasAttr<clang::CUDAConstantAttr>())) {
            register_symbol(variable);
            // Host addresses are registration tokens. Each alias needs its own
            // token even though its device address shares the target's storage.
            std::string declaration;
            llvm::raw_string_ostream stream(declaration);
            variable->getType().print(stream, address_policy(), variable->getName());
            source.omit_device_alias(variable->getAttr<clang::AliasAttr>()->getRange());
            if (auto* weak = variable->getAttr<clang::WeakAttr>()) source.omit_device_weak(weak->getRange());
            auto prefix = variable->hasAttr<clang::WeakAttr>() ? "extern __attribute__((weak)) "
                : variable->hasExternalFormalLinkage() ? "inline " : "static ";
            replace(variable->getSourceRange(), prefix + declaration);
            if (variable->getType().isConstQualified() || variable->hasAttr<clang::WeakAttr>()) {
                auto end = source.span(variable->getSourceRange()).second;
                while (end < source.items.size() && source.items[end].text != ";") ++end;
                if (end == source.items.size()) throw std::runtime_error("unterminated CUDA data alias declaration");
                replace({source.items[end].location, source.items[end].location}, "= {};");
            }
            return true;
        }
        if (variable->isThisDeclarationADefinition() != clang::VarDecl::DeclarationOnly) register_symbol(variable);
        return true;
    }
    bool VisitDeclRefExpr(clang::DeclRefExpr* reference) {
        if (local(reference->getLocation()))
            if (auto* variable = llvm::dyn_cast<clang::VarDecl>(reference->getDecl())) register_symbol(variable);
        return true;
    }
    bool first_body(clang::FunctionDecl* function) {
        // Instantiations can share a source range with their template pattern.
        return rewritten.insert(function->getBody()->getBeginLoc().getRawEncoding()).second;
    }
    bool local(clang::SourceLocation location) { return source.user(location); }
    std::string text(clang::SourceRange range) { return source.text(range); }
    void replace(clang::SourceRange range, const std::string& value) { source.replace(range, value); }
    bool skip_function(clang::FunctionDecl* function) {
        if (!function) return true;
        if (!local(function->getLocation())) return true;
        if (function->hasAttr<clang::CUDAGlobalAttr>() && function->hasAttr<clang::AliasAttr>()) {
            if (function->getType()->isDependentType() || function->getDescribedFunctionTemplate())
                throw std::runtime_error("dependent CUDA kernel alias extraction is pending");
            // A host kernel address is a registration token. Give aliases their
            // own stubs instead of asking the host linker to alias another TU.
            source.omit_device_alias(function->getAttr<clang::AliasAttr>()->getRange());
            std::string declaration;
            llvm::raw_string_ostream stream(declaration);
            function->getType().print(stream, address_policy(), function->getName());
            auto prefix = function->hasAttr<clang::WeakAttr>() ? "extern __attribute__((weak)) "
                : function->hasExternalFormalLinkage() ? "inline " : "static ";
            replace(function->getSourceRange(), prefix + declaration);
            auto end = source.span(function->getSourceRange()).second;
            while (end < source.items.size() && source.items[end].text != ";") ++end;
            if (end == source.items.size()) throw std::runtime_error("unterminated CUDA kernel alias declaration");
            replace({source.items[end].location, source.items[end].location},
                "{ static volatile unsigned char __cudavm_identity = 0; (void)__cudavm_identity; ::cudavm::direct_kernel_call(); }");
            register_kernel(function);
            result.kernels.push_back(llvm::json::Object{{"name", function->getQualifiedNameAsString()},
                {"parameter_count", int64_t(function->getNumParams())}});
            return true;
        }
        if (function->hasAttr<clang::CUDADeviceAttr>() && !function->hasAttr<clang::CUDAHostAttr>()) {
            if (auto* alias = function->getAttr<clang::AliasAttr>()) {
                // The helper definition is removed from host output. Keeping
                // its alias would make the host compiler require that body.
                source.omit_device_alias(alias->getRange());
                return true;
            }
        }
        if (function->doesThisDeclarationHaveABody() && function->hasAttr<clang::CUDAGlobalAttr>()) {
            // Ordinary C++ needs a stable function address for cudaLaunchKernel.
            // A volatile reference to unique writable storage prevents compiler
            // and linker identical-code folding from merging kernel identities,
            // including distinct template instantiations. This cannot run device code.
            if (first_body(function))
                replace(function->getBody()->getSourceRange(), "{ static volatile unsigned char __cudavm_identity = 0; (void)__cudavm_identity; ::cudavm::direct_kernel_call(); }");
            register_kernel(function);
            result.kernels.push_back(llvm::json::Object{{"name", function->getQualifiedNameAsString()},
                {"parameter_count", int64_t(function->getNumParams())}});
            return true;
        }
        if (function->doesThisDeclarationHaveABody() && function->hasAttr<clang::CUDADeviceAttr>() && !function->hasAttr<clang::CUDAHostAttr>()) {
            if (auto* constructor = llvm::dyn_cast<clang::CXXConstructorDecl>(function); constructor && constructor->getNumCtorInitializers())
                throw std::runtime_error("CUDA-only constructor initializer extraction is pending");
            if (first_body(function)) replace(function->getBody()->getSourceRange(), ";");
            return true;
        }
        return false;
    }
    bool TraverseFunctionDecl(clang::FunctionDecl* function) {
        if (skip_function(function)) return true;
        return clang::RecursiveASTVisitor<host_visitor>::TraverseFunctionDecl(function);
    }
    bool TraverseCXXMethodDecl(clang::CXXMethodDecl* function) {
        if (skip_function(function)) return true;
        return clang::RecursiveASTVisitor<host_visitor>::TraverseCXXMethodDecl(function);
    }
    bool TraverseCXXConstructorDecl(clang::CXXConstructorDecl* function) {
        if (skip_function(function)) return true;
        return clang::RecursiveASTVisitor<host_visitor>::TraverseCXXConstructorDecl(function);
    }
    bool TraverseCXXDestructorDecl(clang::CXXDestructorDecl* function) {
        if (skip_function(function)) return true;
        return clang::RecursiveASTVisitor<host_visitor>::TraverseCXXDestructorDecl(function);
    }
    bool TraverseCXXConversionDecl(clang::CXXConversionDecl* function) {
        if (skip_function(function)) return true;
        return clang::RecursiveASTVisitor<host_visitor>::TraverseCXXConversionDecl(function);
    }
    bool VisitCallExpr(clang::CallExpr* call) {
        auto* function = call->getDirectCallee();
        if (!function || !function->getPrimaryTemplate() || !function->getDeclContext()->isTranslationUnit()
            || !local(call->getBeginLoc())) return true;
        auto name = function->getNameAsString();
        unsigned index;
        if (name == "cudaMemcpyToSymbol" || name == "cudaMemcpyToSymbolAsync") index = 0;
        else if (name == "cudaMemcpyFromSymbol" || name == "cudaMemcpyFromSymbolAsync" || name == "cudaGetSymbolAddress" || name == "cudaGetSymbolSize") index = 1;
        else return true;
        auto* symbol = call->getArg(index);
        // The CUDA-only template passes &symbol to the ordinary C ABI. Keep
        // this conversion when the extracted host file is compiled as C++.
        replace(symbol->getSourceRange(), "::cudavm::symbol_address(" + text(symbol->getSourceRange()) + ")");
        return true;
    }
    bool VisitCUDAKernelCallExpr(clang::CUDAKernelCallExpr* call) {
        if (!local(call->getBeginLoc())) return true;
        auto* function = call->getDirectCallee();
        if (!function) throw std::runtime_error("indirect CUDA launch extraction is pending");
        auto symbol = kernel_symbol(function);
        result.launches.push_back(llvm::json::Object{{"source_name", function->getQualifiedNameAsString()},
            {"device_symbol", symbol}, {"parameter_count", int64_t(function->getNumParams())}});
        auto* config = call->getConfig();
        auto expression = [&](const clang::Expr* e, const char* fallback) {
            return llvm::isa<clang::CXXDefaultArgExpr>(e) ? std::string(fallback) : text(e->getSourceRange());
        };
        std::string output = "([&]() -> void { const dim3 __cudavm_grid(" + expression(config->getArg(0), "1")
            + "); const dim3 __cudavm_block(" + expression(config->getArg(1), "1")
            + "); const size_t __cudavm_shared = " + expression(config->getArg(2), "0")
            + "; const cudaStream_t __cudavm_stream = " + expression(config->getArg(3), "nullptr") + "; ";
        clang::PrintingPolicy policy(context.getLangOpts());
        policy.SuppressScope = false;
        for (unsigned i = 0; i < call->getNumArgs(); ++i) {
            // Clang has already checked parameter conversions. Preserve each
            // by-value parameter's exact type instead of packing auto expressions.
            auto type = function->getParamDecl(i)->getType();
            if (type->isReferenceType()) throw std::runtime_error("reference kernel parameters cannot be marshalled");
            auto name = "__cudavm_arg" + std::to_string(i);
            std::string declaration;
            llvm::raw_string_ostream stream(declaration);
            type.print(stream, policy, name);
            output += declaration + " = " + text(call->getArg(i)->getSourceRange()) + "; ";
        }
        if (call->getNumArgs()) {
            output += "void* __cudavm_args[] = {";
            for (unsigned i = 0; i < call->getNumArgs(); ++i) {
                if (i) output += ", ";
                output += "const_cast<void*>(static_cast<const void*>(&__cudavm_arg" + std::to_string(i) + "))";
            }
            output += "}; ";
        }
        output += "const void* __cudavm_function = reinterpret_cast<const void*>(static_cast<" + pointer_type(function) + ">(&"
            + text(call->getCallee()->getSourceRange()) + ")); (void)::cudaLaunchKernel(__cudavm_function, __cudavm_grid, __cudavm_block, "
            + (call->getNumArgs() ? "__cudavm_args" : "nullptr") + ", __cudavm_shared, __cudavm_stream); }())";
        replace(call->getSourceRange(), output);
        return true;
    }
};
struct host_consumer : clang::ASTConsumer {
    clang::CompilerInstance& compiler;
    host_source& source;
    host_extraction& result;
    std::string unit;
    host_consumer(clang::CompilerInstance& c, host_source& input, host_extraction& out, const std::string& key) : compiler(c), source(input), result(out), unit(key) {}
    void HandleTranslationUnit(clang::ASTContext& context) override {
        if (compiler.getDiagnostics().hasErrorOccurred()) return;
        host_visitor visitor(context, source, result, unit);
        visitor.TraverseDecl(context.getTranslationUnitDecl());
    }
};
struct host_action : clang::ASTFrontendAction {
    host_extraction& result;
    std::string unit;
    const compiler_options& values;
    host_source* source = nullptr;
    host_action(host_extraction& out, std::string key, const compiler_options& request)
        : result(out), unit(std::move(key)), values(request) {}
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler, llvm::StringRef) override {
        auto callbacks = std::make_unique<host_source>(compiler, result, values);
        source = callbacks.get();
        compiler.getPreprocessor().addPPCallbacks(std::move(callbacks));
        return std::make_unique<host_consumer>(compiler, *source, result, unit);
    }
    void EndSourceFileAction() override {
        if (getCompilerInstance().getDiagnostics().hasErrorOccurred()) return;
        std::string output;
        llvm::raw_string_ostream stream(output);
        stream << "// Generated by cudavm from expanded CUDA user tokens.\n#include <nvcc/cudavm_host.hpp>\n";
        stream << source->render();
        stream << "\nextern \"C\" void __cudavm_register_" << unit << "(const cudavmProgram* program) noexcept {\n(void)program;\n";
        for (const auto& value : result.registrations) {
            auto* entry = value.getAsObject();
            stream << "(void)::cudavmBindKernel(reinterpret_cast<const void*>(static_cast<"
                << *entry->getString("host_pointer_type") << ">(&::" << *entry->getString("host_name")
                << ")), program, \"" << *entry->getString("device_symbol") << "\");\n";
        }
        for (const auto& value : result.symbols) {
            auto* entry = value.getAsObject();
            stream << "(void)::cudavmBindSymbol(const_cast<const void*>(static_cast<const volatile void*>(&::" << *entry->getString("host_name")
                << ")), program, \"" << *entry->getString("device_symbol") << "\");\n";
        }
        stream << "}\n";
        result.host = std::move(output);
    }
};
struct real_dependency_generator : clang::DependencyFileGenerator {
    std::filesystem::path builtin;
    real_dependency_generator(const clang::DependencyOutputOptions& compiler_options, const std::string& root,
                              const std::vector<std::string>& host_dependencies)
        : clang::DependencyFileGenerator(compiler_options), builtin(std::filesystem::path(root).lexically_normal()) {
        for (const auto& path : host_dependencies) maybeAddDependency(path, false, false, false, false);
    }
    void maybeAddDependency(llvm::StringRef name, bool module, bool system, bool module_file, bool missing) override {
        auto relative = std::filesystem::path(name.str()).lexically_normal().lexically_relative(builtin);
        if (!relative.empty() && *relative.begin() != "..") return;
        clang::DependencyFileGenerator::maybeAddDependency(name, module, system, module_file, missing);
    }
};
// Force CodeGen to materialize host-registered globals using the device AST's
// actual type/layout. An extern used only by cudaGetSymbol*/cudaMemcpy*Symbol
// otherwise disappears from device IR. It remains a declaration until linking.
struct device_symbol_consumer : clang::ASTConsumer, clang::RecursiveASTVisitor<device_symbol_consumer> {
    clang::CodeGenerator& generator;
    std::unordered_set<std::string> names;
    std::unordered_set<const clang::VarDecl*> seen;
    device_symbol_consumer(clang::CodeGenerator& codegen, const llvm::json::Array& symbols) : generator(codegen) {
        for (const auto& entry : symbols) {
            names.insert(entry.getAsObject()->getString("ir_symbol")->str());
            names.insert(entry.getAsObject()->getString("device_symbol")->str());
        }
    }
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool VisitVarDecl(clang::VarDecl* variable) {
        if (!variable->isFileVarDecl() || variable->getType()->isDependentType() || variable->getDescribedVarTemplate()
            || !(variable->hasAttr<clang::CUDADeviceAttr>() || variable->hasAttr<clang::CUDAConstantAttr>())) return true;
        if (names.contains(generator.GetMangledName(clang::GlobalDecl(variable)).str())
            && seen.insert(variable->getCanonicalDecl()).second)
            generator.GetAddrOfGlobal(clang::GlobalDecl(variable), false);
        return true;
    }
    void HandleTranslationUnit(clang::ASTContext& context) override {
        if (!context.getDiagnostics().hasErrorOccurred()) TraverseDecl(context.getTranslationUnitDecl());
    }
};
struct device_action : clang::EmitLLVMOnlyAction {
    std::vector<source_data_alias> aliases;
    std::unique_ptr<llvm::Module>& result;
    std::string builtin;
    const std::vector<std::string>& host_dependencies;
    const llvm::json::Array* host_symbols;
    device_action(llvm::LLVMContext& context, std::unique_ptr<llvm::Module>& output, const std::string& root,
                  const std::vector<std::string>& dependencies, const llvm::json::Array* symbols)
        : clang::EmitLLVMOnlyAction(&context), result(output), builtin(root), host_dependencies(dependencies), host_symbols(symbols) {}
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler, llvm::StringRef file) override {
        auto codegen = clang::EmitLLVMOnlyAction::CreateASTConsumer(compiler, file);
        std::vector<std::unique_ptr<clang::ASTConsumer>> consumers;
        consumers.push_back(std::make_unique<data_alias_consumer>(*getCodeGenerator(), aliases));
        if (host_symbols && !host_symbols->empty())
            consumers.push_back(std::make_unique<device_symbol_consumer>(*getCodeGenerator(), *host_symbols));
        consumers.push_back(std::move(codegen));
        return std::make_unique<clang::MultiplexConsumer>(std::move(consumers));
    }
    bool BeginInvocation(clang::CompilerInstance& compiler) override {
        auto& compiler_options = compiler.getDependencyOutputOpts();
        if (!compiler_options.OutputFile.empty()) {
            compiler.addDependencyCollector(std::make_shared<real_dependency_generator>(compiler_options, builtin, host_dependencies));
            compiler_options.OutputFile.clear();
        }
        return clang::EmitLLVMOnlyAction::BeginInvocation(compiler);
    }
    void EndSourceFileAction() override {
        clang::EmitLLVMOnlyAction::EndSourceFileAction();
        result = takeModule();
        if (result) materialize_data_aliases(*result, aliases);
    }
};
template<class Action> void invoke(const compiler_options& values, bool device, std::unique_ptr<Action> action) {
    clang::FileSystemOptions file_options;
    auto manager = llvm::makeIntrusiveRefCnt<clang::FileManager>(file_options, frontend_files(values.resources));
    clang::tooling::ToolInvocation invocation(frontend_arguments(values, device), std::move(action), manager.get());
    clang::DiagnosticOptions diagnostic_options;
    llvm::raw_string_ostream diagnostics(values.diagnostics);
    clang::TextDiagnosticPrinter printer(diagnostics, diagnostic_options);
    invocation.setDiagnosticConsumer(&printer);
    if (!invocation.run()) throw std::runtime_error(device ? "CUDA device frontend failed" : "CUDA host frontend failed");
}
}
host_extraction extract_host(const compiler_options& values) {
    host_extraction result;
    invoke(values, false, std::make_unique<host_action>(result, translation_unit_id(values), values));
    return result;
}
std::unique_ptr<llvm::Module> compile_device(llvm::LLVMContext& context, const compiler_options& values, const std::vector<std::string>& host_dependencies, const llvm::json::Array* host_symbols) {
    std::unique_ptr<llvm::Module> module;
    invoke(values, true, std::make_unique<device_action>(context, module, values.resources.resource_dir, host_dependencies, host_symbols));
    if (!module) throw std::runtime_error("CUDA device frontend produced no IR");
    llvm::raw_string_ostream diagnostics(values.diagnostics);
    if (llvm::verifyModule(*module, &diagnostics))
        throw std::runtime_error("CUDA device frontend produced invalid IR");
    return module;
}
}
