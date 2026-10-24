#include "nvrtc_frontend.hpp"
#include "spirv_emitter.hpp"
#include "dxil_emitter.hpp"
#include "data_alias_frontend.hpp"
#include "device_conversions.hpp"
#include "device_integers.hpp"
#include "device_simd.hpp"
#include "device_dpx.hpp"
#include "device_math.hpp"
#include "device_warp.hpp"
#include "device_sources.hpp"
#include "embedded_headers.hpp"
#include <filesystem>
#include <mutex>
#include <clang/AST/Attr.h>
#include <clang/AST/Mangle.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/Builtins.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/MultiplexConsumer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
namespace cudavm {
namespace {
struct nvrtc_consumer : clang::ASTConsumer, clang::RecursiveASTVisitor<nvrtc_consumer> {
    clang::CompilerInstance& compiler;
    const std::vector<std::string>& expressions;
    nvrtc_output& output;
    std::unique_ptr<clang::MangleContext> mangler;
    nvrtc_consumer(clang::CompilerInstance& c, const std::vector<std::string>& e, nvrtc_output& o)
        : compiler(c), expressions(e), output(o) {}
    void Initialize(clang::ASTContext& context) override { mangler.reset(context.createMangleContext()); }
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool VisitCallExpr(clang::CallExpr* call) {
        auto* function = call->getDirectCallee();
        if (!function || function->getBuiltinID() < clang::Builtin::FirstTSBuiltin) return true;
        // The consumed Clang archive has incompatible enum-based NVVM builtin
        // lowering. These register builtins resolve through the name table and
        // are verified by GPU execution; other target builtins need explicit
        // lowering before they can enter in-process CodeGen.
        auto name = function->getName();
        constexpr const char* registers[] = {"tid", "ctaid", "ntid", "nctaid"};
        for (auto* reg : registers) for (auto axis : {"x", "y", "z"})
            if (name == std::string("__nvvm_read_ptx_sreg_") + reg + "_" + axis) return true;
        auto id = compiler.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "NVRTC target builtin %0 requires explicit lowering (pending)");
        compiler.getDiagnostics().Report(call->getExprLoc(), id) << name;
        return true;
    }
    bool HandleTopLevelDecl(clang::DeclGroupRef declarations) override {
        for (auto* declaration : declarations) TraverseDecl(declaration);
        // MultiplexConsumer stops before forwarding rejected declarations to
        // CodeGen, including unsupported builtin calls in inline definitions.
        return !compiler.getDiagnostics().hasErrorOccurred();
    }
    bool VisitFunctionDecl(clang::FunctionDecl* function) {
        auto& sources=compiler.getSourceManager();
        auto file=sources.getFilename(sources.getSpellingLoc(function->getLocation()));
        // Modern CUDA headers normally obtain these definitions from NVIDIA's
        // NVRTC builtins library. Compile their provided source here and retain
        // only used inline definitions, like the AOT CUDA header path.
        if((file.ends_with("cuda_fp16.hpp") || file.ends_with("cuda_fp16.h"))
            && function->hasAttr<clang::CUDADeviceAttr>() && function->isThisDeclarationADefinition())
            function->setImplicitlyInline();
        if (function->isImplicit() || !function->doesThisDeclarationHaveABody()
            || compiler.getSourceManager().isInSystemHeader(function->getLocation())) return true;
        if (!function->hasAttr<clang::CUDADeviceAttr>() && !function->hasAttr<clang::CUDAGlobalAttr>()) {
            auto id = compiler.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
                "NVRTC device-only compilation does not accept host function %0");
            compiler.getDiagnostics().Report(function->getLocation(), id) << function->getNameAsString();
        }
        return true;
    }
    bool VisitVarDecl(clang::VarDecl* variable) {
        constexpr llvm::StringLiteral prefix = "__cudavm_nvrtc_name_";
        if (!variable->getName().starts_with(prefix)) return true;
        unsigned index;
        if (variable->getName().drop_front(prefix.size()).getAsInteger(10, index) || index >= expressions.size()) return true;
        auto* expression = variable->getInit()->IgnoreParenImpCasts();
        if (auto* address = llvm::dyn_cast<clang::UnaryOperator>(expression))
            if (address->getOpcode() == clang::UO_AddrOf) expression = address->getSubExpr()->IgnoreParenImpCasts();
        auto* reference = llvm::dyn_cast<clang::DeclRefExpr>(expression);
        auto* named = reference ? llvm::dyn_cast<clang::NamedDecl>(reference->getDecl()) : nullptr;
        if (!named) {
            auto id = compiler.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
                "NVRTC name expression must reference a device function or variable");
            compiler.getDiagnostics().Report(variable->getLocation(), id);
            return true;
        }
        std::string symbol;
        llvm::raw_string_ostream stream(symbol);
        if (auto* function = llvm::dyn_cast<clang::FunctionDecl>(named)) {
            if (!function->hasAttr<clang::CUDAGlobalAttr>()) {
                auto id = compiler.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
                    "NVRTC function name expression must reference a __global__ kernel");
                compiler.getDiagnostics().Report(variable->getLocation(), id);
                return true;
            }
            if (mangler->shouldMangleDeclName(function))
                mangler->mangleName(clang::GlobalDecl(function, clang::KernelReferenceKind::Kernel), stream);
            else stream << function->getName();
        } else {
            if (!named->hasAttr<clang::CUDADeviceAttr>() && !named->hasAttr<clang::CUDAConstantAttr>()) {
                auto id = compiler.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
                    "NVRTC variable name expression must reference a __device__ or __constant__ variable");
                compiler.getDiagnostics().Report(variable->getLocation(), id);
                return true;
            }
            if (mangler->shouldMangleDeclName(named)) mangler->mangleName(named, stream);
            else stream << named->getName();
        }
        output.lowered_names.emplace(expressions[index], std::move(symbol));
        return true;
    }
    void HandleTranslationUnit(clang::ASTContext& context) override {
        if (compiler.getDiagnostics().hasErrorOccurred()) return;
        TraverseDecl(context.getTranslationUnitDecl());
    }
};
struct nvrtc_action : clang::EmitLLVMOnlyAction {
    std::vector<source_data_alias> aliases;
    const std::vector<std::string>& expressions;
    nvrtc_output& output;
    nvrtc_action(llvm::LLVMContext& context, const std::vector<std::string>& e, nvrtc_output& o)
        : clang::EmitLLVMOnlyAction(&context), expressions(e), output(o) {}
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler, llvm::StringRef file) override {
        std::vector<std::unique_ptr<clang::ASTConsumer>> consumers;
        consumers.push_back(std::make_unique<nvrtc_consumer>(compiler, expressions, output));
        auto codegen = clang::EmitLLVMOnlyAction::CreateASTConsumer(compiler, file);
        consumers.push_back(std::make_unique<data_alias_consumer>(*getCodeGenerator(), aliases));
        consumers.push_back(std::move(codegen));
        return std::make_unique<clang::MultiplexConsumer>(std::move(consumers));
    }
};
}
nvrtc_output import_cuda_source(llvm::LLVMContext& context, const std::string& source, const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& headers,
    const std::vector<std::string>& expressions, const nvrtc_options& settings, std::string& log) {
    // Each request has its own virtual files, diagnostics, frontend and IR context.
    // Neither source nor virtual headers are materialized on disk.
    auto root = (std::filesystem::current_path() / "__cudavm_nvrtc_virtual__").generic_string();
    auto files = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    auto add = [&](const std::string& path, llvm::StringRef data) {
        if (!files->addFile(path, 0, llvm::MemoryBuffer::getMemBufferCopy(data)))
            throw std::runtime_error("conflicting NVRTC virtual file: " + path);
    };
    add_embedded_files(*files, root);
    std::string rtc_preamble(device_source("nvrtc_preamble.cu"));
    for(auto operation:device_simd_operations) {
        std::string name=operation.name,arguments="unsigned int x";
        if(operation.arguments==2)arguments+=",unsigned int y";
        rtc_preamble += render_device_source("device_wrapper.cu.in", {
            {"result", "unsigned int"}, {"symbol", "__nv_"+name}, {"arguments", arguments},
            {"name", "__"+name}, {"linkage", ""}, {"call", operation.arguments==2 ? "x,y" : "x"}});
    }
    for(auto operation:device_dot_operations) {
        std::string word=operation.signed_value?"int":"unsigned int";
        std::string arguments=word+" x,"+word+" y,"+word+" c";
        rtc_preamble += render_device_source("device_wrapper.cu.in", {
            {"result", word}, {"symbol", operation.symbol}, {"arguments", arguments},
            {"name", "__"+std::string(operation.name)}, {"linkage", ""}, {"call", "x,y,c"}});
        std::string vector_a=operation.high<0?(operation.signed_value?"char4":"uchar4"):(operation.signed_value?"short2":"ushort2");
        std::string vector_b=operation.signed_value?"char4":"uchar4";
        rtc_preamble += render_device_source("device_dot_vector.cu.in", {
            {"word", word}, {"name", operation.name}, {"vector_a", vector_a},
            {"vector_b", vector_b}, {"symbol", operation.symbol}});
    }
    for(auto operation:device_integer_operations) {
        std::string name=operation.name;
        std::string word=operation.bits==64?"long long":"int";
        if(name.starts_with("u"))word="unsigned "+word;
        std::string result=operation.arguments==3?"unsigned int":word;
        std::string arguments=word+" x,"+word+" y";
        if(operation.arguments==3)arguments+=",unsigned int z";
        rtc_preamble += render_device_source("device_wrapper.cu.in", {
            {"result", result}, {"symbol", "__nv_"+name}, {"arguments", arguments},
            {"name", "__"+name}, {"linkage", ""}, {"call", operation.arguments==3 ? "x,y,z" : "x,y"}});
    }
    for(auto name:{"hadd","rhadd","uhadd","urhadd"}) {
        std::string word=std::string_view(name).starts_with("u")?"unsigned int":"int";
        std::string arguments=word+" x,"+word+" y";
        rtc_preamble += render_device_source("device_wrapper.cu.in", {
            {"result", word}, {"symbol", "__nv_"+std::string(name)}, {"arguments", arguments},
            {"name", "__"+std::string(name)}, {"linkage", ""}, {"call", "x,y"}});
    }
    add(root+"/builtins/cudavm_device_dpx.h",device_dpx_header());
    rtc_preamble+="#include <cudavm_device_dpx.h>\n";
    add(root+"/builtins/cudavm_device_warp.h",device_warp_header());
    rtc_preamble+="#include <cudavm_device_warp.h>\n";
    add(root+"/builtins/cudavm_device_math.h",llvm::StringRef(device_source("device_math.cu")));
    rtc_preamble+="#include <cudavm_device_math.h>\n";
    for(auto operation:device_math_operations) {
        std::string name=operation.name;
        std::string arguments=operation.arguments==1?"float x":"float x, float y";
        std::string call=operation.arguments==1?"x":"x, y";
        rtc_preamble += render_device_source("device_wrapper.cu.in", {
            {"result", "float"}, {"symbol", "__nv_"+name}, {"arguments", arguments},
            {"name", name}, {"linkage", "extern \"C\" "}, {"call", call}});
    }
    for (const auto& conversion : device_conversions) {
        const std::string integer = std::string(conversion.signed_value ? "" : "unsigned ")
            + (conversion.integer_bits == 32 ? "int" : "long long");
        const std::string real = conversion.real_bits == 64 ? "double" : "float";
        const std::string result = conversion.to_real ? real : integer;
        const std::string argument = conversion.to_real ? integer : real;
        const std::string name = conversion.name;
        rtc_preamble += render_device_source("device_wrapper.cu.in", {
            {"result", result}, {"symbol", "__nv_"+name}, {"arguments", argument+" x"},
            {"name", "__"+name}, {"linkage", "extern \"C\" "}, {"call", "x"}});
    }
    for (const auto& conversion : double_conversions) {
        const std::string name = conversion.name;
        rtc_preamble += render_device_source("device_wrapper.cu.in", {
            {"result", "float"}, {"symbol", "__nv_"+name}, {"arguments", "double x"},
            {"name", "__"+name}, {"linkage", "extern \"C\" "}, {"call", "x"}});
    }
    add(root + "/builtins/preamble.h", rtc_preamble);
    for (const auto& [header, contents] : headers) add(root + "/headers/" + header, contents);
    auto input = (std::filesystem::path(root) / "source" / name).generic_string();
    std::string text = source;
    for (size_t i = 0; i < expressions.size(); ++i)
        text += "\n" + render_device_source("nvrtc_name.cu.in", {
            {"index", std::to_string(i)}, {"expression", expressions[i]}});
    add(input, text);
    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(files);
    std::vector<std::string> args {"-triple", "nvptx64-nvidia-cuda", "-aux-triple", LLVM_HOST_TRIPLE,
        "-fcuda-is-device", "-x", "cuda", "-std=" + settings.standard,
        "-target-cpu", settings.architecture, "-target-sdk-version=12.8", "-fms-extensions",
        "-emit-llvm", "-disable-llvm-passes", "-disable-llvm-verifier", "-O1", "-nostdsysteminc", "-nobuiltininc",
        "-internal-isystem", root + "/builtins", "-include", root + "/builtins/preamble.h",
        "-internal-isystem", root + "/include",
        "-internal-isystem", root + "/public/cudart",
        "-internal-isystem", root + "/public/nvcc",
        "-internal-isystem", root + "/public/cuda",
        "-internal-isystem", root + "/public/cudadevrt",
        "-internal-isystem", root + "/public",
        "-internal-isystem", root + "/device",
        "-I", root + "/headers", "-Wno-ignored-attributes", input};
    if (settings.relocatable) args.push_back("-fgpu-rdc");
    for (const auto& include : settings.includes) args.insert(args.end(), {"-I", include});
    for (const auto& define : settings.defines) args.push_back("-D" + define);
    for (const auto& undefine : settings.undefines) args.push_back("-U" + undefine);
    std::vector<const char*> pointers;
    for (const auto& argument : args) pointers.push_back(argument.c_str());
    // Frontend consumers and diagnostic printers can retain these references
    // until CompilerInstance destruction, so their owners must outlive it.
    llvm::raw_string_ostream stream(log);
    auto invocation = std::make_shared<clang::CompilerInvocation>();
    // CompilerInstance otherwise prints its summary directly to stderr. NVRTC
    // owns all diagnostics in the program log instead of writing to the caller.
    invocation->getDiagnosticOpts().ShowCarets = false;
    clang::CompilerInstance compiler(invocation);
    compiler.createDiagnostics(new clang::TextDiagnosticPrinter(stream, invocation->getDiagnosticOpts()));
    if (!clang::CompilerInvocation::CreateFromArgs(*invocation, pointers, compiler.getDiagnostics()))
        throw std::runtime_error("NVRTC frontend configuration failed");
    compiler.setVirtualFileSystem(overlay);
    compiler.createFileManager();
    compiler.createSourceManager();
    nvrtc_output output;
    nvrtc_action action(context, expressions, output);
    if (!compiler.ExecuteAction(action)) throw std::runtime_error("NVRTC CUDA compilation failed");
    auto module = action.takeModule();
    if (!module) throw std::runtime_error("NVRTC device frontend produced no IR");
    materialize_data_aliases(*module, action.aliases);
    // CodeGen can cache an unused constant address while resolving a CUDA
    // builtin's static property getter. It is not a device-program reference;
    // discard these orphan constants before checking optional weak closure.
    for (auto& global : module->globals()) global.removeDeadConstantUsers();
    if (llvm::verifyModule(*module, &stream)) throw std::runtime_error("NVRTC frontend produced invalid IR");
    if (auto* barrier = module->getFunction("__cudavm_rtc_syncthreads")) {
        // Use the intrinsic's name and signature, never the Clang archive's
        // stale NVVM enum. Barrier 0 synchronizes the entire CTA with ordering.
        auto native = module->getOrInsertFunction("llvm.nvvm.barrier.cta.sync.aligned.all",
            llvm::FunctionType::get(llvm::Type::getVoidTy(context), {llvm::Type::getInt32Ty(context)}, false));
        std::vector<llvm::CallInst*> calls;
        for (auto* user : barrier->users()) {
            auto* call = llvm::dyn_cast<llvm::CallInst>(user);
            if (!call) throw std::runtime_error("taking the address of a block barrier is unsupported");
            calls.push_back(call);
        }
        for (auto* call : calls) {
            llvm::IRBuilder<> builder(call);
            builder.CreateCall(native, {builder.getInt32(0)});
            call->eraseFromParent();
        }
        barrier->eraseFromParent();
    }
    lower_device_conversions(*module);
    // Produce the same generic floating LLVM atomic as the AOT CUDA frontend.
    // Named declarations avoid target builtin enum mismatches in embedded Clang.
    for(auto type : {"float","double"}) {
        auto symbol=std::string("__cudavm_rtc_atomic_add_")+type;
        auto* wrapper=module->getFunction(symbol);if(!wrapper)continue;
        auto* real=std::string_view(type)=="float"?llvm::Type::getFloatTy(context):llvm::Type::getDoubleTy(context);
        if(!wrapper->isDeclaration() || wrapper->arg_size()!=2 || wrapper->getReturnType()!=real ||
            !wrapper->getFunctionType()->getParamType(0)->isPointerTy() || wrapper->getFunctionType()->getParamType(1)!=real)
            throw std::runtime_error("invalid NVRTC floating atomic signature: "+symbol);
        std::vector<llvm::CallInst*> calls;
        for(auto* user:wrapper->users()) {
            auto* call=llvm::dyn_cast<llvm::CallInst>(user);
            if(!call)throw std::runtime_error("taking the address of a floating atomic is unsupported");
            calls.push_back(call);
        }
        for(auto* call:calls) {
            llvm::IRBuilder<> builder(call);
            auto* atomic=builder.CreateAtomicRMW(llvm::AtomicRMWInst::FAdd,call->getArgOperand(0),call->getArgOperand(1),
                llvm::Align(std::string_view(type)=="float"?4:8),llvm::AtomicOrdering::Monotonic);
            call->replaceAllUsesWith(atomic);call->eraseFromParent();
        }
        wrapper->eraseFromParent();
    }
    for(auto mode : {"popc","and","or"}) {
        auto symbol=std::string("__cudavm_rtc_block_")+mode;
        auto* wrapper=module->getFunction(symbol);if(!wrapper)continue;
        auto* word=llvm::Type::getInt32Ty(context);
        if(!wrapper->isDeclaration() || wrapper->getFunctionType()!=llvm::FunctionType::get(word,{word},false))
            throw std::runtime_error("invalid NVRTC block vote signature: "+symbol);
        bool count=std::string_view(mode)=="popc";
        auto native=module->getOrInsertFunction(std::string("llvm.nvvm.barrier.cta.red.")+mode+".aligned.all",
            llvm::FunctionType::get(count?word:llvm::Type::getInt1Ty(context),{word,llvm::Type::getInt1Ty(context)},false));
        std::vector<llvm::CallInst*> calls;
        for(auto* user:wrapper->users()) {
            auto* call=llvm::dyn_cast<llvm::CallInst>(user);
            if(!call)throw std::runtime_error("taking the address of a block vote is unsupported");
            calls.push_back(call);
        }
        for(auto* call:calls) {
            llvm::IRBuilder<> builder(call);
            auto* vote=builder.CreateCall(native,{builder.getInt32(0),builder.CreateICmpNE(call->getArgOperand(0),builder.getInt32(0))});
            call->replaceAllUsesWith(count?vote:builder.CreateZExt(vote,word));call->eraseFromParent();
        }
        wrapper->eraseFromParent();
    }
    // The public wrappers use declarations with the intrinsic signatures, so
    // no target-specific Clang builtin IDs enter in-process CodeGen.
    for(auto mode : {"idx","up","down","bfly","vote_all","vote_any","vote_ballot"}) {
        std::string suffix=mode;
        bool vote=suffix.starts_with("vote_");
        auto symbol=std::string("__cudavm_rtc_")+(vote?"":"shfl_")+suffix;
        auto* wrapper=module->getFunction(symbol);
        if(!wrapper)continue;
        auto* word=llvm::Type::getInt32Ty(context);
        auto* predicate=llvm::Type::getInt1Ty(context);
        auto* result=vote && suffix!="vote_ballot"?predicate:word;
        auto* signature=llvm::FunctionType::get(result,vote?std::vector<llvm::Type*>{word,predicate}:std::vector<llvm::Type*>{word,word,word,word},false);
        if(!wrapper->isDeclaration() || wrapper->getFunctionType()!=signature)
            throw std::runtime_error("invalid NVRTC warp intrinsic signature: "+symbol);
        auto name=vote?"llvm.nvvm.vote."+suffix.substr(5)+".sync":"llvm.nvvm.shfl.sync."+suffix+".i32";
        auto native=module->getOrInsertFunction(name,signature);
        wrapper->replaceAllUsesWith(native.getCallee());wrapper->eraseFromParent();
    }
    if(auto* wrapper=module->getFunction("__cudavm_rtc_activemask")) {
        auto* signature=llvm::FunctionType::get(llvm::Type::getInt32Ty(context),{},false);
        if(!wrapper->isDeclaration() || wrapper->getFunctionType()!=signature)
            throw std::runtime_error("invalid NVRTC active mask signature");
        auto native=module->getOrInsertFunction("llvm.nvvm.activemask",signature);
        wrapper->replaceAllUsesWith(native.getCallee());wrapper->eraseFromParent();
    }
    if(auto* wrapper=module->getFunction("__cudavm_rtc_syncwarp")) {
        auto* signature=llvm::FunctionType::get(llvm::Type::getVoidTy(context),{llvm::Type::getInt32Ty(context)},false);
        if(!wrapper->isDeclaration() || wrapper->getFunctionType()!=signature)
            throw std::runtime_error("invalid NVRTC warp barrier signature");
        auto native=module->getOrInsertFunction("llvm.nvvm.bar.warp.sync",signature);
        wrapper->replaceAllUsesWith(native.getCallee());wrapper->eraseFromParent();
    }
    for(auto width : {32u,64u})for(bool all : {false,true}) {
        auto symbol=std::string("__cudavm_rtc_match_")+(all?"all":"any")+std::to_string(width);
        auto* wrapper=module->getFunction(symbol);if(!wrapper)continue;
        auto* word=llvm::Type::getInt32Ty(context);
        auto* bits=llvm::IntegerType::get(context,width);
        std::vector<llvm::Type*> params={word,bits};
        if(all)params.push_back(llvm::PointerType::get(context,0));
        if(!wrapper->isDeclaration() || wrapper->getFunctionType()!=llvm::FunctionType::get(word,params,false))
            throw std::runtime_error("invalid NVRTC warp match signature: "+symbol);
        auto* pair=llvm::StructType::get(context,{word,llvm::Type::getInt1Ty(context)});
        auto* signature=llvm::FunctionType::get(all?static_cast<llvm::Type*>(pair):word,{word,bits},false);
        auto native=module->getOrInsertFunction(std::string("llvm.nvvm.match.")+(all?"all":"any")+".sync.i"+std::to_string(width)+(all?"p":""),signature);
        if(!all)wrapper->replaceAllUsesWith(native.getCallee());
        else {
            std::vector<llvm::CallInst*> calls;
            for(auto* user:wrapper->users()) {
                auto* call=llvm::dyn_cast<llvm::CallInst>(user);
                if(!call)throw std::runtime_error("taking the address of a warp match is unsupported");
                calls.push_back(call);
            }
            for(auto* call:calls) {
                llvm::IRBuilder<> builder(call);
                auto* matched=builder.CreateCall(native,{call->getArgOperand(0),call->getArgOperand(1)});
                builder.CreateStore(builder.CreateZExt(builder.CreateExtractValue(matched,1),word),call->getArgOperand(2));
                call->replaceAllUsesWith(builder.CreateExtractValue(matched,0));call->eraseFromParent();
            }
        }
        wrapper->eraseFromParent();
    }
    for(auto mode : {"add","min","max","umin","umax","and","or","xor"}) {
        auto symbol=std::string("__cudavm_rtc_reduce_")+mode;
        auto* wrapper=module->getFunction(symbol);if(!wrapper)continue;
        auto* word=llvm::Type::getInt32Ty(context);
        auto* signature=llvm::FunctionType::get(word,{word,word},false);
        if(!wrapper->isDeclaration() || wrapper->getFunctionType()!=signature)
            throw std::runtime_error("invalid NVRTC warp reduction signature: "+symbol);
        auto native=module->getOrInsertFunction(std::string("llvm.nvvm.redux.sync.")+mode,signature);
        wrapper->replaceAllUsesWith(native.getCallee());wrapper->eraseFromParent();
    }
    lower_device_math(*module);
    lower_device_integers(*module);
    lower_device_synchronization(*module);
    std::vector<llvm::GlobalVariable*> anchors;
    for (auto& global : module->globals())
        if (global.getName().contains("__cudavm_nvrtc_name_")) anchors.push_back(&global);
    // Anchors force template instantiation during parsing, but are private
    // compiler bookkeeping and must never reach the returned PTX.
    llvm::removeFromUsedLists(*module, [&](llvm::Constant* value) {
        return value->stripPointerCasts()->getName().contains("__cudavm_nvrtc_name_");
    });
    for (auto* anchor : anchors) anchor->eraseFromParent();
    std::vector<llvm::GlobalValue*> named_roots;
    for (const auto& [expression, symbol] : output.lowered_names) {
        (void)expression;
        auto* value = module->getNamedValue(symbol);
        if (!value) throw std::runtime_error("NVRTC name expression has no device definition: " + symbol);
        named_roots.push_back(value);
    }
    llvm::appendToCompilerUsed(*module, named_roots);
    if (!settings.relocatable) resolve_weak_device_references(*module);
    // NVRTC is a source compiler, not the final device linker. NVIDIA also
    // permits external device declarations in default-mode PTX. Neither mode
    // may invoke the AOT graph closure or discard named device definitions.
    output.module = std::move(module);
    return output;
}
}
