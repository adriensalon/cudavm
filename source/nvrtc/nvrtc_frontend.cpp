#include "nvrtc_frontend.hpp"
#include "../cudavm/spirv_emitter.hpp"
#include "../cudavm/dxil_emitter.hpp"
#include "data_alias_frontend.hpp"
#include "../cudavm/device_conversions.hpp"
#include "../cudavm/device_integers.hpp"
#include "../cudavm/device_simd.hpp"
#include "../cudavm/device_dpx.hpp"
#include "../cudavm/device_math.hpp"
#include "../cudavm/device_warp.hpp"
#include "../cudavm/device_math_source.hpp"
#include "nvrtc_headers.hpp"
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
constexpr auto preamble = R"cudavm(
#pragma clang system_header
#define __CUDACC__ 1
#define __CUDACC_RTC__ 1
#define __CUDACC_VER_MAJOR__ 13
#define __CUDACC_VER_MINOR__ 4
#define __CUDACC_VER_BUILD__ 1
#define CUDA_VERSION 13040
#define __FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__ 1
#if __has_include(<nv/target>)
#include <nv/target>
#endif
#include <vector_types.h>
#include <__clang_cuda_builtin_vars.h>
typedef __SIZE_TYPE__ size_t;
#include <cuda_runtime_api.h>
#include <__clang_cuda_texture_intrinsics.h>
#include <texture_indirect_functions.h>
#include <surface_indirect_functions.h>
__device__ inline float2 make_float2(float x,float y) { return float2{x,y}; }
// NVRTC supplies this utility to the CUDA numeric headers without a host C++
// standard library. The lvalue overload also coexists with a caller's utility.
namespace std { template<class T> __device__ constexpr T&& move(T& value) { return static_cast<T&&>(value); } }
#ifndef NULL
#define NULL __null
#endif
__device__ inline __cuda_builtin_threadIdx_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_threadIdx_t::operator uint3() const { return uint3{x,y,z}; }
__device__ inline __cuda_builtin_blockIdx_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_blockIdx_t::operator uint3() const { return uint3{x,y,z}; }
__device__ inline __cuda_builtin_blockDim_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_blockDim_t::operator uint3() const { return uint3{x,y,z}; }
__device__ inline __cuda_builtin_gridDim_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_gridDim_t::operator uint3() const { return uint3{x,y,z}; }
extern "C" __device__ void __cudavm_rtc_syncthreads();
#define __syncthreads() __cudavm_rtc_syncthreads()
extern "C" __device__ void __trap();
extern "C" __device__ inline void* memcpy(void* destination,const void* source,__SIZE_TYPE__ bytes) {return __builtin_memcpy(destination,source,bytes);}
extern "C" __device__ __attribute__((convergent)) void __cudavm_rtc_barrier_sync(unsigned);
extern "C" __device__ __attribute__((convergent)) void __cudavm_rtc_barrier_count(unsigned,unsigned);
__device__ inline void __barrier_sync(unsigned id) {__cudavm_rtc_barrier_sync(id);}
__device__ inline void __barrier_sync_count(unsigned id,unsigned count) {__cudavm_rtc_barrier_count(id,count);}
extern "C" __device__ void __cudavm_rtc_fence_block();
extern "C" __device__ void __cudavm_rtc_fence_device();
extern "C" __device__ void __cudavm_rtc_fence_system();
__device__ inline void __threadfence_block() {__cudavm_rtc_fence_block();}
__device__ inline void __threadfence() {__cudavm_rtc_fence_device();}
__device__ inline void __threadfence_system() {__cudavm_rtc_fence_system();}
extern "C" __device__ __attribute__((convergent)) int __cudavm_rtc_block_popc(int);
extern "C" __device__ __attribute__((convergent)) int __cudavm_rtc_block_and(int);
extern "C" __device__ __attribute__((convergent)) int __cudavm_rtc_block_or(int);
__device__ inline int __syncthreads_count(int p) {return __cudavm_rtc_block_popc(p);}
__device__ inline int __syncthreads_and(int p) {return __cudavm_rtc_block_and(p);}
__device__ inline int __syncthreads_or(int p) {return __cudavm_rtc_block_or(p);}
// Generic Clang builtins produce portable LLVM integer intrinsics. The zero
// guards retain CUDA's defined clz/ffs results without target builtin enums.
__device__ inline int __popc(unsigned x) {return __builtin_popcount(x);}
__device__ inline int __popcll(unsigned long long x) {return __builtin_popcountll(x);}
__device__ inline int __clz(int x) {return x?__builtin_clz(unsigned(x)):32;}
__device__ inline int __clzll(long long x) {return x?__builtin_clzll(static_cast<unsigned long long>(x)):64;}
__device__ inline int __ffs(int x) {return __builtin_ffs(x);}
__device__ inline int __ffsll(long long x) {return __builtin_ffsll(x);}
__device__ inline unsigned __fns(unsigned mask,unsigned base,int offset) {
    if(offset==0)return base;
    unsigned remaining=offset<0?0u-unsigned(offset):unsigned(offset);
    for(unsigned i=0;i<32;++i) {
        unsigned lane=offset<0?base-i:base+i;
        if(lane<32u && (mask&(1u<<lane)) && --remaining==0u)return lane;
    }
    return 0xffffffffu;
}
__device__ inline unsigned __brev(unsigned x) {return __builtin_bitreverse32(x);}
__device__ inline unsigned long long __brevll(unsigned long long x) {return __builtin_bitreverse64(x);}
__device__ inline float __uint_as_float(unsigned x) {return __builtin_bit_cast(float,x);}
__device__ inline float __int_as_float(int x) {return __builtin_bit_cast(float,x);}
__device__ inline unsigned __float_as_uint(float x) {return __builtin_bit_cast(unsigned,x);}
__device__ inline int __float_as_int(float x) {return __builtin_bit_cast(int,x);}
__device__ inline double __longlong_as_double(long long x) {return __builtin_bit_cast(double,x);}
__device__ inline long long __double_as_longlong(double x) {return __builtin_bit_cast(long long,x);}
__device__ inline float __fmaf_rn(float x,float y,float z) {return __builtin_elementwise_fma(x,y,z);}
__device__ inline float __fsub_rn(float x,float y) {return x-y;}
// Parsed by the half header's optional transcendental wrappers. A caller using
// it still needs a device implementation, like other unresolved NVRTC calls.
extern "C" __device__ float tanhf(float);
extern "C" __device__ unsigned __nv_byte_perm(unsigned,unsigned,unsigned);
__device__ inline unsigned __byte_perm(unsigned x,unsigned y,unsigned s) {return __nv_byte_perm(x,y,s);}
__device__ inline unsigned __funnelshift_l(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.l.wrap.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
__device__ inline unsigned __funnelshift_lc(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.l.clamp.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
__device__ inline unsigned __funnelshift_r(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.r.wrap.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
__device__ inline unsigned __funnelshift_rc(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.r.clamp.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
// Use generic Clang atomic builtins, which emit ordinary LLVM atomics without
// depending on the target-specific NVVM builtin enum or an installed toolkit.
__device__ inline unsigned atomicAdd(unsigned* p,unsigned v) {return __atomic_fetch_add(p,v,__ATOMIC_RELAXED);}
__device__ inline unsigned atomicOr(unsigned* p,unsigned v) {return __atomic_fetch_or(p,v,__ATOMIC_RELAXED);}
__device__ inline unsigned atomicAnd(unsigned* p,unsigned v) {return __atomic_fetch_and(p,v,__ATOMIC_RELAXED);}
__device__ inline unsigned atomicCAS(unsigned* p,unsigned c,unsigned v) {__atomic_compare_exchange_n(p,&c,v,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED);return c;}
__device__ inline unsigned min(unsigned a,unsigned b) {return a<b?a:b;}
__device__ inline unsigned max(unsigned a,unsigned b) {return a>b?a:b;}
extern "C" __device__ float __cudavm_rtc_atomic_add_float(float*,float);
extern "C" __device__ double __cudavm_rtc_atomic_add_double(double*,double);
__device__ inline float atomicAdd(float* p,float v) { return __cudavm_rtc_atomic_add_float(p,v); }
__device__ inline double atomicAdd(double* p,double v) { return __cudavm_rtc_atomic_add_double(p,v); }
__device__ inline float atomicExch(float* p,float v) { union {float f;unsigned u;} bits;bits.f=v;bits.u=__atomic_exchange_n(reinterpret_cast<unsigned*>(p),bits.u,__ATOMIC_RELAXED);return bits.f; }
__device__ inline unsigned long long atomicAdd(unsigned long long* p,unsigned long long v) { return __atomic_fetch_add(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicExch(unsigned long long* p,unsigned long long v) { return __atomic_exchange_n(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicCAS(unsigned long long* p,unsigned long long c,unsigned long long v) { __atomic_compare_exchange_n(p,&c,v,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED);return c; }
__device__ inline unsigned long long atomicAnd(unsigned long long* p,unsigned long long v) { return __atomic_fetch_and(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicOr(unsigned long long* p,unsigned long long v) { return __atomic_fetch_or(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicXor(unsigned long long* p,unsigned long long v) { return __atomic_fetch_xor(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicMin(unsigned long long* p,unsigned long long v) { return __atomic_fetch_min(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicMax(unsigned long long* p,unsigned long long v) { return __atomic_fetch_max(p,v,__ATOMIC_RELAXED); }
__device__ inline long long atomicMin(long long* p,long long v) { return __atomic_fetch_min(p,v,__ATOMIC_RELAXED); }
__device__ inline long long atomicMax(long long* p,long long v) { return __atomic_fetch_max(p,v,__ATOMIC_RELAXED); }
)cudavm";
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
nvrtc_output compile_nvrtc_source(const std::string& source, const std::string& name,
    const std::vector<std::pair<std::string, std::string>>& headers,
    const std::vector<std::string>& expressions, const nvrtc_options& settings, std::string& log) {
    // The current in-process LLVM pipeline has shared target state. Keep
    // concurrent public programs isolated while their compiler requests run.
    static std::mutex compilation_mutex;
    std::lock_guard compilation_lock(compilation_mutex);
    // Each request has its own virtual files, diagnostics, frontend and IR context.
    // Neither source nor virtual headers are materialized on disk.
    auto root = (std::filesystem::current_path() / "__cudavm_nvrtc_virtual__").generic_string();
    auto files = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    auto add = [&](const std::string& path, llvm::StringRef data) {
        if (!files->addFile(path, 0, llvm::MemoryBuffer::getMemBufferCopy(data)))
            throw std::runtime_error("conflicting NVRTC virtual file: " + path);
    };
    add(root + "/builtins/vector_types.h", nvrtc_vector_types);
    add(root + "/builtins/crt/host_defines.h", nvrtc_host_defines);
    add(root + "/builtins/__clang_cuda_builtin_vars.h", nvrtc_builtin_variables);
    for (const auto& header : nvrtc_image_headers) add(root + "/builtins/" + header.path, {header.contents, header.size});
    std::string rtc_preamble = preamble;
    for(auto operation:device_simd_operations) {
        std::string name=operation.name,arguments="unsigned int x";
        if(operation.arguments==2)arguments+=",unsigned int y";
        rtc_preamble+="extern \"C\" __device__ unsigned int __nv_"+name+"("+arguments+");\n";
        rtc_preamble+="__device__ inline unsigned int __"+name+"("+arguments+") {return __nv_"+name+
            "(x"+(operation.arguments==2?",y":"")+");}\n";
    }
    for(auto operation:device_dot_operations) {
        std::string word=operation.signed_value?"int":"unsigned int";
        std::string arguments=word+" x,"+word+" y,"+word+" c";
        rtc_preamble+="extern \"C\" __device__ "+word+" "+operation.symbol+"("+arguments+");\n";
        rtc_preamble+="__device__ inline "+word+" __"+operation.name+"("+arguments+") {return "+operation.symbol+"(x,y,c);}\n";
        std::string vector_a=operation.high<0?(operation.signed_value?"char4":"uchar4"):(operation.signed_value?"short2":"ushort2");
        std::string vector_b=operation.signed_value?"char4":"uchar4";
        rtc_preamble+="__device__ inline "+word+" __"+operation.name+"("+vector_a+" x,"+vector_b+" y,"+word+
            " c) {return "+operation.symbol+"(__builtin_bit_cast("+word+",x),__builtin_bit_cast("+word+",y),c);}\n";
    }
    for(auto operation:device_integer_operations) {
        std::string name=operation.name;
        std::string word=operation.bits==64?"long long":"int";
        if(name.starts_with("u"))word="unsigned "+word;
        std::string result=operation.arguments==3?"unsigned int":word;
        std::string arguments=word+" x,"+word+" y";
        if(operation.arguments==3)arguments+=",unsigned int z";
        rtc_preamble+="extern \"C\" __device__ "+result+" __nv_"+name+"("+arguments+");\n";
        rtc_preamble+="__device__ inline "+result+" __"+name+"("+arguments+") {return __nv_"+name+
            "(x,y"+(operation.arguments==3?",z":"")+");}\n";
    }
    for(auto name:{"hadd","rhadd","uhadd","urhadd"}) {
        std::string word=std::string_view(name).starts_with("u")?"unsigned int":"int";
        std::string arguments=word+" x,"+word+" y";
        rtc_preamble+="extern \"C\" __device__ "+word+" __nv_"+name+"("+arguments+");\n";
        rtc_preamble+="__device__ inline "+word+" __"+name+"("+arguments+") {return __nv_"+name+"(x,y);}\n";
    }
    add(root+"/builtins/cudavm_device_dpx.h",device_dpx_header());
    rtc_preamble+="#include <cudavm_device_dpx.h>\n";
    add(root+"/builtins/cudavm_device_warp.h",device_warp_header());
    rtc_preamble+="#include <cudavm_device_warp.h>\n";
    add(root+"/builtins/cudavm_device_math.h",device_math_header());
    rtc_preamble+="#include <cudavm_device_math.h>\n";
    for(auto operation:device_math_operations) {
        std::string name=operation.name;
        std::string arguments=operation.arguments==1?"float x":"float x, float y";
        std::string call=operation.arguments==1?"x":"x, y";
        rtc_preamble+="extern \"C\" __device__ float __nv_"+name+"("+arguments+");\n";
        rtc_preamble+="extern \"C\" __device__ inline float "+name+"("+arguments+") {return __nv_"+name+"("+call+");}\n";
    }
    for (const auto& conversion : device_conversions) {
        const std::string integer = std::string(conversion.signed_value ? "" : "unsigned ")
            + (conversion.integer_bits == 32 ? "int" : "long long");
        const std::string real = conversion.real_bits == 64 ? "double" : "float";
        const std::string result = conversion.to_real ? real : integer;
        const std::string argument = conversion.to_real ? integer : real;
        const std::string name = conversion.name;
        rtc_preamble += "extern \"C\" __device__ " + result + " __nv_" + name + "(" + argument + ");\n";
        rtc_preamble += "extern \"C\" __device__ inline " + result + " __" + name + "(" + argument
            + " x) { return __nv_" + name + "(x); }\n";
    }
    for (const auto& conversion : double_conversions) {
        const std::string name = conversion.name;
        rtc_preamble += "extern \"C\" __device__ float __nv_" + name + "(double);\n";
        rtc_preamble += "extern \"C\" __device__ inline float __" + name
            + "(double x) { return __nv_" + name + "(x); }\n";
    }
    add(root + "/builtins/preamble.h", rtc_preamble);
    for (const auto& [header, contents] : headers) add(root + "/headers/" + header, contents);
    auto input = (std::filesystem::path(root) / "source" / name).generic_string();
    std::string text = source;
    for (size_t i = 0; i < expressions.size(); ++i)
        text += "\n__device__ __attribute__((used)) auto __cudavm_nvrtc_name_" + std::to_string(i) + " = (" + expressions[i] + ");\n";
    add(input, text);
    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(files);
    std::vector<std::string> args {"-triple", "nvptx64-nvidia-cuda", "-aux-triple", LLVM_HOST_TRIPLE,
        "-fcuda-is-device", "-x", "cuda", "-std=" + settings.standard,
        "-target-cpu", settings.architecture, "-target-sdk-version=12.8", "-fms-extensions",
        "-emit-llvm", "-disable-llvm-passes", "-disable-llvm-verifier", "-O1", "-nostdsysteminc", "-nobuiltininc",
        "-internal-isystem", root + "/builtins", "-include", root + "/builtins/preamble.h",
        "-I", root + "/headers", "-Wno-ignored-attributes", input};
    if (settings.relocatable) args.push_back("-fgpu-rdc");
    for (const auto& include : settings.includes) args.insert(args.end(), {"-I", include});
    for (const auto& define : settings.defines) args.push_back("-D" + define);
    for (const auto& undefine : settings.undefines) args.push_back("-U" + undefine);
    std::vector<const char*> pointers;
    for (const auto& argument : args) pointers.push_back(argument.c_str());
    // Frontend consumers and diagnostic printers can retain these references
    // until CompilerInstance destruction, so their owners must outlive it.
    llvm::LLVMContext context;
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
    output.ptx = emit_cuda_ir(*module, settings.architecture);
    output.ptx += emit_native_spirv_payload(*module);
    output.ptx += emit_native_dxil_payload(*module);
    output.ptx += emit_native_module(*module);
    return output;
}
}
