#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <llvm/Support/JSON.h>
namespace llvm { class LLVMContext; class Module; }
namespace cudavm {
// Source, link and import entry points share LLVM target state in one library.
std::mutex& compiler_mutex();
struct frontend_resources {
    std::string origin, resource_dir, wrapper, cuda_include, cuda_crt_include, public_include;
};
frontend_resources resolve_resources(const std::string& root = {});
llvm::json::Object describe_resources(const frontend_resources&);
struct compiler_ir_input { std::string bytes, name; };
struct compiler_options {
    std::string input, host, bitcode, portable, ir, manifest, binary, target, depfile, module;
    std::string standard = "c++20", architecture = "sm_75";
    std::vector<std::string> inputs, includes, defines, library_inputs, host_inputs;
    std::vector<compiler_ir_input> ir_inputs;
    frontend_resources resources;
    mutable std::string diagnostics;
};
struct extracted_host_header { std::string source, output; };
struct host_extraction {
    std::string host;
    std::vector<extracted_host_header> headers;
    std::vector<std::string> dependencies;
    llvm::json::Array kernels, launches, registrations, symbols;
};
host_extraction extract_host(const compiler_options&);
std::unique_ptr<llvm::Module> compile_device(llvm::LLVMContext&, const compiler_options&, const std::vector<std::string>& host_dependencies = {}, const llvm::json::Array* host_symbols = nullptr);
void mark_translation_unit(llvm::Module&, const compiler_options&, const host_extraction&);
std::unique_ptr<llvm::Module> link_device(llvm::LLVMContext&, const compiler_options&);
void prepare_portable(llvm::Module&);
void lower_device_conversions(llvm::Module&);
void lower_device_math(llvm::Module&);
void lower_half_math_intrinsics(llvm::Module&);
void lower_device_integers(llvm::Module&);
void lower_device_synchronization(llvm::Module&);
// Final optional references and canonical target-plus-offset initializers.
void resolve_weak_device_references(llvm::Module&);
void normalize_device_relocations(llvm::Module&);
// Final linking resolves selected weak/ODR definitions. Native exports retain
// legal PTX helper aliases and materialize kernel/weak aliases as entry bodies.
void validate_device_function_aliases(const llvm::Module&, bool final_link = false);
void resolve_device_function_aliases(llvm::Module&, bool final_link = false, bool native_exports = false);
// AOT preserves replaced original backings; NVRTC resolves its module's views.
void validate_device_data_aliases(const llvm::Module&, bool final_link = false);
void resolve_device_data_aliases(llvm::Module&, bool final_link = false);
std::string emit_data_views(const llvm::Module&);
llvm::json::Object describe(const llvm::Module&);
std::string emit_cuda(const llvm::Module&, const std::string& architecture);
// Emit native frontend IR without closing the graph (relocatable PTX).
std::string emit_cuda_ir(const llvm::Module&, const std::string& architecture, std::string* target_data_layout = nullptr);
std::string emit_native_module(const llvm::Module&);
std::string translation_unit_id(const compiler_options&);
std::string embed_cuda(const llvm::Module&, const std::string& code);
void write_file(const std::string&, const std::string&);
void write_bitcode(const std::string&, const llvm::Module&);
std::vector<std::string> frontend_arguments(const compiler_options&, bool device);
}
