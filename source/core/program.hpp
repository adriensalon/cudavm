#pragma once
#include "compiler.hpp"
#include "program_api.hpp"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

namespace cudavm {
// Context precedes module so LLVM values are destroyed before their owner.
struct compiler_program {
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
    std::string architecture = "sm_75";
    std::map<std::string, std::string, std::less<>> lowered_names;
};
std::unique_ptr<compiler_program> import_compiler_program(const compiler_input_request&, compiler_result&);
std::unique_ptr<compiler_program> link_compiler_programs(size_t, const compiler_program* const*, compiler_result&);
std::unique_ptr<compiler_program> prepare_compiler_program(const compiler_program&, const compiler_pipeline_request&);
void emit_compiler_program(const compiler_program&, const compiler_emit_request&, compiler_result&);
}
