#include "device_numeric_ir.hpp"
#include "device_math_source.hpp"
#include <stdexcept>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

namespace cudavm {
void link_device_numeric_ir(llvm::Module& module) {
    auto files=llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    auto source=std::string("#define __device__ __attribute__((device))\n")+device_math_header();
    files->addFile("/cudavm_numeric.cu",0,llvm::MemoryBuffer::getMemBufferCopy(source));
    const char* args[]={"-triple","nvptx64-nvidia-cuda","-fcuda-is-device","-x","cuda","-std=c++17",
        "-target-cpu","sm_75","-emit-llvm","-disable-llvm-passes","-O1","-nostdsysteminc","-nobuiltininc","/cudavm_numeric.cu"};
    std::string log;llvm::raw_string_ostream stream(log);
    auto invocation=std::make_shared<clang::CompilerInvocation>();
    clang::CompilerInstance compiler(invocation);
    compiler.createDiagnostics(new clang::TextDiagnosticPrinter(stream,invocation->getDiagnosticOpts()));
    if(!clang::CompilerInvocation::CreateFromArgs(*invocation,args,compiler.getDiagnostics()))throw std::runtime_error(log);
    compiler.setVirtualFileSystem(files);compiler.createFileManager();compiler.createSourceManager();
    clang::EmitLLVMOnlyAction action(&module.getContext());
    if(!compiler.ExecuteAction(action))throw std::runtime_error("numeric source compilation failed: "+log);
    auto numeric=action.takeModule();
    if(!numeric)throw std::runtime_error("numeric frontend produced no IR");
    numeric->setDataLayout(module.getDataLayout());
    if(llvm::Linker::linkModules(module,std::move(numeric)))throw std::runtime_error("numeric source linking failed");
}
}
