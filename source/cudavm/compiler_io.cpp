#include "compiler.hpp"
#include <stdexcept>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
namespace cudavm {
std::mutex& compiler_mutex() { static std::mutex value; return value; }
void write_file(const std::string& path, const std::string& content) {
    if (path.empty()) return;
    std::error_code error;
    llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
    if (error) throw std::runtime_error("cannot write " + path + ": " + error.message());
    stream << content;
    stream.flush();
    if (stream.has_error()) throw std::runtime_error("write failed: " + path);
}
void write_bitcode(const std::string& path, const llvm::Module& module) {
    if (path.empty()) return;
    std::error_code error;
    llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
    if (error) throw std::runtime_error("cannot write bitcode " + path + ": " + error.message());
    llvm::WriteBitcodeToFile(module, stream);
    stream.flush();
    if (stream.has_error()) throw std::runtime_error("bitcode write failed: " + path);
}
}
