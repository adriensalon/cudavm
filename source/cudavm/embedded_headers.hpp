#pragma once
#include <span>
#include <cstddef>
#include <llvm/Support/VirtualFileSystem.h>
namespace cudavm {
struct embedded_file { const char* path; const char* contents; size_t size; };
std::span<const embedded_file> embedded_headers();
struct frontend_resources;
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> frontend_files(const frontend_resources&);
}
