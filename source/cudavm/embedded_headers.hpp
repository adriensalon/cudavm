#pragma once
#include <span>
#include <cstddef>
#include <string>
#include <string_view>
#include <llvm/Support/VirtualFileSystem.h>
namespace cudavm {
struct embedded_file { const char* path; const char* contents; size_t size; };
std::span<const embedded_file> embedded_files();
std::string_view embedded_source(std::string_view path);
void add_embedded_files(llvm::vfs::InMemoryFileSystem&, const std::string& root);
struct frontend_resources;
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> frontend_files(const frontend_resources&);
}
