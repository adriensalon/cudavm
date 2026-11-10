#include "driver.hpp"
#include <cstdio>
#include <stdexcept>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace cudavm {
std::string nvcc_quote_argument(const std::string& value) {
    std::string result = "\""; size_t slashes = 0;
    for (char c : value) {
        if (c == '\\') { ++slashes; continue; }
        result.append(slashes * (c == '"' ? 2 : 1), '\\'); slashes = 0;
        if (c == '"') result += '\\'; result += c;
    }
    result.append(slashes * 2, '\\'); return result + '"';
}
std::filesystem::path nvcc_executable_path() {
#if defined(_WIN32)
    std::wstring path(32768, L'\0'); auto size = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    if (!size || size >= path.size()) throw std::runtime_error("cannot locate compiler executable"); path.resize(size); return path;
#else
    std::string path(32768, '\0'); auto size = readlink("/proc/self/exe", path.data(), path.size());
    if (size < 0 || size_t(size) == path.size()) throw std::runtime_error("cannot locate compiler executable"); path.resize(size_t(size)); return path;
#endif
}
std::filesystem::path nvcc_find_program(const std::string& value) {
    std::filesystem::path path(value);
    if (path.has_parent_path()) {
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("host tool is unavailable: " + value);
        return std::filesystem::absolute(path);
    }
#if defined(_WIN32)
    std::wstring result(32768, L'\0'); auto size = SearchPathW(nullptr, path.c_str(), L".exe", DWORD(result.size()), result.data(), nullptr);
    if (!size || size >= result.size()) throw std::runtime_error("host tool is unavailable: " + value); result.resize(size); return result;
#else
    const char* environment = std::getenv("PATH"); std::string paths = environment ? environment : "";
    size_t begin = 0;
    do { auto end = paths.find(':', begin); auto candidate = std::filesystem::path(paths.substr(begin, end - begin)) / path;
        if (access(candidate.c_str(), X_OK) == 0) return std::filesystem::absolute(candidate);
        if (end == std::string::npos) break; begin = end + 1;
    } while (true);
    throw std::runtime_error("host tool is unavailable: " + value);
#endif
}
int nvcc_execute(const std::vector<std::string>& arguments, bool verbose) {
    if (arguments.empty()) throw std::runtime_error("empty host command");
    std::string command;
    for (const auto& argument : arguments) { if (!command.empty()) command += ' '; command += nvcc_quote_argument(argument); }
    if (verbose) { std::fprintf(stderr, "#$ %s\n", command.c_str()); std::fflush(stderr); }
#if defined(_WIN32)
    auto wide = std::filesystem::path(command).wstring(); auto executable = std::filesystem::path(arguments.front()).wstring();
    STARTUPINFOW startup{}; startup.cb = sizeof(startup); PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), wide.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process))
        throw std::runtime_error("cannot start host tool: " + arguments.front());
    CloseHandle(process.hThread); WaitForSingleObject(process.hProcess, INFINITE);
    DWORD result{}; GetExitCodeProcess(process.hProcess, &result); CloseHandle(process.hProcess); return int(result);
#else
    auto child = fork(); if (child < 0) throw std::runtime_error("cannot fork host tool");
    if (!child) { std::vector<char*> args; for (const auto& argument : arguments) args.push_back(const_cast<char*>(argument.c_str())); args.push_back(nullptr); execv(args[0], args.data()); _exit(127); }
    int result{}; if (waitpid(child, &result, 0) < 0) throw std::runtime_error("cannot wait for host tool"); return WIFEXITED(result) ? WEXITSTATUS(result) : 128 + WTERMSIG(result);
#endif
}
}
