#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <cctype>
namespace cudavm {
struct trace_breakpoint { unsigned char original{}; std::string name; bool armed = false; };
struct trace_return { std::string name; uint64_t output{}, query{}; std::string detail; unsigned requested_version = 0; };
class runtime_trace {
    HANDLE process{};
    std::map<uint64_t, trace_breakpoint> points;
    std::map<DWORD, uint64_t> stepping;
    std::map<std::pair<DWORD, uint64_t>, std::vector<trace_return>> returns;
    std::ofstream log;
    template<class T> T read(uint64_t address) {
        T value{}; SIZE_T size{};
        if (!ReadProcessMemory(process, reinterpret_cast<void*>(address), &value, sizeof(value), &size) || size != sizeof(value))
            throw std::runtime_error("cannot read debuggee memory");
        return value;
    }
    std::string ansi(uint64_t address) {
        std::string value;
        for (size_t i = 0; address && i < 512; ++i) {
            auto c = read<char>(address + i); if (!c) break; value += c;
        }
        return value;
    }
    std::string wide(uint64_t address) {
        std::wstring value;
        for (size_t i = 0; address && i < 512; ++i) {
            auto c = read<wchar_t>(address + i * 2); if (!c) break; value += c;
        }
        return std::filesystem::path(value).string();
    }
    std::string uuid(uint64_t address) {
        std::string value; char byte[3];
        for (size_t i = 0; address && i < 16; ++i) { std::snprintf(byte, sizeof(byte), "%02x", read<unsigned char>(address + i)); value += byte; }
        return value;
    }
    void write(uint64_t address, unsigned char byte) {
        SIZE_T size{};
        if (!WriteProcessMemory(process, reinterpret_cast<void*>(address), &byte, 1, &size) || size != 1)
            throw std::runtime_error("cannot set trace breakpoint");
        FlushInstructionCache(process, reinterpret_cast<void*>(address), 1);
    }
    void add(uint64_t address, std::string name) {
        if (points.contains(address)) return;
        trace_breakpoint point{read<unsigned char>(address), std::move(name), true};
        points.emplace(address, std::move(point)); write(address, 0xcc);
    }
    void system_symbol(const std::string& module, uint64_t remote, const char* symbol) {
        auto local = GetModuleHandleA(module.c_str());
        if (!local) return;
        auto entry = GetProcAddress(local, symbol);
        if (entry && reinterpret_cast<uint64_t>(entry) - reinterpret_cast<uint64_t>(local) < 0x10000000)
            add(remote + reinterpret_cast<uint64_t>(entry) - reinterpret_cast<uint64_t>(local), symbol);
    }
    void load(HANDLE file, uint64_t base) {
        char buffer[32768]{};
        auto size = GetFinalPathNameByHandleA(file, buffer, sizeof(buffer), FILE_NAME_NORMALIZED);
        if (!size || size >= sizeof(buffer)) return;
        std::string path(buffer, size), name = std::filesystem::path(path).filename().string();
        for (auto& c : name) c = char(std::tolower(static_cast<unsigned char>(c)));
        log << "load " << std::hex << base << ' ' << path << '\n'; log.flush();
        if (name == "kernelbase.dll" || name == "kernel32.dll")
            for (auto symbol : {"LoadLibraryExA", "LoadLibraryExW", "LoadLibraryA", "LoadLibraryW", "GetProcAddress", "GetModuleHandleW"})
                system_symbol(name, base, symbol);
        if (name == "nvcuda.dll") {
            // Map exports from disk without executing DLL initialization in this tracer.
            auto local = LoadLibraryExA(path.c_str(), nullptr, DONT_RESOLVE_DLL_REFERENCES);
            if (!local) throw std::runtime_error("cannot map Driver export addresses for tracing");
            for (auto symbol : {"cuGetProcAddress", "cuGetProcAddress_v2", "cuGetExportTable", "cuInit", "cuDriverGetVersion"}) {
                auto entry = GetProcAddress(local, symbol);
                if (entry) add(base + reinterpret_cast<uint64_t>(entry) - reinterpret_cast<uint64_t>(local), symbol);
            }
            FreeLibrary(local);
        }
    }
    void entry(DWORD thread, uint64_t address, CONTEXT& context) {
        auto& point = points.at(address);
        auto key = std::make_pair(thread, address);
        if (auto found = returns.find(key); found != returns.end() && !found->second.empty()) {
            auto record = found->second.back(); found->second.pop_back();
            log << "return " << record.name << ' ' << record.detail << " status=" << std::hex << context.Rax;
            if (record.output) {
                auto output = read<uint64_t>(record.output); log << " output=" << output;
                if ((record.name == "cuGetProcAddress" || record.name == "cuGetProcAddress_v2") && !context.Rax && output) {
                    auto symbol = record.detail;
                    add(output, symbol == "cuGetProcAddress" || symbol == "cuGetExportTable" || symbol == "cuInit" || symbol == "cuDriverGetVersion"
                        ? symbol == "cuGetProcAddress" ? record.requested_version >= 12000 ? "cuGetProcAddress_v2" : "cuGetProcAddress" : symbol : "driver:" + symbol);
                }
                if (record.name == "cuGetExportTable" && !context.Rax && output) {
                    log << " table=";
                    auto bytes = read<uint64_t>(output);
                    const bool storage = record.detail == "c693336e1121df11a8c368f355d89593";
                    for (size_t i = 0; i < (storage ? 3 : bytes < 4096 ? bytes / 8 : 0); ++i) {
                        auto entry = read<uint64_t>(output + 8 * i); log << entry << ',';
                        MEMORY_BASIC_INFORMATION information{};
                        if ((storage || i) && entry && VirtualQueryEx(process, reinterpret_cast<void*>(entry), &information, sizeof(information))
                            && (information.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)))
                            add(entry, "table:" + record.detail + ":" + std::to_string(i));
                    }
                }
            }
            if (record.query) log << " query=" << read<unsigned>(record.query);
            log << '\n';
        }
        if (point.name.empty()) return;
        auto name = point.name;
        std::string detail;
        trace_return result{name};
        if (name == "GetProcAddress") {
            if (context.Rdx > 0xffff) detail = ansi(context.Rdx); else detail = "ordinal";
            log << "call GetProcAddress module=" << std::hex << context.Rcx << " symbol=" << detail << '\n';
        } else if (name == "LoadLibraryExA" || name == "LoadLibraryExW" || name == "LoadLibraryA" || name == "LoadLibraryW" || name == "GetModuleHandleW") {
            detail = name.ends_with('W') ? wide(context.Rcx) : ansi(context.Rcx);
            log << "call " << name << " path=" << detail << " flags=" << std::hex << context.R8 << '\n';
        } else if (name == "cuGetProcAddress" || name == "cuGetProcAddress_v2") {
            detail = ansi(context.Rcx); result.output = context.Rdx;
            result.requested_version = unsigned(context.R8);
            if (name.ends_with("_v2")) result.query = read<uint64_t>(context.Rsp + 40);
            log << "call " << name << " symbol=" << detail << " version=" << std::dec << context.R8 << " flags=" << context.R9 << '\n';
        } else if (name == "cuGetExportTable") {
            detail = uuid(context.Rdx); result.output = context.Rcx;
            log << "call " << name << " uuid=" << detail << '\n';
        } else {
            log << "call " << name << " rcx=" << std::hex << context.Rcx << " rdx=" << context.Rdx << " r8=" << context.R8 << " r9=" << context.R9 << " stack=";
            for (size_t i = 0; i < 5; ++i) log << read<uint64_t>(context.Rsp + 40 + 8 * i) << ',';
            log << '\n';
        }
        result.detail = detail;
        auto return_address = read<uint64_t>(context.Rsp);
        add(return_address, ""); returns[{thread, return_address}].push_back(result);
        log.flush();
    }
    static std::wstring quote(std::wstring_view argument) {
        std::wstring result = L"\""; size_t slashes = 0;
        for (auto c : argument) {
            if (c == L'\\') { ++slashes; continue; }
            result.append(slashes * (c == L'\"' ? 2 : 1), L'\\'); slashes = 0;
            if (c == L'\"') result += L'\\';
            result += c;
        }
        result.append(slashes * 2, L'\\'); result += L'\"'; return result;
    }
public:
    ~runtime_trace() {
        if (process) { TerminateProcess(process, 124); CloseHandle(process); }
    }
    int run(int argc, wchar_t** argv) {
        const auto* executable = argv[1];
        auto output = std::filesystem::path(argv[2]);
        log.open(output);
        if (!log) throw std::runtime_error("cannot create audit log");
        log << "trace_version=1 instrumentation=debugger-breakpoints\n";
        STARTUPINFOW startup{}; startup.cb = sizeof(startup); PROCESS_INFORMATION info{};
        std::wstring command = quote(executable);
        for (int i = 3; i < argc; ++i) command += L" " + quote(argv[i]);
        if (!CreateProcessW(executable, command.data(), nullptr, nullptr, FALSE, DEBUG_ONLY_THIS_PROCESS, nullptr, nullptr, &startup, &info))
            throw std::runtime_error("cannot start audited process");
        process = info.hProcess; CloseHandle(info.hThread);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (std::chrono::steady_clock::now() < deadline) {
            DEBUG_EVENT event{};
            if (!WaitForDebugEvent(&event, 500)) continue;
            DWORD action = DBG_CONTINUE;
            if (event.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT) {
                load(event.u.CreateProcessInfo.hFile, reinterpret_cast<uint64_t>(event.u.CreateProcessInfo.lpBaseOfImage));
                CloseHandle(event.u.CreateProcessInfo.hFile);
                CloseHandle(event.u.CreateProcessInfo.hThread);
                CloseHandle(event.u.CreateProcessInfo.hProcess);
            } else if (event.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT) {
                load(event.u.LoadDll.hFile, reinterpret_cast<uint64_t>(event.u.LoadDll.lpBaseOfDll));
                if (event.u.LoadDll.hFile) CloseHandle(event.u.LoadDll.hFile);
            } else if (event.dwDebugEventCode == CREATE_THREAD_DEBUG_EVENT) {
                CloseHandle(event.u.CreateThread.hThread);
            } else if (event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT) {
                auto code = event.u.Exception.ExceptionRecord.ExceptionCode;
                auto address = reinterpret_cast<uint64_t>(event.u.Exception.ExceptionRecord.ExceptionAddress);
                if (code == EXCEPTION_BREAKPOINT && points.contains(address)) {
                    auto thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, event.dwThreadId);
                    CONTEXT context{}; context.ContextFlags = CONTEXT_FULL;
                    if (!GetThreadContext(thread, &context)) throw std::runtime_error("cannot read traced registers");
                    write(address, points.at(address).original); points.at(address).armed = false;
                    entry(event.dwThreadId, address, context);
                    context.Rip = address; context.EFlags |= 0x100;
                    if (!SetThreadContext(thread, &context)) throw std::runtime_error("cannot resume traced registers");
                    CloseHandle(thread); stepping[event.dwThreadId] = address;
                } else if (code == EXCEPTION_SINGLE_STEP && stepping.contains(event.dwThreadId)) {
                    auto address = stepping.at(event.dwThreadId); write(address, 0xcc); points.at(address).armed = true;
                    auto thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, event.dwThreadId);
                    CONTEXT context{}; context.ContextFlags = CONTEXT_CONTROL;
                    if (!GetThreadContext(thread, &context)) throw std::runtime_error("cannot finish traced instruction");
                    context.EFlags &= ~0x100u;
                    if (!SetThreadContext(thread, &context)) throw std::runtime_error("cannot clear trace flag");
                    CloseHandle(thread);
                    stepping.erase(event.dwThreadId);
                } else if (code != EXCEPTION_BREAKPOINT) {
                    log << "exception " << std::hex << code << " address=" << address << '\n'; log.flush();
                    action = DBG_EXCEPTION_NOT_HANDLED;
                }
            } else if (event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT) {
                log << "exit " << std::dec << event.u.ExitProcess.dwExitCode << '\n';
                ContinueDebugEvent(event.dwProcessId, event.dwThreadId, action);
                CloseHandle(process); process = nullptr; return int(event.u.ExitProcess.dwExitCode);
            }
            ContinueDebugEvent(event.dwProcessId, event.dwThreadId, action);
        }
        throw std::runtime_error("audit deadline exceeded");
    }
};
}
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc < 3) return 2;
        return cudavm::runtime_trace{}.run(argc, argv);
    } catch (const std::exception& error) { std::fprintf(stderr, "trace: %s\n", error.what()); return 2; }
}
