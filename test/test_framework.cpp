#include "test_framework.hpp"
#ifdef CUDAVM_CONFORMANCE_SOURCE_CHECK
#include "test_fixtures.hpp"
#include <nvrtc.h>
#endif
#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
#include <cuda/cudavm_cuda.h>
#ifdef CUDAVM_CONFORMANCE_EXECUTION
#include <nvrtc.h>
#ifndef CUDAVM_CONFORMANCE_SOURCE_ONLY
#include <nvvm.h>
#include <nvJitLink.h>
#endif
#endif
#endif
#include <algorithm>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <csignal>
#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
#include <dlfcn.h>
#endif
#include <sys/wait.h>
#include <sys/socket.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#endif

namespace cudavm {
#ifdef CUDAVM_CONFORMANCE_REGISTRATION
template<class T> void registration_word(std::vector<unsigned char>& bytes, size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
std::vector<unsigned char> registration_ptx_image(std::string_view source, unsigned major, unsigned minor, unsigned architecture) {
    constexpr size_t header_bytes = 72;
    const size_t payload_bytes = (source.size() + 1 + 7) & ~size_t(7);
    std::vector<unsigned char> image(16 + header_bytes + payload_bytes);
    registration_word<uint32_t>(image, 0, 0xba55ed50);
    registration_word<uint16_t>(image, 4, 1); registration_word<uint16_t>(image, 6, 16);
    registration_word<uint64_t>(image, 8, header_bytes + payload_bytes);
    registration_word<uint16_t>(image, 16, 1); registration_word<uint16_t>(image, 18, 0x101);
    registration_word<uint32_t>(image, 20, header_bytes); registration_word<uint64_t>(image, 24, payload_bytes);
    registration_word<uint32_t>(image, 36, 64); registration_word<uint32_t>(image, 40, (major << 16) | minor);
    registration_word<uint32_t>(image, 44, architecture);
#ifdef _WIN32
    registration_word<uint64_t>(image, 56, 0x41);
#else
    registration_word<uint64_t>(image, 56, 0x11);
#endif
    std::memcpy(image.data() + 16 + header_bytes, source.data(), source.size());
    return image;
}
#endif
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
#ifdef CUDAVM_CONFORMANCE_NVRTC
compiler_fixture_program::compiler_fixture_program(const std::string& source, const char* name) {
    check(nvrtcCreateProgram(&value, source.c_str(), name, 0, nullptr, nullptr));
}
compiler_fixture_program::~compiler_fixture_program() { if (value) nvrtcDestroyProgram(&value); }
void compiler_fixture_program::check(nvrtcResult result) {
    require(result == NVRTC_SUCCESS, std::string("compiler fixture API: ") + nvrtcGetErrorString(result));
}
nvrtcResult compiler_fixture_program::compile(const std::vector<std::string>& options) const {
    std::vector<const char*> pointers;
    for (const auto& option : options) pointers.push_back(option.c_str());
    return nvrtcCompileProgram(value, int(pointers.size()), pointers.data());
}
std::string compiler_fixture_program::text(bool log) const {
    struct { size_t before = 0x17593bad; size_t value{}; size_t after = 0xa739e521; } size;
    check(log ? nvrtcGetProgramLogSize(value, &size.value) : nvrtcGetPTXSize(value, &size.value));
    require(size.before == 0x17593bad && size.after == 0xa739e521, "compiler text size output guards");
    require(size.value > 0 && size.value < 64 * 1024 * 1024, "compiler text extent includes NUL and is bounded");
    std::vector<char> bytes(size.value + 32, char(0x63));
    check(log ? nvrtcGetProgramLog(value, bytes.data() + 16) : nvrtcGetPTX(value, bytes.data() + 16));
    for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= size.value + 16)
        require(bytes[i] == char(0x63), "compiler text retrieval exact extent guards");
    require(bytes[size.value + 15] == 0
        && std::find(bytes.begin() + 16, bytes.begin() + size.value + 15, '\0') == bytes.begin() + size.value + 15,
        "compiler text has exactly its reported terminal NUL");
    return {bytes.begin() + 16, bytes.begin() + size.value + 15};
}
std::string compiler_fixture_program::success(const std::vector<std::string>& options) const {
    const auto result = compile(options);
    require(result == NVRTC_SUCCESS, std::string(nvrtcGetErrorString(result)) + ": " + text(true));
    return text();
}
std::vector<unsigned char> compiler_fixture_program::lto() const {
    return compiler_fixture_bytes([&](size_t* size) { check(nvrtcGetLTOIRSize(value, size)); },
        [&](unsigned char* bytes) { check(nvrtcGetLTOIR(value, reinterpret_cast<char*>(bytes))); });
}
#endif
#ifdef CUDAVM_CONFORMANCE_NVJITLINK
linker_fixture_program::linker_fixture_program(const std::vector<std::string>& options) {
    std::vector<const char*> pointers;
    for (const auto& option : options) pointers.push_back(option.c_str());
    const auto status = nvJitLinkCreate(&value, uint32_t(pointers.size()), pointers.data());
    try { check(status, "create"); }
    catch (...) { if (value) nvJitLinkDestroy(&value); throw; }
}
linker_fixture_program::~linker_fixture_program() { if (value) nvJitLinkDestroy(&value); }
void linker_fixture_program::check(nvJitLinkResult status, const char* operation) const {
    if (status == NVJITLINK_SUCCESS) return;
    size_t size{};
    std::string diagnostic;
    if (value && nvJitLinkGetErrorLogSize(value, &size) == NVJITLINK_SUCCESS && size && size < 64 * 1024 * 1024) {
        std::vector<char> log(size);
        if (nvJitLinkGetErrorLog(value, log.data()) == NVJITLINK_SUCCESS)
            diagnostic.assign(log.begin(), std::find(log.begin(), log.end(), '\0'));
    }
    throw std::runtime_error(std::string("linker fixture ") + operation + ": " + std::to_string(status) + "; " + diagnostic);
}
void linker_fixture_program::add(const std::vector<unsigned char>& image, const char* name, nvJitLinkInputType kind) const {
    check(nvJitLinkAddData(value, kind, image.data(), image.size(), name), "add data");
}
void linker_fixture_program::complete() const { check(nvJitLinkComplete(value), "complete"); }
std::vector<unsigned char> linker_fixture_program::image() const {
    return compiler_fixture_bytes([&](size_t* size) { check(nvJitLinkGetLinkedCubinSize(value, size), "binary size"); },
        [&](unsigned char* bytes) { check(nvJitLinkGetLinkedCubin(value, bytes), "binary retrieval"); });
}
#endif
// Start before GPU initialization. DLLs with separate CRTs and native drivers
// must inherit the redirected output at process creation, not after startup.
conformance_child_result run_conformance_child(const char* executable, const char* backend, const char* case_name,
                                               bool capture_stderr) {
    FILE* file = nullptr;
#ifdef _WIN32
    tmpfile_s(&file);
#else
    file = std::tmpfile();
#endif
    require(file != nullptr, "create child output file");
    struct file_guard { FILE* value; ~file_guard() { std::fclose(value); } } cleanup{file};
    int status = 1;
#ifdef _WIN32
    std::array<wchar_t, 32768> path{};
    auto length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    require(length && length < path.size(), "locate conformance executable");
    auto widen = [](const char* value) { std::wstring result; for (; *value; ++value) result += wchar_t(*value); return result; };
    std::wstring command = L"\"" + std::wstring(path.data(), length) + L"\" " + widen(backend) + L" " + widen(case_name) + L" --worker";
    HANDLE output = INVALID_HANDLE_VALUE;
    require(DuplicateHandle(GetCurrentProcess(), reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file))),
        GetCurrentProcess(), &output, 0, TRUE, DUPLICATE_SAME_ACCESS), "create inheritable child output handle");
    STARTUPINFOW startup{}; startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE); startup.hStdOutput = output;
    startup.hStdError = capture_stderr ? output : GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    const bool created = CreateProcessW(path.data(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(output);
    require(created, "start conformance worker");
    CloseHandle(process.hThread);
    const auto waited = WaitForSingleObject(process.hProcess, 270000);
    if (waited != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 1); WaitForSingleObject(process.hProcess, 5000);
        CloseHandle(process.hProcess);
        throw std::runtime_error("conformance worker exceeded its deadline");
    }
    DWORD exit_status = 1;
    const bool queried = GetExitCodeProcess(process.hProcess, &exit_status);
    CloseHandle(process.hProcess);
    require(queried, "read conformance worker status");
    status = int(exit_status);
#else
    const auto child = fork();
    require(child >= 0, "start conformance worker");
    if (child == 0) {
        if (dup2(fileno(file), STDOUT_FILENO) < 0) _exit(126);
        if (capture_stderr && dup2(fileno(file), STDERR_FILENO) < 0) _exit(126);
        execlp(executable, executable, backend, case_name, "--worker", static_cast<char*>(nullptr));
        _exit(127);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(270);
    int raw_status = 0;
    for (;;) {
        auto observed = waitpid(child, &raw_status, WNOHANG);
        if (observed == child) break;
        if (observed < 0 && errno == EINTR) continue;
        if (observed < 0 || std::chrono::steady_clock::now() >= deadline) {
            kill(child, SIGKILL); waitpid(child, &raw_status, 0);
            throw std::runtime_error("conformance worker failed to complete before its deadline");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    status = WIFEXITED(raw_status) ? WEXITSTATUS(raw_status) : 1;
#endif
    require(std::fseek(file, 0, SEEK_END) == 0, "seek child output end");
    const auto bytes = std::ftell(file);
    require(bytes >= 0 && std::fseek(file, 0, SEEK_SET) == 0, "seek child output start");
    std::string output_text(static_cast<size_t>(bytes), '\0');
    require(std::fread(output_text.data(), 1, output_text.size(), file) == output_text.size(), "read complete child output");
    return {status, std::move(output_text)};
}

uintptr_t peer_integer(std::string_view text) {
    uintptr_t result{}; const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(), "valid inherited peer handle");
    return result;
}
struct conformance_peer::state {
    bool owner = false, waited = false;
    int status = -1;
#ifdef _WIN32
    HANDLE input = INVALID_HANDLE_VALUE, output = INVALID_HANDLE_VALUE, process = nullptr, job = nullptr;
    ~state() {
        if (owner && process && !waited) { TerminateProcess(process, 1); WaitForSingleObject(process, 5000); }
        if (input != INVALID_HANDLE_VALUE) CloseHandle(input);
        if (output != INVALID_HANDLE_VALUE && output != input) CloseHandle(output);
        if (process) CloseHandle(process);
        if (job) CloseHandle(job);
    }
#else
    int input = -1, output = -1;
    pid_t process = -1;
    ~state() {
        if (owner && process > 0 && !waited) {
            int raw{}; pid_t observed;
            do { observed = waitpid(process, &raw, WNOHANG); } while (observed < 0 && errno == EINTR);
            if (observed == 0) { kill(process, SIGKILL); while (waitpid(process, &raw, 0) < 0 && errno == EINTR) {} }
        }
        if (input >= 0) close(input);
        if (output >= 0 && output != input) close(output);
    }
#endif
};
conformance_peer::conformance_peer(std::unique_ptr<state> value) : storage(std::move(value)) {}
conformance_peer::~conformance_peer() = default;
conformance_peer::conformance_peer(conformance_peer&&) noexcept = default;
conformance_peer::conformance_peer(const char* executable, const char* backend, const char* case_name)
    : storage(std::make_unique<state>()) {
    for (auto token : {std::string_view(backend), std::string_view(case_name)})
        require(!token.empty() && token.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == token.npos,
            "peer backend/case is one plain identifier");
#ifdef _WIN32
    (void)executable;
    struct handle_guard { HANDLE value = INVALID_HANDLE_VALUE; ~handle_guard() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); } } child_read, child_write, partner, diagnostic, thread;
    SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
    require(CreatePipe(&storage->input, &child_write.value, &security, 4096), "create peer response pipe");
    require(CreatePipe(&child_read.value, &storage->output, &security, 4096), "create peer request pipe");
    require(SetHandleInformation(storage->input, HANDLE_FLAG_INHERIT, 0) && SetHandleInformation(storage->output, HANDLE_FLAG_INHERIT, 0), "keep parent pipe ends private");
    require(DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), GetCurrentProcess(), &partner.value,
        PROCESS_DUP_HANDLE | SYNCHRONIZE, TRUE, 0), "create restricted parent process capability");
    require(DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(), &diagnostic.value,
        0, TRUE, DUPLICATE_SAME_ACCESS), "inherit peer diagnostics");
    std::array<HANDLE, 4> inherited{child_read.value, child_write.value, partner.value, diagnostic.value};
    SIZE_T attribute_bytes{}; InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
    std::vector<std::byte> attributes(attribute_bytes);
    auto* attribute_list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    require(InitializeProcThreadAttributeList(attribute_list, 1, 0, &attribute_bytes), "initialize peer handle allowlist");
    struct attribute_guard { LPPROC_THREAD_ATTRIBUTE_LIST value; ~attribute_guard() { DeleteProcThreadAttributeList(value); } } cleanup{attribute_list};
    require(UpdateProcThreadAttribute(attribute_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(), sizeof(inherited), nullptr, nullptr), "restrict peer inherited handles");
    std::array<wchar_t, 32768> path{}; const auto length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    require(length && length < path.size(), "locate peer executable");
    auto widen = [](const char* value) { std::wstring result; for (; *value; ++value) result += wchar_t(*value); return result; };
    std::wstring command = L"\"" + std::wstring(path.data(), length) + L"\" " + widen(backend) + L" " + widen(case_name) + L" --peer " +
        std::to_wstring(reinterpret_cast<uintptr_t>(child_read.value)) + L" " + std::to_wstring(reinterpret_cast<uintptr_t>(child_write.value)) + L" " + std::to_wstring(reinterpret_cast<uintptr_t>(partner.value));
    storage->job = CreateJobObjectW(nullptr, nullptr); require(storage->job != nullptr, "create owned peer job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{}; limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    require(SetInformationJobObject(storage->job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)), "retire peer when its owner exits");
    STARTUPINFOEXW startup{}; startup.StartupInfo.cb = sizeof(startup); startup.lpAttributeList = attribute_list;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES; startup.StartupInfo.hStdInput = child_read.value;
    startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = diagnostic.value;
    PROCESS_INFORMATION process{};
    require(CreateProcessW(path.data(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo, &process), "start independent peer process");
    storage->process = process.hProcess; storage->owner = true; thread.value = process.hThread;
    require(AssignProcessToJobObject(storage->job, storage->process), "bind peer to owned job");
    require(ResumeThread(thread.value) != DWORD(-1), "run independent peer process");
#else
    int channel[2]{}; require(socketpair(AF_UNIX, SOCK_STREAM, 0, channel) == 0, "create peer socket pair");
    storage->input = storage->output = channel[0];
    struct descriptor_guard { int value; ~descriptor_guard() { close(value); } } child_channel{channel[1]};
    require(fcntl(channel[0], F_SETFD, FD_CLOEXEC) == 0 && fcntl(channel[1], F_SETFD, 0) == 0, "set peer descriptor inheritance");
    const auto descriptor = std::to_string(channel[1]); const auto parent = getpid();
    std::array<char*, 8> arguments{const_cast<char*>(executable), const_cast<char*>(backend), const_cast<char*>(case_name),
        const_cast<char*>("--peer"), const_cast<char*>(descriptor.c_str()), const_cast<char*>(descriptor.c_str()), const_cast<char*>("0"), nullptr};
    const auto child = fork(); require(child >= 0, "start independent peer process");
    if (child == 0) {
#ifdef __linux__
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(126);
#endif
        close(channel[0]); execv(executable, arguments.data()); _exit(127);
    }
    storage->process = child; storage->owner = true;
#endif
}
conformance_peer conformance_peer::connect(std::string_view read, std::string_view write, std::string_view partner) {
    auto value = std::make_unique<state>();
#ifdef _WIN32
    value->input = reinterpret_cast<HANDLE>(peer_integer(read)); value->output = reinterpret_cast<HANDLE>(peer_integer(write));
    value->process = reinterpret_cast<HANDLE>(peer_integer(partner));
#else
    value->input = int(peer_integer(read)); value->output = int(peer_integer(write)); (void)partner;
#endif
    return conformance_peer(std::move(value));
}
void conformance_peer::send(std::span<const std::byte> bytes) {
    require(bytes.size() <= 1024, "peer control message is bounded");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!bytes.empty()) {
#ifdef _WIN32
        DWORD written{}; require(WriteFile(storage->output, bytes.data(), DWORD(bytes.size()), &written, nullptr) && written, "write peer control bytes");
#else
        auto written = ::send(storage->output, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
        if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            require(std::chrono::steady_clock::now() < deadline, "peer write deadline");
            pollfd descriptor{storage->output, POLLOUT, 0}; poll(&descriptor, 1, 10); continue;
        }
        require(written > 0, "write peer control bytes");
#endif
        bytes = bytes.subspan(size_t(written));
        require(std::chrono::steady_clock::now() < deadline, "peer write completion deadline");
    }
}
void conformance_peer::receive(std::span<std::byte> bytes) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!bytes.empty()) {
        require(std::chrono::steady_clock::now() < deadline, "peer read deadline");
#ifdef _WIN32
        DWORD available{}; require(PeekNamedPipe(storage->input, nullptr, 0, nullptr, &available, nullptr), "inspect peer control pipe");
        if (!available) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
        DWORD received{}; require(ReadFile(storage->input, bytes.data(), DWORD(std::min<size_t>(bytes.size(), available)), &received, nullptr) && received, "read peer control bytes");
#else
        pollfd descriptor{storage->input, POLLIN, 0}; const auto ready = poll(&descriptor, 1, 10);
        if (ready < 0 && errno == EINTR) continue;
        require(ready >= 0, "poll peer control bytes"); if (!ready) continue;
        auto received = recv(storage->input, bytes.data(), bytes.size(), MSG_DONTWAIT);
        if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        require(received > 0, "read peer control bytes");
#endif
        bytes = bytes.subspan(size_t(received));
    }
}
void conformance_peer::send_handle(uintptr_t handle) {
#ifdef _WIN32
    HANDLE transferred{};
    require(DuplicateHandle(GetCurrentProcess(), reinterpret_cast<HANDLE>(handle), storage->process, &transferred, 0, FALSE, DUPLICATE_SAME_ACCESS), "transfer real OS handle to peer");
    const auto value = reinterpret_cast<uintptr_t>(transferred); send(std::as_bytes(std::span(&value, 1)));
#else
    std::byte marker{0x71}; iovec data{&marker, 1}; alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(sizeof(int))> control{};
    msghdr message{}; message.msg_iov = &data; message.msg_iovlen = 1; message.msg_control = control.data(); message.msg_controllen = control.size();
    auto* header = CMSG_FIRSTHDR(&message); header->cmsg_level = SOL_SOCKET; header->cmsg_type = SCM_RIGHTS; header->cmsg_len = CMSG_LEN(sizeof(int));
    const int descriptor = int(handle); std::memcpy(CMSG_DATA(header), &descriptor, sizeof(descriptor));
    require(sendmsg(storage->output, &message, MSG_NOSIGNAL) == 1, "transfer real OS descriptor to peer");
#endif
}
uintptr_t conformance_peer::receive_handle() {
#ifdef _WIN32
    uintptr_t value{}; receive(std::as_writable_bytes(std::span(&value, 1))); return value;
#else
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        require(std::chrono::steady_clock::now() < deadline, "peer descriptor read deadline");
        pollfd descriptor{storage->input, POLLIN, 0}; const auto ready = poll(&descriptor, 1, 10);
        if (ready < 0 && errno == EINTR) continue;
        require(ready >= 0, "poll peer descriptor"); if (!ready) continue;
        std::byte marker{}; iovec data{&marker, 1}; alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(sizeof(int))> control{};
        msghdr message{}; message.msg_iov = &data; message.msg_iovlen = 1; message.msg_control = control.data(); message.msg_controllen = control.size();
        const auto received = recvmsg(storage->input, &message, MSG_CMSG_CLOEXEC);
        if (received < 0 && errno == EINTR) continue;
        auto* header = CMSG_FIRSTHDR(&message);
        int result = -1;
        if (received > 0 && header && header->cmsg_level == SOL_SOCKET && header->cmsg_type == SCM_RIGHTS && header->cmsg_len >= CMSG_LEN(sizeof(int)))
            std::memcpy(&result, CMSG_DATA(header), sizeof(result));
        const bool valid = received == 1 && marker == std::byte{0x71} && !(message.msg_flags & MSG_CTRUNC) &&
            result >= 0 && header->cmsg_len == CMSG_LEN(sizeof(int));
        if (!valid && result >= 0) close(result);
        require(valid, "receive one complete real peer descriptor"); return uintptr_t(result);
    }
#endif
}
int conformance_peer::wait() {
    require(storage->owner, "only the owner waits for its peer process");
    if (storage->waited) return storage->status;
#ifdef _WIN32
    require(WaitForSingleObject(storage->process, 30000) == WAIT_OBJECT_0, "peer process completion deadline");
    DWORD status{}; require(GetExitCodeProcess(storage->process, &status), "read peer process exit status"); storage->status = int(status);
#else
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30); int status{};
    for (;;) {
        const auto observed = waitpid(storage->process, &status, WNOHANG);
        if (observed == storage->process) break;
        if (observed < 0 && errno == EINTR) continue;
        require(observed >= 0 && std::chrono::steady_clock::now() < deadline, "peer process completion deadline");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    storage->status = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
    storage->waited = true; return storage->status;
}

int run_host_conformance(int argc, char** argv, void (*execute)()) {
    try {
        require(argc == 1 || argc == 2, "usage: <executable> [case]");
        std::cout << "host-only case=" << (argc == 2 ? argv[1] : "all") << std::endl;
        execute();
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

#ifdef CUDAVM_CONFORMANCE_SOURCE_CHECK
void check_source_fixture() {
    struct source_program {
        nvrtcProgram value{};
        ~source_program() { if (value) nvrtcDestroyProgram(&value); }
    } program;
    auto check = [&](nvrtcResult status) {
        if (status == NVRTC_SUCCESS) return;
        size_t size = 0;
        if (program.value) nvrtcGetProgramLogSize(program.value, &size);
        std::string diagnostic(size, '\0');
        if (size) nvrtcGetProgramLog(program.value, diagnostic.data());
        throw std::runtime_error(std::string(nvrtcGetErrorString(status)) + ": " + diagnostic);
    };
    check(nvrtcCreateProgram(&program.value, fixtures.source.data(), "checked_fixture.cu", 0, nullptr, nullptr));
    const auto architecture = "-arch=compute_" + conformance_architecture(fixtures);
    const auto standard = "--std=c++" + std::to_string(fixtures.source_standard);
    const char* options[]{architecture.c_str(), standard.c_str()};
    check(nvrtcCompileProgram(program.value, 2, options));
    size_t size = 0; check(nvrtcGetPTXSize(program.value, &size));
    require(size > 1, "compiled source fixture has a nonempty PTX result");
    std::vector<char> output(size + 32, char(0x5b));
    check(nvrtcGetPTX(program.value, output.data() + 16));
    check(nvrtcDestroyProgram(&program.value));
    for (size_t index = 0; index < output.size(); ++index)
        if (index < 16 || index >= size + 16) require(output[index] == char(0x5b), "source fixture PTX extent guards");
    require(output[size + 15] == 0, "source fixture PTX has its declared terminator");
    const std::string_view ptx(output.data() + 16, size - 1);
    require(ptx.find(".entry ") != ptx.npos && ptx.find(".target sm_" + conformance_architecture(fixtures)) != ptx.npos,
        "owned source fixture output retains its kernel and declared architecture");
    if (fixtures.minimum_ptx) {
        const auto version = ptx.find(".version ");
        require(version != ptx.npos, "source fixture declares a PTX dialect");
        const auto* first = ptx.data() + version + 9;
        const auto* end = ptx.data() + ptx.size();
        unsigned major = 0, minor = 0;
        const auto whole = std::from_chars(first, end, major);
        require(whole.ec == std::errc{} && whole.ptr != end && *whole.ptr == '.', "source fixture PTX dialect syntax");
        const auto fraction = std::from_chars(whole.ptr + 1, end, minor);
        require(fraction.ec == std::errc{} && minor < 10 && major * 10 + minor >= fixtures.minimum_ptx,
            "source fixture satisfies its required PTX dialect floor");
    }
}
#endif

#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
struct conformance_unavailable : std::runtime_error { using std::runtime_error::runtime_error; };
void require_capability(bool available, const std::string& reason) {
    if (!available) throw conformance_unavailable(reason);
}
struct conformance_driver_error : std::runtime_error {
    CUresult result;
    conformance_driver_error(CUresult status, const std::string& message)
        : std::runtime_error(message), result(status) {}
};
std::string native_cuda_reference_library() {
#if defined(_WIN32)
    std::string directory(32768, '\0');
    auto size = GetSystemDirectoryA(directory.data(), unsigned(directory.size()));
    require(size && size < directory.size(), "native CUDA reference system directory is available");
    directory.resize(size);
    return directory + "/nvcuda.dll";
#else
    const auto* selected = std::getenv("CUDAVM_CUDA_DRIVER");
    return selected ? selected : "libcuda.so.1";
#endif
}
int native_cuda_version() {
    const auto path = native_cuda_reference_library();
#if defined(_WIN32)
    auto library = LoadLibraryExA(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    require_capability(library, "native CUDA driver unavailable");
    auto query = reinterpret_cast<decltype(&cuDriverGetVersion)>(GetProcAddress(library, "cuDriverGetVersion"));
    int version{}; auto status = query ? query(&version) : CUDA_ERROR_NOT_SUPPORTED; FreeLibrary(library);
#else
    auto library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    require_capability(library, "native CUDA driver unavailable");
    if (dlsym(library, "cudavmGetDiagnostic")) {
        dlclose(library);
        require_capability(false, "native CUDA reference resolved to the facade; set CUDAVM_CUDA_DRIVER to its vendor path");
    }
    auto query = reinterpret_cast<decltype(&cuDriverGetVersion)>(dlsym(library, "cuDriverGetVersion"));
    int version{}; auto status = query ? query(&version) : CUDA_ERROR_NOT_SUPPORTED; dlclose(library);
#endif
    CONFORMANCE_CUDA(status); return version;
}
void check_driver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    const char* diagnostic = cudavmGetDiagnostic();
    throw conformance_driver_error(result, std::string(operation) + ": " + (name ? name : "unknown Driver error")
        + (diagnostic && *diagnostic ? std::string("; ") + diagnostic : ""));
}
void expect_words(std::span<const uint32_t> actual, std::span<const uint32_t> expected,
                  const std::string& context) {
    require(actual.size() == expected.size(), context + ": output extent");
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i])
            throw std::runtime_error(context + ": word " + std::to_string(i)
                + " got " + std::to_string(actual[i]) + ", expected " + std::to_string(expected[i]));
}

struct device_context {
    CUcontext value{};
    explicit device_context(CUdevice device, unsigned flags = 0) { CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, flags, device)); }
    ~device_context() { if (value) cuCtxDestroy(value); }
};

struct conformance_device {
    CUdevice value{-1};
    std::string name;
};

conformance_device select_conformance_device(std::string_view backend, void (*probe)() = nullptr, unsigned architecture = 0, bool specific = false) {
    const std::array<std::string_view, 5> backends{"cuda", "vulkan", "directx", "opencl", "opengl"};
    const std::array<std::string_view, 5> labels{"CUDA", "Vulkan", "DirectX", "OpenCL", "OpenGL"};
    const auto found = std::find(backends.begin(), backends.end(), backend);
    require(backend == "any" || found != backends.end(), "unknown backend: " + std::string(backend));
    const auto status = cuInit(0);
    if (status == CUDA_ERROR_NO_DEVICE || status == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) {
        std::cout << "SKIP: " << backend << " device/driver unavailable\n";
        return {};
    }
    CONFORMANCE_CUDA(status);
    int count = 0;
    CONFORMANCE_CUDA(cuDeviceGetCount(&count));
    const auto suffix = backend == "any" ? std::string{} : " (" + std::string(labels[found - backends.begin()]) + ")";
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUdevice candidate{};
        char name[256]{};
        CONFORMANCE_CUDA(cuDeviceGet(&candidate, ordinal));
        CONFORMANCE_CUDA(cuDeviceGetName(name, sizeof(name), candidate));
        if (!std::string_view(name).ends_with(suffix)) continue;
        if (backend == "cuda" && architecture) {
            int major = 0, minor = 0;
            CONFORMANCE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, candidate));
            CONFORMANCE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, candidate));
            if (unsigned(major * 10 + minor) < architecture || (specific && unsigned(major * 10 + minor) != architecture)) {
                std::cout << "unavailable device=" << name << "; requires sm_" << architecture << (specific ? "a" : "")
                    << ", reports sm_" << major << minor << std::endl;
                continue;
            }
        }
        if (probe) {
            try {
                device_context context(candidate);
                probe();
            } catch (const conformance_driver_error& error) {
                if (error.result != CUDA_ERROR_NOT_SUPPORTED
                    && error.result != CUDA_ERROR_NO_BINARY_FOR_GPU
                    && error.result != CUDA_ERROR_UNSUPPORTED_PTX_VERSION) throw;
                std::cout << "unavailable device=" << name << "; " << error.what() << std::endl;
                continue;
            }
        }
        return {candidate, name};
    }
    std::cout << "SKIP: no " << (probe ? "executable " : "") << backend << " device\n";
    return {};
}

int run_context_conformance(int argc, char** argv, void (*execute)(), unsigned context_flags,
                            void (*probe)(), bool isolated_fault) {
    try {
        require(argc == 2, "usage: <executable> <backend>");
        const auto device = select_conformance_device(argv[1], probe);
        if (device.value == -1) return 77;
        std::cout << "backend=" << argv[1] << " device=" << device.name << std::endl;
        device_context context(device.value, context_flags);
        // The process, rather than CUDA teardown, retires this context. Fatal
        // native faults can make even destruction unsafe or non-progressing.
        if (isolated_fault) context.value = nullptr;
        execute();
        if (!isolated_fault) CONFORMANCE_CUDA(cuCtxSynchronize());
        std::cout << "PASS\n";
        return 0;
    } catch (const conformance_unavailable& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

int run_conformance(int argc, char** argv, void (*execute)(), unsigned context_flags, void (*probe)()) {
    return run_context_conformance(argc, argv, execute, context_flags, probe, false);
}

[[noreturn]] void run_isolated_fault_conformance(int argc, char** argv, void (*execute)(), void (*probe)()) {
    const int status = run_context_conformance(argc, argv, execute, 0, probe, true);
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(status);
}

#ifdef CUDAVM_CONFORMANCE_EXECUTION
struct source_program {
    nvrtcProgram value{};
    ~source_program() { if (value) nvrtcDestroyProgram(&value); }
    void check(nvrtcResult result) const {
        if (result == NVRTC_SUCCESS) return;
        size_t size = 0;
        nvrtcGetProgramLogSize(value, &size);
        std::string log(size, '\0');
        if (size) nvrtcGetProgramLog(value, log.data());
        throw std::runtime_error(std::string(nvrtcGetErrorString(result)) + ": " + log);
    }
};
#ifndef CUDAVM_CONFORMANCE_SOURCE_ONLY
struct nvvm_program {
    nvvmProgram value{};
    ~nvvm_program() { if (value) nvvmDestroyProgram(&value); }
    void check(nvvmResult result) const {
        if (result == NVVM_SUCCESS) return;
        size_t size = 0;
        nvvmGetProgramLogSize(value, &size);
        std::string log(size, '\0');
        if (size) nvvmGetProgramLog(value, log.data());
        throw std::runtime_error(std::string(nvvmGetErrorString(result)) + ": " + log);
    }
};
struct link_program {
    nvJitLinkHandle value{};
    ~link_program() { if (value) nvJitLinkDestroy(&value); }
    void check(nvJitLinkResult result) const {
        if (result == NVJITLINK_SUCCESS) return;
        size_t size = 0;
        nvJitLinkGetErrorLogSize(value, &size);
        std::string log(size + 1, '\0');
        nvJitLinkGetErrorLog(value, log.data());
        throw std::runtime_error("nvJitLink error " + std::to_string(result) + ": " + log);
    }
};

#endif
std::vector<char> import_fixture(const fixture_data& fixtures, std::string_view importer) {
#ifdef CUDAVM_CONFORMANCE_SOURCE_ONLY
    require(importer == "cu", "this execution profile declares only CUDA source input");
#endif
    if (importer == "ptx") {
        std::vector<char> bytes(fixtures.ptx.begin(), fixtures.ptx.end());
        bytes.push_back('\0');
        return bytes;
    }
    if (importer == "cubin") return {fixtures.cubin.begin(), fixtures.cubin.end()};
    if (importer == "cu") {
        source_program program;
        program.check(nvrtcCreateProgram(&program.value, fixtures.source.data(), "fixture.cu", 0, nullptr, nullptr));
        const auto architecture = "-arch=sm_" + conformance_architecture(fixtures);
        const auto standard = "--std=c++" + std::to_string(fixtures.source_standard);
#ifdef CUDAVM_CONFORMANCE_SOURCE_LTO
        const char* options[]{architecture.c_str(), standard.c_str(), "-dlto"};
        program.check(nvrtcCompileProgram(program.value, 3, options));
        size_t size = 0;
        program.check(nvrtcGetLTOIRSize(program.value, &size));
#else
        const char* options[]{architecture.c_str(), standard.c_str()};
        program.check(nvrtcCompileProgram(program.value, 2, options));
        size_t size = 0;
        program.check(nvrtcGetCUBINSize(program.value, &size));
#endif
        require(size > 0, "NVRTC returned an empty program");
        std::vector<char> bytes(size);
#ifdef CUDAVM_CONFORMANCE_SOURCE_LTO
        program.check(nvrtcGetLTOIR(program.value, bytes.data()));
#else
        program.check(nvrtcGetCUBIN(program.value, bytes.data()));
#endif
        return bytes;
    }
#ifndef CUDAVM_CONFORMANCE_SOURCE_ONLY
    require(importer == "nvvm", "unknown importer: " + std::string(importer));
    nvvm_program program;
    program.check(nvvmCreateProgram(&program.value));
    program.check(nvvmAddModuleToProgram(program.value, fixtures.nvvm.data(), fixtures.nvvm.size(), "fixture.ll"));
    const auto architecture = "-arch=compute_" + conformance_architecture(fixtures);
    const char* options[]{architecture.c_str(), "-gen-lto"};
    program.check(nvvmCompileProgram(program.value, 2, options));
    size_t size = 0;
    program.check(nvvmGetCompiledResultSize(program.value, &size));
    require(size > 0, "libNVVM returned an empty program");
    std::vector<char> ir(size);
    program.check(nvvmGetCompiledResult(program.value, ir.data()));
    link_program link;
    const auto link_architecture = "-arch=sm_" + conformance_architecture(fixtures);
    const char* link_options[]{link_architecture.c_str(), "-lto"};
    link.check(nvJitLinkCreate(&link.value, 2, link_options));
    link.check(nvJitLinkAddData(link.value, NVJITLINK_INPUT_LTOIR, ir.data(), ir.size(), "fixture"));
    link.check(nvJitLinkComplete(link.value));
    link.check(nvJitLinkGetLinkedCubinSize(link.value, &size));
    require(size > 0, "nvJitLink returned an empty program");
    std::vector<char> bytes(size);
    link.check(nvJitLinkGetLinkedCubin(link.value, bytes.data()));
    return bytes;
#else
    throw std::runtime_error("unknown source execution importer");
#endif
}

#endif // CUDAVM_CONFORMANCE_EXECUTION
// Public module/library loading also serves compiler-independent API fixtures.
loaded_program::loaded_program(const void* image, bool use_library) {
    if (use_library) CONFORMANCE_CUDA(cuLibraryLoadData(&library, image, nullptr, nullptr, 0, nullptr, nullptr, 0));
    else CONFORMANCE_CUDA(cuModuleLoadData(&module, image));
}
loaded_program::~loaded_program() {
    if (library) cuLibraryUnload(library);
    if (module) cuModuleUnload(module);
}
CUfunction loaded_program::function(const char* name) const {
    CUfunction result{};
    if (library) {
        CUkernel kernel{};
        CONFORMANCE_CUDA(cuLibraryGetKernel(&kernel, library, name));
        CONFORMANCE_CUDA(cuKernelGetFunction(&result, kernel));
    } else CONFORMANCE_CUDA(cuModuleGetFunction(&result, module, name));
    return result;
}
CUkernel loaded_program::library_kernel(const char* name) const {
    CUkernel result{};
    if (library) CONFORMANCE_CUDA(cuLibraryGetKernel(&result, library, name));
    return result;
}
bool loaded_program::contains_function(const char* name) const {
    if (library) {
        CUkernel result{};
        const auto status = cuLibraryGetKernel(&result, library, name);
        if (status == CUDA_ERROR_NOT_FOUND) return false;
        CONFORMANCE_CUDA(status);
        require(result != nullptr, "kernel presence lookup publishes a handle");
    } else {
        CUfunction result{};
        const auto status = cuModuleGetFunction(&result, module, name);
        if (status == CUDA_ERROR_NOT_FOUND) return false;
        CONFORMANCE_CUDA(status);
        require(result != nullptr, "function presence lookup publishes a handle");
    }
    return true;
}
CUdeviceptr loaded_program::global(const char* name, size_t expected_size) const {
    CUdeviceptr result = 0;
    size_t size = 0;
    if (library) CONFORMANCE_CUDA(cuLibraryGetGlobal(&result, &size, library, name));
    else CONFORMANCE_CUDA(cuModuleGetGlobal(&result, &size, module, name));
    require(result && size == expected_size, std::string("global symbol extent: ") + name);
    return result;
}

// Availability is checked independently of the slice's input and features.
// Enumerating an OpenCL endpoint does not imply SPIR-V ingestion.
void probe_execution() {
    static constexpr char probe_ptx[] = R"(
.version 6.5
.target sm_75
.address_size 64
.visible .entry conformance_probe(.param .u64 out) {
 .reg .b64 %out;
 .reg .b32 %value;
 ld.param.u64 %out,[out];
 mov.u32 %value,0x43564d31;
 st.global.u32 [%out],%value;
 ret;
}
)";
    loaded_program probe(probe_ptx, false);
    const auto kernel = probe.function("conformance_probe");
    device_buffer output(sizeof(uint32_t));
    CONFORMANCE_CUDA(cuMemsetD32(output.value, 0, 1));
    void* args[]{&output.value};
    CONFORMANCE_CUDA(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    uint32_t actual = 0;
    CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, output.value, sizeof(actual)));
    require(actual == 0x43564d31, "backend availability probe produced an incorrect result");
}

#ifdef CUDAVM_CONFORMANCE_EXECUTION
int run_conformance(int argc, char** argv, const fixture_data& fixtures,
                    void (*execute)(const loaded_program&), unsigned load_repetitions, unsigned context_flags) {
    try {
        require(argc == 3, "usage: <executable> <backend> <cu|ptx|cubin|nvvm>");
        require(load_repetitions > 0, "at least one loader execution is required");
        const std::string backend = argv[1];
        const auto device = select_conformance_device(backend, probe_execution, fixtures.architecture, !fixtures.architecture_suffix.empty());
        if (device.value == -1) return 77;
        std::cout << "backend=" << backend << " importer=" << argv[2] << " device=" << device.name << std::endl;
        device_context context(device.value, context_flags);
        // Compiler handles are destroyed before either loader sees the copy.
        const auto image = import_fixture(fixtures, argv[2]);
        for (unsigned repeat = 0; repeat < load_repetitions; ++repeat)
        for (bool library : {false, true}) {
            std::cout << "loader=" << (library ? "library" : "module") << std::endl;
            loaded_program program(image.data(), library);
            execute(program);
            CONFORMANCE_CUDA(cuCtxSynchronize());
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const conformance_unavailable& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
#endif
#endif
}

#ifdef CUDAVM_CONFORMANCE_SOURCE_CHECK
int main(int argc, char** argv) {
    return cudavm::run_host_conformance(argc, argv, cudavm::check_source_fixture);
}
#endif
