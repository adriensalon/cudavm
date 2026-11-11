#include "program_image_fixture.hpp"

namespace cudavm {
std::string_view program_image_validation_case;
// Fixture editing follows docs/native-abi.md. No private codec is linked or
// included: statuses, isolated retries and execution supply the public oracle.
constexpr std::string_view program_image_validation_magic = "CUDAVM-PROGRAM-IMAGE\n";
uint64_t program_image_validation_hash(const program_image_bytes& bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (auto byte : bytes) { hash ^= byte; hash *= 1099511628211ull; }
    return hash;
}
uint64_t program_image_validation_number(const program_image_bytes& bytes, size_t offset, unsigned width = 4) {
    require(width <= 8 && offset <= bytes.size() && width <= bytes.size() - offset, "bounded trusted fixture field");
    uint64_t value{};
    for (unsigned i = 0; i < width; ++i) value |= uint64_t(bytes[offset + i]) << (i * 8);
    return value;
}
void program_image_validation_number(program_image_bytes& bytes, size_t offset, uint64_t value, unsigned width) {
    require(width <= 8 && offset <= bytes.size() && width <= bytes.size() - offset, "bounded owned fixture mutation");
    for (unsigned i = 0; i < width; ++i) bytes[offset + i] = static_cast<unsigned char>(value >> (i * 8));
}
void program_image_validation_append(program_image_bytes& bytes, uint64_t value, unsigned width = 4) {
    for (unsigned i = 0; i < width; ++i) bytes.push_back(static_cast<unsigned char>(value >> (i * 8)));
}
void program_image_validation_string(program_image_bytes& bytes, std::string_view text) {
    program_image_validation_append(bytes, text.size()); bytes.insert(bytes.end(), text.begin(), text.end());
}
program_image_bytes program_image_validation_payload(const program_image_bytes& image) {
    require(image.size() > program_image_validation_magic.size() + 18
        && std::equal(program_image_validation_magic.begin(), program_image_validation_magic.end(), image.begin())
        && image[image.size() - 2] == '\n' && image.back() == 0, "trusted producer uses documented image framing");
    const auto end = image.size() - 2;
    require((end - program_image_validation_magic.size()) % 2 == 0, "trusted fixture hexadecimal extent");
    auto digit = [](unsigned char byte) -> unsigned {
        require((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'), "trusted fixture hexadecimal digit");
        return byte <= '9' ? byte - '0' : byte - 'a' + 10;
    };
    program_image_bytes bytes;
    for (size_t i = program_image_validation_magic.size(); i < end; i += 2)
        bytes.push_back(static_cast<unsigned char>(digit(image[i]) * 16 + digit(image[i + 1])));
    require(bytes.size() >= 12, "trusted payload contains version and checksum");
    const auto checksum = program_image_validation_number(bytes, bytes.size() - 8, 8); bytes.resize(bytes.size() - 8);
    require(program_image_validation_hash(bytes) == checksum, "trusted public result checksum");
    return bytes;
}
program_image_bytes program_image_validation_encode(program_image_bytes bytes) {
    program_image_validation_append(bytes, program_image_validation_hash(bytes), 8);
    program_image_bytes image(program_image_validation_magic.begin(), program_image_validation_magic.end());
    constexpr char digits[] = "0123456789abcdef";
    for (auto byte : bytes) { image.push_back(digits[byte >> 4]); image.push_back(digits[byte & 15]); }
    image.push_back('\n'); image.push_back(0); return image;
}
struct program_image_validation_layout {
    std::array<size_t, 5> fields{};
    size_t names{}, artifacts{};
    std::vector<std::pair<size_t, size_t>> records;
    explicit program_image_validation_layout(const program_image_bytes& bytes) {
        size_t cursor = 4;
        auto string = [&] {
            const auto size = program_image_validation_number(bytes, cursor); cursor += 4;
            require(cursor <= bytes.size() && size <= bytes.size() - cursor, "trusted length-prefixed fixture string");
            cursor += size_t(size);
        };
        for (auto& field : fields) { field = cursor; string(); }
        names = cursor; const auto name_count = program_image_validation_number(bytes, cursor); cursor += 4;
        require(name_count <= 4096, "trusted fixture name inventory bound");
        for (size_t i = 0; i < name_count; ++i) { string(); string(); }
        artifacts = cursor; const auto artifact_count = program_image_validation_number(bytes, cursor); cursor += 4;
        require(artifact_count <= 5, "trusted fixture native inventory bound");
        for (size_t i = 0; i < artifact_count; ++i) {
            const auto start = cursor; cursor += 20; string(); records.emplace_back(start, cursor);
        }
        require(cursor == bytes.size(), "trusted payload has exact documented fields");
    }
};
void program_image_validation_replace(program_image_bytes& bytes, size_t offset, std::string_view text) {
    const auto size = program_image_validation_number(bytes, offset);
    require(offset + 4 <= bytes.size() && size <= bytes.size() - offset - 4, "trusted replaceable string extent");
    program_image_bytes field; program_image_validation_string(field, text);
    bytes.erase(bytes.begin() + offset, bytes.begin() + offset + 4 + size_t(size));
    bytes.insert(bytes.begin() + offset, field.begin(), field.end());
}
struct program_image_invalid_fixture { std::string reason; program_image_bytes image; };
std::vector<program_image_invalid_fixture> program_image_validation_invalid(std::string_view family, const program_image_bytes& image) {
    const auto payload = program_image_validation_payload(image); const program_image_validation_layout layout(payload);
    std::vector<program_image_invalid_fixture> fixtures;
    auto raw = [&](std::string reason, program_image_bytes bytes) { fixtures.push_back({std::move(reason), std::move(bytes)}); };
    auto encoded = [&](std::string reason, program_image_bytes bytes) { raw(std::move(reason), program_image_validation_encode(std::move(bytes))); };
    auto edit = [&](std::string reason, size_t offset, uint64_t value, unsigned width = 4) {
        auto bytes = payload; program_image_validation_number(bytes, offset, value, width); encoded(std::move(reason), std::move(bytes));
    };
    if (family == "framing") {
        auto bytes = image; bytes.erase(bytes.end() - 2); raw("truncated program image", std::move(bytes));
        bytes = image; bytes[program_image_validation_magic.size()] = 'g'; raw("invalid program image encoding", std::move(bytes));
        bytes = image; bytes.erase(bytes.begin() + program_image_validation_magic.size()); raw("invalid program image extent", std::move(bytes));
        bytes = image; bytes[program_image_validation_magic.size() + 12] ^= 1; raw("program image checksum mismatch", std::move(bytes));
        bytes = image; bytes.insert(bytes.end() - 1, 'x'); raw("truncated program image", std::move(bytes));
        bytes = image; bytes[program_image_validation_magic.size() + 12] = 0; raw("invalid program image encoding", std::move(bytes));
    } else if (family == "versions") {
        edit("unsupported program image version", 0, 0); edit("unsupported program image version", 0, 2);
        auto bytes = payload; program_image_validation_replace(bytes, layout.fields[0], "unrecognized-producer-profile");
        encoded("unsupported program image IR profile", std::move(bytes));
        bytes = payload; program_image_validation_replace(bytes, layout.fields[1], "sm_80");
        encoded("program image architecture mismatch", std::move(bytes));
    } else if (family == "extents") {
        edit("truncated program image string", layout.fields[0], 0xffffffffu);
        edit("too many program image name expressions", layout.names, 4097);
        edit("too many program image artifacts", layout.artifacts, 6);
        auto bytes = payload; bytes.push_back(0x57); encoded("extra program image data", std::move(bytes));
        bytes = payload; bytes.resize(4); encoded("truncated program image", std::move(bytes));
        bytes.assign(payload.begin(), payload.begin() + layout.names); program_image_validation_append(bytes, 2);
        for (unsigned i = 0; i < 2; ++i) { program_image_validation_string(bytes, "&owned_name"); program_image_validation_string(bytes, "module_alpha"); }
        program_image_validation_append(bytes, 0); encoded("invalid program image name expression", std::move(bytes));
    } else if (family == "metadata") {
        auto bytes = payload; program_image_validation_replace(bytes, layout.fields[4], "{}");
        encoded("program image ABI/linkage metadata mismatch", std::move(bytes));
        bytes = payload; program_image_validation_replace(bytes, layout.fields[0], ""); encoded("incomplete program image", std::move(bytes));
        bytes = payload; program_image_validation_replace(bytes, layout.fields[1], ""); encoded("incomplete program image", std::move(bytes));
        bytes = payload; program_image_validation_replace(bytes, layout.fields[3], ""); encoded("incomplete program image", std::move(bytes));
        bytes = payload; program_image_validation_replace(bytes, layout.fields[4], ""); encoded("incomplete program image", std::move(bytes));
    } else {
        require(family == "artifacts" && !layout.records.empty(), "native fixture has an editable cache record");
        const auto first = layout.records.front().first;
        edit("invalid program image artifact binding", first, 5);
        edit("invalid program image artifact binding", first + 12, program_image_validation_number(payload, first + 12, 8) ^ 1, 8);
        auto bytes = payload; program_image_validation_replace(bytes, first + 20, "");
        encoded("invalid program image artifact binding", std::move(bytes));
        bytes.assign(payload.begin(), payload.begin() + layout.artifacts); program_image_validation_append(bytes, 2);
        for (unsigned i = 0; i < 2; ++i) bytes.insert(bytes.end(), payload.begin() + first, payload.begin() + layout.records.front().second);
        encoded("invalid program image artifact binding", std::move(bytes));
    }
    return fixtures;
}
program_image_bytes program_image_validation_fallback(const program_image_bytes& image, unsigned mode) {
    auto bytes = program_image_validation_payload(image); const program_image_validation_layout layout(bytes);
    require(!layout.records.empty() && mode < 3, "known native cache fallback fixture");
    if (mode == 2) {
        bytes.resize(layout.artifacts); program_image_validation_append(bytes, 0);
    } else for (auto record = layout.records.rbegin(); record != layout.records.rend(); ++record) {
        const auto position = record->first + (mode == 0 ? 4 : 8);
        program_image_validation_number(bytes, position, program_image_validation_number(bytes, position) + 1, 4);
        // If a consumer ignores the incompatible profile, this cache cannot run.
        program_image_validation_replace(bytes, record->first + 20, "incompatible native cache must be rebuilt from IR");
    }
    return program_image_validation_encode(std::move(bytes));
}
std::string program_image_validation_ptx(const program_image_bytes& image) {
    linker_fixture_program linker({"-arch=sm_75", "-lto", "-ptx", "-O0"});
    linker.add(image, "repaired_public_image", NVJITLINK_INPUT_CUBIN); linker.complete();
    auto bytes = program_image_result([&](size_t* size) { linker.check(nvJitLinkGetLinkedPtxSize(linker.value, size), "validation PTX size"); },
        [&](unsigned char* output) { linker.check(nvJitLinkGetLinkedPtx(linker.value, reinterpret_cast<char*>(output)), "validation PTX output"); });
    require(bytes.back() == 0, "validation PTX terminal NUL"); return {bytes.begin(), bytes.end() - 1};
}
program_image_bytes program_image_validation_sentinel() {
    compiler_fixture_program program("extern \"C\" __global__ void image_validation_sentinel() {}", "image_sentinel.cu");
    require(program.compile({"-arch=compute_75", "-dlto", "--std=c++17"}) == NVRTC_SUCCESS, "independent source sentinel compiles");
    return program_image_result([&](size_t* size) { compiler_fixture_program::check(nvrtcGetLTOIRSize(program.value, size)); },
        [&](unsigned char* bytes) { compiler_fixture_program::check(nvrtcGetLTOIR(program.value, reinterpret_cast<char*>(bytes))); });
}
void program_image_validation_host() {
    const auto image = program_image_nvrtc_produce(program_image_validation_case == "artifacts" || program_image_validation_case == "profiles" ? "nvrtc_cubin" : "nvrtc_lto");
    if (program_image_validation_case == "profiles") {
        for (unsigned mode = 0; mode < 3; ++mode) {
            const auto text = program_image_validation_ptx(program_image_validation_fallback(image, mode));
            require(text.find(".entry module_alpha(") != text.npos && text.find(".entry module_beta(") != text.npos, "incompatible/missing caches preserve public IR reimport");
        }
        return;
    }
    auto without_nul = image; without_nul.pop_back();
    // Explicit extents permit the optional convenience NUL to be omitted.
    program_image_link(without_nul, NVJITLINK_INPUT_ANY, true, true);
    const auto sentinel = program_image_validation_sentinel();
    const auto fixtures = program_image_validation_invalid(program_image_validation_case, image);
    for (size_t index = 0; index < fixtures.size(); ++index) {
        const auto& fixture = fixtures[index]; std::cout << "image rejection=" << fixture.reason << std::endl;
        program_image_bytes repaired;
        {
            linker_fixture_program linker({"-arch=sm_75", "-lto", "-r", "-O0"});
            linker.add(sentinel, "retained_sentinel", NVJITLINK_INPUT_LTOIR);
            constexpr std::array kinds{NVJITLINK_INPUT_CUBIN, NVJITLINK_INPUT_LTOIR, NVJITLINK_INPUT_ANY};
            const auto kind = kinds[index % kinds.size()]; std::string label = "owned_invalid_program.image";
            std::unique_ptr<program_image_file> file;
            if (index % 2) { file = std::make_unique<program_image_file>(fixture.image); label = file->path.string(); }
            const auto status = file ? nvJitLinkAddFile(linker.value, kind, label.c_str())
                : nvJitLinkAddData(linker.value, kind, fixture.image.data(), fixture.image.size(), label.c_str());
            require(status == NVJITLINK_ERROR_INVALID_INPUT, "exact malformed-image status, got " + std::to_string(status));
            const auto log = program_image_result([&](size_t* size) { linker.check(nvJitLinkGetErrorLogSize(linker.value, size), "invalid-image log size"); },
                [&](unsigned char* output) { linker.check(nvJitLinkGetErrorLog(linker.value, reinterpret_cast<char*>(output)), "invalid-image log output"); });
            require(log.back() == 0, "invalid-image diagnostic terminates"); const std::string diagnostic(log.begin(), log.end() - 1);
            require(diagnostic.find(label) != diagnostic.npos && diagnostic.find(fixture.reason) != diagnostic.npos, "public diagnostic identifies the owned image and violated rule: " + diagnostic);
            program_image_output<size_t> size;
            require(nvJitLinkGetLinkedCubinSize(linker.value, &size.value) == NVJITLINK_ERROR_INTERNAL, "failed addition publishes no premature binary"); size.get();
            linker.add(image, "repaired_image", NVJITLINK_INPUT_CUBIN); linker.complete(); repaired = program_image_linked_result(linker);
        }
        const auto text = program_image_validation_ptx(repaired);
        for (auto name : {"module_alpha", "module_beta", "image_validation_sentinel"})
            require(text.find(std::string(".entry ") + name + "(") != text.npos, "repaired link preserves accepted input and valid replacement after rejection");
    }
}
void program_image_validation_reject_load(const program_image_bytes& image, unsigned route) {
    program_image_output<CUmodule> module; program_image_output<CUlibrary> library;
    program_image_output<cudaLibrary_t> runtime_library;
    std::unique_ptr<program_image_file> file;
    if (route == 2 || route == 4 || route == 6) file = std::make_unique<program_image_file>(image);
    const auto path = file ? file->path.string() : std::string(); CUresult status = CUDA_SUCCESS;
    if (route == 0) status = cuModuleLoadData(&module.value, image.data());
    else if (route == 1) status = cuModuleLoadDataEx(&module.value, image.data(), 0, nullptr, nullptr);
    else if (route == 2) status = cuModuleLoad(&module.value, path.c_str());
    else if (route == 3) status = cuLibraryLoadData(&library.value, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    else if (route == 4) status = cuLibraryLoadFromFile(&library.value, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    else {
        const auto runtime = route == 5 ? cudaLibraryLoadData(&runtime_library.value, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0)
            : cudaLibraryLoadFromFile(&runtime_library.value, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
        require(runtime == cudaErrorInvalidKernelImage, "Runtime rejects malformed image with its pinned image status, got " + std::to_string(runtime));
    }
    if (route < 5) require(status == CUDA_ERROR_INVALID_IMAGE, "Driver rejects malformed program image, got " + std::to_string(status));
    require(!module.get() && !library.get() && !runtime_library.get(), "failed program loads preserve output guards and publish no handle");
}
void execute_program_image_validation() {
    const auto image = program_image_nvrtc_produce("nvrtc_cubin");
    if (program_image_validation_case == "fallback") {
        for (unsigned mode = 0; mode < 3; ++mode) for (unsigned route : {0u, 3u, 5u}) {
            program_image_loaded program(program_image_validation_fallback(image, mode), route); program_image_execute(program);
        }
    } else {
        require(program_image_validation_case == "rejection", "known image validation execution profile");
        program_image_loaded retained(image, 0);
        for (auto family : {"framing", "artifacts"}) {
            const auto fixtures = program_image_validation_invalid(family, image);
            for (size_t index : {size_t(0), size_t(3)}) {
                auto malformed = fixtures[index].image;
                for (unsigned route = 0; route < 7; ++route) program_image_validation_reject_load(malformed, route);
            }
        }
        // Exercise IR metadata validation without any cached native shortcut.
        const auto portable = program_image_nvrtc_produce("nvrtc_lto");
        for (auto family : {"versions", "metadata"}) {
            const auto fixtures = program_image_validation_invalid(family, portable);
            const auto& malformed = fixtures[family == std::string_view("versions") ? 2 : 0].image;
            for (unsigned route = 0; route < 7; ++route) program_image_validation_reject_load(malformed, route);
        }
        program_image_execute(retained); program_image_load_all(image);
    }
}
}
int main(int argc, char** argv) {
    if (argc == 2) {
        cudavm::program_image_validation_case = argv[1];
        return cudavm::run_host_test(argc, argv, cudavm::program_image_validation_host);
    }
    if (argc == 3) { cudavm::program_image_validation_case = argv[2]; argc = 2; }
    return cudavm::run_test(argc, argv, cudavm::execute_program_image_validation, 0, cudavm::probe_execution);
}
