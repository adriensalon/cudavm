#include "program_image_producers.hpp"

namespace cudavm {
std::string_view program_image_persistence_case;
const char* program_image_persistence_executable{};
const char* program_image_persistence_backend{};
std::filesystem::path program_image_persistence_export_directory;
std::unique_ptr<conformance_peer> program_image_persistence_peer;
struct program_image_cache_message {
    std::array<char, 768> path{};
    uint64_t size{}, digest{};
};
void program_image_cache_send(conformance_peer& peer, const program_image_cache_message& message) {
    peer.send(std::as_bytes(std::span(&message, 1)));
}
program_image_cache_message program_image_cache_receive(conformance_peer& peer) {
    program_image_cache_message message;
    peer.receive(std::as_writable_bytes(std::span(&message, 1)));
    require(message.path.back() == 0 && message.path.front() != 0, "bounded cache path in process control message");
    return message;
}
uint64_t program_image_cache_digest(const program_image_bytes& bytes) {
    uint64_t digest = 14695981039346656037ull;
    for (auto byte : bytes) { digest ^= byte; digest *= 1099511628211ull; }
    return digest;
}
program_image_bytes program_image_cache_read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), "open persisted program cache");
    const auto size = input.tellg();
    require(size > 0 && size < 64 * 1024 * 1024, "bounded persisted public result extent");
    program_image_bytes bytes(static_cast<size_t>(size)); input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    require(bool(input) && input.peek() == std::char_traits<char>::eof(), "read exact persisted result extent");
    return bytes;
}
void program_image_cache_write(const std::filesystem::path& path, const program_image_bytes& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size())); output.close();
    require(bool(output), "persist exact opaque public result bytes");
}
void program_image_persistence_uninitialized() {
    CUcontext context{};
    require(cuCtxGetCurrent(&context) == CUDA_ERROR_NOT_INITIALIZED, "persistence producer initializes no GPU");
}
void program_image_persistence_producer() {
    program_image_persistence_uninitialized();
    auto message = program_image_cache_receive(*program_image_persistence_peer);
    const auto producer = program_image_persistence_case.substr(std::string_view("produce.").size());
    const auto image = program_image_produce(producer);
    program_image_cache_write(message.path.data(), image);
    message.size = image.size(); message.digest = program_image_cache_digest(image);
    program_image_persistence_uninitialized();
    program_image_cache_send(*program_image_persistence_peer, message);
}
program_image_cache_message program_image_persistence_cache(program_image_file& file, std::string_view producer) {
    program_image_cache_message message;
    const auto original_path = file.path.string();
    require(original_path.size() < message.path.size(), "owned cache path fits the bounded control protocol");
    std::copy(original_path.begin(), original_path.end(), message.path.begin());
    const auto child_case = "produce." + std::string(producer);
    {
        conformance_peer child(program_image_persistence_executable, program_image_persistence_backend, child_case.c_str());
        program_image_cache_send(child, message); message = program_image_cache_receive(child);
        require(child.wait() == 0, "producer process exits successfully before cache consumption");
    }
    auto original = program_image_cache_read(file.path);
    require(original.size() == message.size && program_image_cache_digest(original) == message.digest, "persisted result matches the retired producer's extent and digest");
    program_image_bytes copied(original.begin(), original.end());
    std::fill(original.begin(), original.end(), 0x57); original.clear(); original.shrink_to_fit();
    const auto cached_path = file.root / "copied cache.image";
    program_image_cache_write(cached_path, copied);
    require(std::filesystem::remove(file.path), "remove the producer's original file before consuming the caller's cache copy");
    file.path = cached_path;
    const auto path = file.path.string(); require(path.size() < message.path.size(), "copied cache path extent");
    message.path.fill(0); std::copy(path.begin(), path.end(), message.path.begin());
    require(program_image_cache_read(file.path) == copied, "cache copy preserves every opaque result byte");
    return message;
}
void program_image_persistence_consumer() {
    const auto message = program_image_cache_receive(*program_image_persistence_peer);
    const auto image = program_image_cache_read(message.path.data());
    require(image.size() == message.size && program_image_cache_digest(image) == message.digest, "fresh loader receives the complete copied cache after producer exit");
    const bool first = program_image_persistence_case == "load_first";
    require(first || program_image_persistence_case == "load_second", "known fresh loader group");
    for (unsigned route = first ? 0 : 4; route < (first ? 4u : 7u); ++route) {
        program_image_loaded program(image, route); program_image_execute(program); CONFORMANCE_CUDA(cuCtxSynchronize());
        program_image_cache_send(*program_image_persistence_peer, message);
    }
    require(program_image_cache_read(message.path.data()) == image, "execution and module storage updates preserve the persistent cache");
}
void execute_program_image_persistence() {
    if (program_image_persistence_peer) { program_image_persistence_consumer(); return; }
    program_image_file cache({});
    const auto message = program_image_persistence_cache(cache, program_image_persistence_case);
    for (const char* group : {"load_first", "load_second"}) {
        conformance_peer child(program_image_persistence_executable, program_image_persistence_backend, group);
        program_image_cache_send(child, message);
        const auto count = std::string_view(group) == "load_first" ? 4u : 3u;
        for (unsigned i = 0; i < count; ++i) {
            const auto reply = program_image_cache_receive(child);
            require(reply.size == message.size && reply.digest == message.digest, "fresh process completes each independent guarded execution route");
        }
        require(child.wait() == 0, "fresh loader process completes and retires its private module storage");
        const auto bytes = program_image_cache_read(cache.path);
        require(bytes.size() == message.size && program_image_cache_digest(bytes) == message.digest, "persistent cache remains reusable after loader process exit");
    }
}
void program_image_persistence_copies() {
    program_image_persistence_uninitialized();
    for (auto producer : program_image_producers) {
        program_image_file cache({}); program_image_persistence_cache(cache, producer);
    }
    program_image_persistence_uninitialized();
}
program_image_bytes program_image_persistence_probe_image() {
    compiler_fixture_program program(R"cu(
extern "C" __global__ void cached_availability_probe(unsigned* out) { *out = 0x43564d31u; }
)cu", "cached_probe.cu");
    require(program.compile({"-arch=sm_75", "--std=c++17"}) == NVRTC_SUCCESS, "independent cached execution probe compiles");
    return program_image_result([&](size_t* size) { compiler_fixture_program::check(nvrtcGetCUBINSize(program.value, size)); },
        [&](unsigned char* bytes) { compiler_fixture_program::check(nvrtcGetCUBIN(program.value, reinterpret_cast<char*>(bytes))); });
}
void program_image_persistence_export() {
    program_image_persistence_uninitialized();
    require(std::filesystem::is_directory(program_image_persistence_export_directory), "cache export directory is owned by the process harness");
    program_image_cache_write(program_image_persistence_export_directory / "cached.image", program_image_nvrtc_produce("nvrtc_cubin"));
    program_image_cache_write(program_image_persistence_export_directory / "portable.image", program_image_nvrtc_produce("nvrtc_lto"));
    program_image_cache_write(program_image_persistence_export_directory / "probe.image", program_image_persistence_probe_image());
    program_image_persistence_uninitialized();
}
}
int main(int argc, char** argv) {
    cudavm::program_image_persistence_executable = argv[0];
    cudavm::program_image_persistence_backend = argc >= 2 ? argv[1] : "host";
    if (argc == 3 && std::string_view(argv[1]) == "--cache-files") {
        cudavm::program_image_persistence_export_directory = argv[2];
        return cudavm::run_host_conformance(1, argv, cudavm::program_image_persistence_export);
    }
    if (argc == 7 && std::string_view(argv[3]) == "--peer") {
        cudavm::program_image_persistence_peer = std::make_unique<cudavm::conformance_peer>(cudavm::conformance_peer::connect(argv[4], argv[5], argv[6]));
        cudavm::program_image_persistence_case = argv[2];
        if (cudavm::program_image_persistence_case.starts_with("produce."))
            return cudavm::run_host_conformance(1, argv, cudavm::program_image_persistence_producer);
        argc = 3;
    }
    if (argc == 2 && std::string_view(argv[1]) == "copies") {
        cudavm::program_image_persistence_backend = "host";
        return cudavm::run_host_conformance(argc, argv, cudavm::program_image_persistence_copies);
    }
    if (argc == 3) { cudavm::program_image_persistence_case = argv[2]; argc = 2; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_program_image_persistence, 0, cudavm::probe_execution);
}
