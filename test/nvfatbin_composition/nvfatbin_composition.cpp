#include "test_framework.hpp"
#include <nvFatbin.h>
#include <nvJitLink.h>
#include <nvPTXCompiler.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <utility>

namespace cudavm {
void composition_pack(nvFatbinResult result) {
    if (result != NVFATBIN_SUCCESS) throw std::runtime_error(std::string("nvFatbin composition: ") + nvFatbinGetErrorString(result));
}
void composition_jit(nvJitLinkResult result) { require(result == NVJITLINK_SUCCESS, "nvFatbin downstream linker status " + std::to_string(result)); }
void composition_ptx(nvPTXCompileResult result) { require(result == NVPTXCOMPILE_SUCCESS, "nvFatbin image producer status " + std::to_string(result)); }
template<class T> struct composition_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    composition_output() { before.fill(0xf3); after.fill(0x51); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0xf3; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x51; }), "container composition scalar output guards");
        return value;
    }
};
std::string composition_ptx_source(unsigned architecture, const std::string& kernel) {
    return std::string(architecture >= 80 ? ".version 7.0\n" : ".version 6.5\n")
        + ".target sm_" + std::to_string(architecture) + "\n.address_size 64\n.visible .entry " + kernel + "() { ret; }\n";
}
struct composition_container {
    nvFatbinHandle value{};
    explicit composition_container(const std::vector<std::string>& options = {"-compress=false"}) {
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        composition_output<nvFatbinHandle> output;
        composition_pack(nvFatbinCreate(&output.value, pointers.data(), pointers.size())); value = output.get();
        require(value != nullptr, "container composition publishes packager handle");
    }
    ~composition_container() { if (value) nvFatbinDestroy(&value); }
    composition_container(const composition_container&) = delete;
    composition_container& operator=(const composition_container&) = delete;
    void add(unsigned architecture, const std::string& kernel) const {
        auto source = composition_ptx_source(architecture, kernel);
        // Repetitive caller-owned data exercises forced/default compression paths.
        for (unsigned line = 0; line < 128; ++line) source += "// conformance lossless container payload padding\n";
        auto arch = std::to_string(architecture), identifier = kernel;
        composition_pack(nvFatbinAddPTX(value, source.data(), source.size(), arch.c_str(), identifier.c_str(), ""));
        std::fill(source.begin(), source.end(), '!'); std::fill(identifier.begin(), identifier.end(), '!');
    }
    std::vector<unsigned char> image() const {
        composition_output<size_t> size, repeated;
        composition_pack(nvFatbinSize(value, &size.value)); composition_pack(nvFatbinSize(value, &repeated.value));
        const auto count = size.get(); require(count > 0 && repeated.get() == count, "packaged container has stable nonempty extent");
        std::vector<unsigned char> first(count + 32, 0x2b), second(count + 32, 0x65);
        composition_pack(nvFatbinGet(value, first.data() + 16)); composition_pack(nvFatbinGet(value, second.data() + 16));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= count + 16) require(first[i] == 0x2b && second[i] == 0x65, "container Get exact binary guards");
            else require(first[i] == second[i], "container Get writes all reported bytes");
        }
        return {first.begin() + 16, first.end() - 16};
    }
};
struct composition_link {
    nvJitLinkHandle value{};
    explicit composition_link(unsigned architecture = 75) {
        const auto target = "-arch=sm_" + std::to_string(architecture);
        const char* options[]{target.c_str(), "-lto", "-ptx", "-O0"};
        composition_output<nvJitLinkHandle> output;
        composition_jit(nvJitLinkCreate(&output.value, 4, options)); value = output.get();
        require(value != nullptr, "downstream container linker publishes handle");
    }
    ~composition_link() { if (value) nvJitLinkDestroy(&value); }
    composition_link(const composition_link&) = delete;
    composition_link& operator=(const composition_link&) = delete;
    void check(nvJitLinkResult result, const char* operation) const {
        if (result == NVJITLINK_SUCCESS) return;
        size_t size = 0;
        std::string diagnostic;
        if (nvJitLinkGetErrorLogSize(value, &size) == NVJITLINK_SUCCESS && size) {
            std::vector<char> bytes(size);
            if (nvJitLinkGetErrorLog(value, bytes.data()) == NVJITLINK_SUCCESS) diagnostic.assign(bytes.data());
        }
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result) + "; " + diagnostic);
    }
    void add(const std::vector<unsigned char>& image, nvJitLinkInputType kind = NVJITLINK_INPUT_FATBIN) const {
        check(nvJitLinkAddData(value, kind, image.data(), image.size(), "composition.container"), "container AddData");
    }
    std::string finish() const {
        check(nvJitLinkComplete(value), "container Complete");
        composition_output<size_t> size;
        composition_jit(nvJitLinkGetLinkedPtxSize(value, &size.value));
        const auto count = size.get(); require(count > 1, "container linker returns nonempty PTX text");
        std::vector<char> first(count + 32, char(0x3b)), second(count + 32, char(0x71));
        composition_jit(nvJitLinkGetLinkedPtx(value, first.data() + 16)); composition_jit(nvJitLinkGetLinkedPtx(value, second.data() + 16));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= count + 16) require(first[i] == char(0x3b) && second[i] == char(0x71), "downstream PTX exact byte guards");
            else require(first[i] == second[i], "downstream PTX Get writes all bytes");
        }
        require(first[count + 15] == 0 && std::find(first.begin() + 16, first.begin() + count + 15, '\0') == first.begin() + count + 15, "downstream PTX exact NUL termination");
        return {first.begin() + 16, first.begin() + count + 15};
    }
};
void composition_selected(const std::string& text, const std::string& selected, const std::string& excluded = {}) {
    require(text.find(selected) != std::string::npos, "downstream PTX retains independently selected container member");
    if (!excluded.empty()) require(text.find(excluded) == std::string::npos, "downstream PTX excludes unselected alternative");
}
void composition_targets() {
    composition_container container;
    container.add(75, "fatbin_choice75"); container.add(80, "fatbin_choice80");
    const auto image = container.image();
    for (unsigned architecture : {75u, 80u}) for (auto kind : {NVJITLINK_INPUT_FATBIN, NVJITLINK_INPUT_ANY}) {
        composition_link link(architecture); link.add(image, kind);
        composition_selected(link.finish(), architecture == 75 ? "fatbin_choice75" : "fatbin_choice80", architecture == 75 ? "fatbin_choice80" : "fatbin_choice75");
    }
    composition_container newer; newer.add(80, "fatbin_only80");
    const auto unavailable = newer.image(); composition_link older(75);
    require(nvJitLinkAddData(older.value, NVJITLINK_INPUT_FATBIN, unavailable.data(), unavailable.size(), "newer.container") == NVJITLINK_ERROR_ARCH_MISMATCH, "link target older than every member has architecture-mismatch status");
    older.add(image); composition_selected(older.finish(), "fatbin_choice75", "fatbin_choice80");
}
void composition_compression() {
    const std::vector<std::vector<std::string>> modes{{"-compress=false"}, {"-compress-all"}, {"-compress-mode=speed"},
        {"-compress-mode=size"}, {"-compress-mode=balance"}, {"-compress=true", "-compress-mode=none"}};
    for (const auto& mode : modes) {
        composition_container container(mode); container.add(75, "fatbin_compressed75"); container.add(80, "fatbin_compressed80");
        const auto image = container.image(); composition_link link;
        link.add(image); composition_selected(link.finish(), "fatbin_compressed75", "fatbin_compressed80");
    }
}
struct composition_file {
    std::filesystem::path root, path;
    explicit composition_file(const std::vector<unsigned char>& bytes) {
        root = std::filesystem::current_path() / ("nvfatbin_composition_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create unique container file fixture directory");
        path = root / "input.fatbin";
        try {
            std::ofstream output(path, std::ios::binary); output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            require(bool(output), "write exact container bytes to caller file");
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() const noexcept { std::error_code ignored; std::filesystem::remove(path, ignored); std::filesystem::remove(root, ignored); }
    ~composition_file() { cleanup(); }
};
void composition_files() {
    std::vector<unsigned char> image;
    { composition_container container({"-compress-all"}); container.add(75, "fatbin_file75"); image = container.image(); }
    for (auto kind : {NVJITLINK_INPUT_FATBIN, NVJITLINK_INPUT_ANY}) {
        composition_file file(image); composition_link link;
        composition_jit(nvJitLinkAddFile(link.value, kind, file.path.string().c_str()));
        file.cleanup(); // The downstream linker must own input before returning.
        composition_selected(link.finish(), "fatbin_file75");
    }
}
std::vector<unsigned char> composition_program_image() {
    const auto source = composition_ptx_source(75, "fatbin_image75");
    nvPTXCompilerHandle producer{}; composition_ptx(nvPTXCompilerCreate(&producer, source.size(), source.data()));
    struct retire { nvPTXCompilerHandle value; ~retire() { nvPTXCompilerDestroy(&value); } } owner{producer};
    const char* options[]{"-arch=sm_75", "-c"}; composition_ptx(nvPTXCompilerCompile(producer, 2, options));
    composition_output<size_t> size; composition_ptx(nvPTXCompilerGetCompiledProgramSize(producer, &size.value));
    require(size.get() > 0, "portable compiler member has nonempty opaque extent");
    std::vector<unsigned char> result(size.get()); composition_ptx(nvPTXCompilerGetCompiledProgram(producer, result.data())); return result;
}
void composition_images() {
    auto image = composition_program_image();
    for (bool lto : {false, true}) {
        composition_container container;
        if (lto) composition_pack(nvFatbinAddLTOIR(container.value, image.data(), image.size(), "75", "portable_lto", ""));
        else {
            container.add(75, "fatbin_fallback75");
            composition_pack(nvFatbinAddCubin(container.value, image.data(), image.size(), "75", "portable_cubin"));
        }
        const auto packaged = container.image(); composition_link link; link.add(packaged);
        composition_selected(link.finish(), "fatbin_image75", lto ? "" : "fatbin_fallback75");
    }
}
void composition_ownership() {
    std::vector<unsigned char> image;
    { composition_container container; container.add(75, "fatbin_owned75"); image = container.image(); }
    const auto retained = image;
    composition_link link; link.add(image);
    std::fill(image.begin(), image.end(), 0x85);
    composition_selected(link.finish(), "fatbin_owned75");
    composition_link another; another.add(retained, NVJITLINK_INPUT_ANY);
    composition_selected(another.finish(), "fatbin_owned75");
}
std::string_view nvfatbin_composition_case = "all";
void execute_nvfatbin_composition() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"targets", composition_targets}, {"compression", composition_compression}, {"files", composition_files},
        {"images", composition_images}, {"ownership", composition_ownership}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (nvfatbin_composition_case == "all" || nvfatbin_composition_case == name) { selected = true; execute(); }
    require(selected, "unknown nvfatbin_composition case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvfatbin_composition_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvfatbin_composition);
}
