#include "program_image_producers.hpp"
#include "program_image_packaging.hpp"
namespace cudavm {
std::string_view program_image_roundtrip_case = "nvrtc_cubin.direct";
void execute_program_image_roundtrip() {
    const auto separator = program_image_roundtrip_case.find('.'); require(separator != std::string_view::npos, "program producer and consumer family are explicit");
    const auto producer = program_image_roundtrip_case.substr(0, separator), family = program_image_roundtrip_case.substr(separator + 1);
    const auto image = program_image_produce(producer);
    if (family == "direct") program_image_load_all(image);
    else if (family == "linker") {
        for (auto kind : {NVJITLINK_INPUT_CUBIN, NVJITLINK_INPUT_LTOIR, NVJITLINK_INPUT_ANY}) for (bool file : {false, true})
            program_image_load_all(program_image_link(image, kind, file));
    } else {
        require(family == "fatbin", "known program consumer family");
        for (bool lto : {false, true}) {
            std::cout << "program_image package_member=" << (lto ? "ltoir" : "cubin") << std::endl;
            const auto container = program_image_package(image, lto); program_image_load_all(container, true);
            for (auto kind : {NVJITLINK_INPUT_FATBIN, NVJITLINK_INPUT_ANY}) for (bool file : {false, true}) {
                const auto linked = program_image_link(container, kind, file);
                for (unsigned route : {0u, 3u}) { program_image_loaded program(linked, route); program_image_execute(program); }
            }
        }
    }
}
void program_image_roundtrip_producers() {
    for (auto producer : program_image_producers) program_image_link(program_image_produce(producer), NVJITLINK_INPUT_ANY, false, true);
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "producers") return cudavm::run_host_test(argc, argv, cudavm::program_image_roundtrip_producers);
    if (argc == 3) { cudavm::program_image_roundtrip_case = argv[2]; argc = 2; }
    return cudavm::run_test(argc, argv, cudavm::execute_program_image_roundtrip, 0, cudavm::probe_execution);
}
