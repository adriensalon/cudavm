#include "conformance.hpp"
#include "cubin_fixture.hpp"
#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
#include <algorithm>

namespace cudavm {
using cubin_bytes = std::vector<unsigned char>;
std::string_view cubin_import_case;
uint64_t cubin_number(const cubin_bytes& bytes, size_t offset, unsigned width = 8) {
    require(width <= 8 && offset <= bytes.size() && width <= bytes.size() - offset, "bounded fixture ELF read");
    uint64_t result = 0;
    for (unsigned i = 0; i < width; ++i) result |= uint64_t(bytes[offset + i]) << (i * 8);
    return result;
}
void cubin_put(cubin_bytes& bytes, size_t offset, uint64_t value, unsigned width = 8) {
    require(offset <= bytes.size() && width <= bytes.size() - offset, "bounded fixture ELF edit");
    for (unsigned i = 0; i < width; ++i) bytes[offset + i] = static_cast<unsigned char>(value >> (i * 8));
}
size_t cubin_section_header(const cubin_bytes& bytes, std::string_view section_name) {
    const auto table = cubin_number(bytes, 40), stride = cubin_number(bytes, 58, 2), count = cubin_number(bytes, 60, 2);
    const auto names = table + cubin_number(bytes, 62, 2) * stride, strings = cubin_number(bytes, names + 24);
    for (size_t i = 0; i < count; ++i) {
        const auto header = table + i * stride, name = strings + cubin_number(bytes, header, 4);
        require(name < bytes.size(), "bounded fixture section name");
        const auto end = std::find(bytes.begin() + name, bytes.end(), 0);
        require(end != bytes.end(), "terminated fixture section name");
        const std::string text(bytes.begin() + name, end);
        if (text == section_name) return size_t(header);
    }
    throw std::runtime_error("missing fixture section " + std::string(section_name));
}
size_t cubin_instruction(const cubin_bytes& bytes, unsigned opcode, bool uniform_address = false,
        std::string_view kernel = "frame_results") {
    const auto header = cubin_section_header(bytes, ".text." + std::string(kernel));
    const auto start = size_t(cubin_number(bytes, header + 24)), size = size_t(cubin_number(bytes, header + 32));
    require(start <= bytes.size() && size <= bytes.size() - start && size % 16 == 0, "bounded fixture instruction section");
    for (size_t offset = start; offset < start + size; offset += 16)
        if ((cubin_number(bytes, offset) & 0xfff) == opcode
            && (!uniform_address || (cubin_number(bytes, offset + 8) & 0x8000000))) return offset;
    throw std::runtime_error("missing checked fixture opcode " + std::to_string(opcode));
}
void cubin_link_check(nvJitLinkResult status) { require(status == NVJITLINK_SUCCESS, "cubin import API status " + std::to_string(status)); }
struct cubin_link {
    nvJitLinkHandle value{};
    cubin_link() {
        const char* options[]{"-arch=sm_75", "-r"};
        cubin_link_check(nvJitLinkCreate(&value, 2, options));
    }
    ~cubin_link() { if (value) nvJitLinkDestroy(&value); }
    nvJitLinkResult add(const cubin_bytes& bytes, nvJitLinkInputType type = NVJITLINK_INPUT_CUBIN, const char* name = "checked_stack.cubin") {
        return nvJitLinkAddData(value, type, bytes.data(), bytes.size(), name);
    }
    std::string log() const {
        size_t size = 0; cubin_link_check(nvJitLinkGetErrorLogSize(value, &size));
        std::vector<char> text(size + 1); cubin_link_check(nvJitLinkGetErrorLog(value, text.data()));
        return text.data();
    }
    cubin_bytes complete() {
        cubin_link_check(nvJitLinkComplete(value));
        struct { uint64_t before = 0x123456789abcdef0; size_t value = 0; uint64_t after = 0xfedcba9876543210; } size;
        cubin_link_check(nvJitLinkGetLinkedCubinSize(value, &size.value));
        require(size.before == 0x123456789abcdef0 && size.after == 0xfedcba9876543210 && size.value > 0, "guarded portable result size");
        cubin_bytes first(size.value + 32, 0xa5), second(size.value + 32, 0x5a);
        cubin_link_check(nvJitLinkGetLinkedCubin(value, first.data() + 16));
        cubin_link_check(nvJitLinkGetLinkedCubin(value, second.data() + 16));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= size.value + 16) require(first[i] == 0xa5 && second[i] == 0x5a, "cubin result output guards");
            else require(first[i] == second[i], "cubin result writes its complete reported extent");
        }
        return {first.begin() + 16, first.end() - 16};
    }
};
cubin_bytes cubin_original() { return {std::begin(cubin_import_fixture), std::end(cubin_import_fixture)}; }
void cubin_rejection(cubin_bytes invalid, nvJitLinkResult expected, std::string_view diagnostic,
        const cubin_bytes& recovery = cubin_original()) {
    cubin_link link;
    require(link.add(invalid, NVJITLINK_INPUT_CUBIN, "mutated_stack.cubin") == expected, "exact malformed/unsupported cubin public status");
    const auto log = link.log();
    require(log.find("mutated_stack.cubin") != log.npos && log.find(diagnostic) != log.npos, "cubin input diagnostic identifies label and rejected contract: " + log);
    size_t size = 999;
    require(nvJitLinkGetLinkedCubinSize(link.value, &size) == NVJITLINK_ERROR_INTERNAL, "rejected import publishes no linked result");
    cubin_link_check(link.add(recovery));
    require(!link.complete().empty(), "rejected import leaves the handle usable for a valid input");
}
void execute_cubin_import() {
    const auto original = cubin_original();
    if (cubin_import_case == "ownership") {
        for (auto type : {NVJITLINK_INPUT_CUBIN, NVJITLINK_INPUT_ANY}) {
            cubin_bytes result;
            {
                cubin_link producer; auto bytes = original; std::string name = "owned_stack.cubin";
                cubin_link_check(producer.add(bytes, type, name.c_str()));
                std::fill(bytes.begin(), bytes.end(), '!'); std::fill(name.begin(), name.end(), '?');
                result = producer.complete();
            }
            cubin_link consumer; cubin_link_check(consumer.add(result));
            std::fill(result.begin(), result.end(), '!'); require(!consumer.complete().empty(), "copied result relinks after producer destruction");
        }
    } else if (cubin_import_case == "metadata") {
        cubin_rejection({original.begin(), original.begin() + 63}, NVJITLINK_ERROR_INVALID_INPUT, "cubin");
        auto invalid = original; invalid.push_back(0); cubin_rejection(invalid, NVJITLINK_ERROR_INVALID_INPUT, "size mismatch");
        invalid = original; invalid[48] = 80; cubin_rejection(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "SM75");
        const auto text = cubin_section_header(original, ".text.frame_results");
        for (uint64_t flags : {uint64_t(6) | (uint64_t(17) << 20), uint64_t(6) | (uint64_t(1) << 19)}) {
            invalid = original; cubin_put(invalid, text + 8, flags);
            cubin_rejection(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "code section");
        }
    } else if (cubin_import_case == "registers") {
        const auto at = cubin_instruction(original, 0x3c2), lo = cubin_number(original, at), hi = cubin_number(original, at + 8);
        auto invalid = original; cubin_put(invalid, at, (lo & ~uint64_t(0xff000000)) | uint64_t(12) << 24);
        cubin_rejection(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "nonconstant register-to-uniform");
        invalid = original; cubin_put(invalid, at, lo & ~uint64_t(0x7000)); cubin_rejection(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "instruction encoding");
        invalid = original; cubin_put(invalid, at + 8, hi | 1); cubin_rejection(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "instruction encoding");
        invalid = original; cubin_put(invalid, at, (lo & ~uint64_t(0xff0000)) | uint64_t(63) << 16);
        cubin_rejection(invalid, NVJITLINK_ERROR_INVALID_INPUT, "invalid uniform register");
    } else if (cubin_import_case == "addressing") {
        auto at = cubin_instruction(original, 0xc11); auto invalid = original;
        cubin_put(invalid, at + 8, cubin_number(original, at + 8) ^ 0x100); cubin_rejection(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "instruction encoding");
        at = cubin_instruction(original, 0x987); invalid = original;
        cubin_put(invalid, at + 8, cubin_number(original, at + 8) ^ 0x10000); cubin_rejection(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "local memory encoding");
        at = cubin_instruction(original, 0x983, true); invalid = original;
        cubin_put(invalid, at, (cubin_number(original, at) & ~uint64_t(0xff00000000)) | uint64_t(64) << 32);
        cubin_rejection(invalid, NVJITLINK_ERROR_INVALID_INPUT, "invalid uniform register");
    } else if (cubin_import_case == "spaces") {
        const cubin_bytes address(std::begin(cubin_address_fixture), std::end(cubin_address_fixture));
        auto rejected = [&](cubin_bytes bytes, nvJitLinkResult expected, std::string_view diagnostic) {
            cubin_rejection(std::move(bytes), expected, diagnostic, address);
        };
        auto at = cubin_instruction(address, 0x3aa, false, "address_results");
        auto invalid = address; cubin_put(invalid, at + 8, cubin_number(address, at + 8) | 0x10);
        rejected(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "address-space query encoding");
        invalid = address; cubin_put(invalid, at, cubin_number(address, at) & ~uint64_t(0xff0000));
        rejected(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "address-space query encoding");
        invalid = address; cubin_put(invalid, at, cubin_number(address, at) | uint64_t(1) << 24);
        rejected(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "address-space query encoding");
        at = cubin_instruction(address, 0x980, false, "address_results");
        invalid = address; cubin_put(invalid, at + 8, cubin_number(address, at + 8) | 0x1000000);
        rejected(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "generic load encoding");
        invalid = address; cubin_put(invalid, at, cubin_number(address, at) | uint64_t(1) << 24);
        rejected(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "generic load encoding");
        const auto shared = cubin_section_header(address, ".nv.shared.address_results");
        invalid = address; cubin_put(invalid, shared + 32, 0);
        rejected(invalid, NVJITLINK_ERROR_UNRECOGNIZED_INPUT, "generic address requires defined shared/local storage");
        const auto constant = cubin_section_header(address, ".nv.constant2.address_results");
        invalid = address; cubin_put(invalid, constant + 44, 0, 4);
        rejected(invalid, NVJITLINK_ERROR_INVALID_INPUT, "global/constant storage section");
    } else throw std::runtime_error("unknown cubin_import case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::cubin_import_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_cubin_import);
}
