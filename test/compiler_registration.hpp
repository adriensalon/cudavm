#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

// Private compiler ABI declarations belong to these fixtures, not public headers.
extern "C" {
void** CUDARTAPI __cudaRegisterFatBinary(void*);
void CUDARTAPI __cudaRegisterFatBinaryEnd(void**);
void CUDARTAPI __cudaUnregisterFatBinary(void**);
void CUDARTAPI __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
void CUDARTAPI __cudaRegisterVar(void**, char*, char*, const char*, int, size_t, int, int);
void CUDARTAPI __cudaRegisterManagedVar(void**, void**, char*, const char*, int, size_t, int, int);
char CUDARTAPI __cudaInitModule(void**);
}
namespace cudavm {
struct registration_wrapper { uint32_t magic = 0x466243b1, version = 1; const void* image{}; const void* auxiliary{}; };
template<class T> inline void registration_word(std::vector<unsigned char>& bytes, size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
// Independently encode one uncompressed, 64-bit NVIDIA PTX fatbinary member.
inline std::vector<unsigned char> registration_ptx_image(std::string_view source, unsigned major, unsigned minor, unsigned architecture) {
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
}
