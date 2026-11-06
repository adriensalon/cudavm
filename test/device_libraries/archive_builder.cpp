#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudavm {
std::vector<char> archive_fixture_read(const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    if (!size || size > 16 * 1024 * 1024) throw std::runtime_error("invalid fixture extent: " + path.string());
    std::vector<char> result(static_cast<size_t>(size));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(result.data(), static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("cannot read fixture: " + path.string());
    return result;
}
void archive_fixture_write(const std::filesystem::path& output, std::span<char*> paths) {
    struct member { std::string header; std::vector<char> bytes; };
    std::vector<member> members;
    for (const auto* argument : paths) {
        const std::filesystem::path path(argument);
        const auto name = path.filename().string();
        if (name.size() > 15) throw std::runtime_error("fixture archive requires short member names");
        auto bytes = archive_fixture_read(path);
        std::ostringstream header;
        header << std::left << std::setw(16) << (name + '/') << std::setw(12) << "0"
            << std::setw(6) << "0" << std::setw(6) << "0" << std::setw(8) << "100644"
            << std::setw(10) << bytes.size() << "`\n";
        if (header.str().size() != 60) throw std::runtime_error("fixture archive header extent");
        members.push_back({header.str(), std::move(bytes)});
    }
    std::ofstream archive(output, std::ios::binary | std::ios::trunc);
    archive.write("!<arch>\n", 8);
    for (const auto& member : members) {
        archive.write(member.header.data(), static_cast<std::streamsize>(member.header.size()));
        archive.write(member.bytes.data(), static_cast<std::streamsize>(member.bytes.size()));
        if (member.bytes.size() & 1) archive.put('\n');
    }
    archive.close();
    if (!archive) throw std::runtime_error("cannot write fixture archive: " + output.string());
}
void archive_fixture_embed(const std::filesystem::path& output, std::span<char*> paths) {
    constexpr std::array<std::string_view, 4> names{
        "device_library_chain_forward", "device_library_chain_reverse", "device_library_cycle_forward", "device_library_cycle_reverse"};
    if (paths.size() != names.size()) throw std::runtime_error("fixture embedding requires four owned images");
    std::array<std::vector<char>, names.size()> images;
    for (size_t i = 0; i < images.size(); ++i) images[i] = archive_fixture_read(paths[i]);
    std::ofstream header(output, std::ios::binary | std::ios::trunc);
    header << "#pragma once\nnamespace cudavm {\n";
    constexpr char hex[] = "0123456789abcdef";
    for (size_t image = 0; image < images.size(); ++image) {
        header << "inline constexpr char " << names[image] << "[] =\n";
        const auto& bytes = images[image];
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i % 512 == 0) header.put('"');
            const auto byte = static_cast<unsigned char>(bytes[i]);
            header.put('\\').put('x').put(hex[byte >> 4]).put(hex[byte & 15]);
            if (i % 512 == 511 || i + 1 == bytes.size()) header << "\"\n";
        }
        header << ";\n";
    }
    header << "}\n";
    header.close();
    if (!header) throw std::runtime_error("cannot write fixture header: " + output.string());
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: --archive|--embed <output> <inputs...>");
        const std::span<char*> paths(argv + 3, static_cast<size_t>(argc - 3));
        if (std::string_view(argv[1]) == "--archive") cudavm::archive_fixture_write(argv[2], paths);
        else if (std::string_view(argv[1]) == "--embed") cudavm::archive_fixture_embed(argv[2], paths);
        else throw std::runtime_error("unknown fixture builder mode");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
