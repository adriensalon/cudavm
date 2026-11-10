#include "driver.hpp"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <set>

namespace cudavm {
std::vector<std::string> nvcc_split_arguments(const std::string& text) {
    std::vector<std::string> result; std::string word; bool quoted = false, active = false;
    for (size_t i = 0; i < text.size();) {
        char c = text[i++];
        if (!quoted && (c == ' ' || c == '\n' || c == '\r' || c == '\t')) {
            if (active) { result.push_back(std::move(word)); word.clear(); active = false; } continue;
        }
        active = true;
        if (c == '\\') {
            size_t count = 1; while (i < text.size() && text[i] == '\\') { ++count; ++i; }
            if (i < text.size() && text[i] == '"') {
                word.append(count / 2, '\\');
                if (count % 2) { word += '"'; ++i; }
                else { quoted = !quoted; ++i; }
            } else word.append(count, '\\');
        } else if (c == '"') quoted = !quoted;
        else word += c;
    }
    if (quoted) throw std::runtime_error("unterminated quote in options file");
    if (active) result.push_back(std::move(word));
    return result;
}
void nvcc_expand_options(const std::vector<std::string>& input, std::vector<std::string>& output, unsigned depth = 0) {
    if (depth > 8) throw std::runtime_error("options-file nesting limit exceeded");
    for (size_t i = 0; i < input.size(); ++i) {
        std::string file;
        if (input[i].starts_with('@')) file = input[i].substr(1);
        else if (input[i] == "--options-file" || input[i] == "-optf") {
            if (++i == input.size()) throw std::runtime_error("missing options-file argument"); file = input[i];
        } else if (input[i].starts_with("--options-file=")) file = input[i].substr(15);
        else { output.push_back(input[i]); continue; }
        std::ifstream stream(file, std::ios::binary); if (!stream) throw std::runtime_error("cannot read options file: " + file);
        std::string text(std::istreambuf_iterator<char>(stream), {});
        if (text.starts_with("\xef\xbb\xbf")) text.erase(0, 3);
        nvcc_expand_options(nvcc_split_arguments(text), output, depth + 1);
    }
}
std::vector<std::string> nvcc_comma_list(const std::string& value) {
    std::vector<std::string> result; size_t begin = 0;
    for (;;) { auto end = value.find(',', begin); result.push_back(value.substr(begin, end - begin)); if (end == std::string::npos) return result; begin = end + 1; }
}
nvcc_options parse_nvcc_options(int argc, char** argv) {
    std::vector<std::string> input; for (int i = 1; i < argc; ++i) input.emplace_back(argv[i]);
    std::vector<std::string> args; nvcc_expand_options(input, args);
    nvcc_options result; std::set<nvcc_phase> phases;
    auto macro_name = [](const std::string& value) { return value.substr(0, value.find_first_of("=(")); };
    auto define = [&](std::string value) {
        auto name = macro_name(value);
        std::erase_if(result.defines, [&](const auto& previous) { return macro_name(previous) == name; });
        std::erase(result.undefines, name); result.defines.push_back(std::move(value));
    };
    auto undefine = [&](std::string name) {
        std::erase_if(result.defines, [&](const auto& previous) { return macro_name(previous) == name; });
        std::erase(result.undefines, name); result.undefines.push_back(std::move(name));
    };
    auto phase = [&](nvcc_phase value) { phases.insert(value); result.phase = value; };
    std::string selected_architecture;
    auto architecture = [&](std::string value) {
        if (value.starts_with("compute_")) value.replace(0, 8, "sm_");
        if (!value.starts_with("sm_") || value.size() < 5 || value.find_first_not_of("0123456789", 3) != std::string::npos)
            throw std::runtime_error("unsupported architecture: " + value);
        if (!selected_architecture.empty() && selected_architecture != value) throw std::runtime_error("this compiler profile requires one source architecture");
        selected_architecture = value; result.architecture = value;
    };
    for (size_t i = 0; i < args.size(); ++i) {
        auto arg = args[i]; auto equal = arg.find('='); auto key = arg.substr(0, equal);
        auto value = [&]() -> std::string {
            if (equal != std::string::npos) return arg.substr(equal + 1);
            if (++i == args.size()) throw std::runtime_error("missing argument for " + key); return args[i];
        };
        auto short_value = [&](size_t prefix) { return arg.size() > prefix ? arg.substr(prefix) : value(); };
        if (arg == "-c" || arg == "--compile") phase(nvcc_phase::compile);
        else if (arg == "-dc" || arg == "--device-c") { phase(nvcc_phase::device_compile); result.relocatable = true; }
        else if (arg == "-dlink" || arg == "--device-link") phase(nvcc_phase::device_link);
        else if (arg == "-lib" || arg == "--lib") phase(nvcc_phase::archive);
        else if (arg == "-ptx" || arg == "--ptx") phase(nvcc_phase::ptx);
        else if (arg == "-cubin" || arg == "--cubin" || arg == "-fatbin" || arg == "--fatbin") phase(nvcc_phase::image);
        else if (arg == "-E" || arg == "--preprocess") phase(nvcc_phase::preprocess);
        else if (arg == "-M" || arg == "-MM" || arg == "--generate-dependencies" || arg == "--generate-nonsystem-dependencies") phase(nvcc_phase::dependencies);
        else if (arg == "-MD" || arg == "-MMD" || arg == "--generate-dependencies-with-compile" || arg == "--generate-nonsystem-dependencies-with-compile") result.dep_compile = true;
        else if (key == "-MF" || key == "--dependency-output") result.depfile = value();
        else if (key == "-MT" || key == "--dependency-target-name") result.dependency_target = value();
        else if (key == "-o" || key == "--output-file") result.output = value();
        else if (key == "-std" || key == "--std") result.standard = value();
        else if (key == "-ccbin" || key == "--compiler-bindir") result.host_compiler = value();
        else if (key == "-arch" || key == "--gpu-architecture") architecture(value());
        else if (key == "-code" || key == "--gpu-code") { for (auto code : nvcc_comma_list(value())) architecture(code); }
        else if (key == "-gencode" || key == "--generate-code") {
            auto text = value();
            if (!text.starts_with("arch=compute_")) throw std::runtime_error("unsupported generate-code specification: " + text);
            auto separator = text.find(",code="); if (separator == std::string::npos) throw std::runtime_error("missing generate-code code field");
            architecture(text.substr(5, separator - 5));
            auto code = text.substr(separator + 6);
            if (code.starts_with('[') && code.ends_with(']')) code = code.substr(1, code.size() - 2);
            for (auto member : nvcc_comma_list(code)) architecture(member);
        }
        else if (key == "-rdc" || key == "--relocatable-device-code") { auto text = value(); if (text != "true" && text != "false") throw std::runtime_error("rdc requires true or false"); result.relocatable = text == "true"; }
        else if (key == "--cudart" || key == "-cudart") result.runtime = value();
        else if (key == "--cudadevrt" || key == "-cudadevrt") {
            auto setting = value(); if (setting != "none" && setting != "static") throw std::runtime_error("cudadevrt requires none or static");
            result.device_runtime = setting == "static";
        }
        else if (key == "-Xcompiler" || key == "--compiler-options") {
            std::vector<std::string> flags;
            for (auto chunk : nvcc_comma_list(value())) {
                if (chunk.find(" -") != std::string::npos || chunk.find(" /") != std::string::npos) {
                    auto split = nvcc_split_arguments(chunk); flags.insert(flags.end(), split.begin(), split.end());
                } else if (!chunk.empty()) flags.push_back(chunk);
            }
            for (auto flag : flags) {
                if (flag == "/MT" || flag == "-MT") result.host_runtime = "static";
                else if (flag == "/MD" || flag == "-MD") result.host_runtime = "dll";
                else if (flag == "/MTd" || flag == "-MTd") result.host_runtime = "static_dbg";
                else if (flag == "/MDd" || flag == "-MDd") result.host_runtime = "dll_dbg";
                else result.host_flags.push_back(flag);
            }
        }
        else if (key == "-Xlinker" || key == "--linker-options") { for (auto flag : nvcc_comma_list(value())) result.link_flags.push_back("-Wl," + flag); }
        else if (arg.starts_with("-I")) result.includes.push_back(short_value(2));
        else if (key == "-isystem" || key == "--system-include") result.includes.push_back(value());
        else if (key == "--include-path") result.includes.push_back(value());
        else if (arg.starts_with("-D")) define(short_value(2));
        else if (key == "--define-macro") define(value());
        else if (arg.starts_with("-U")) undefine(short_value(2));
        else if (key == "--undefine-macro") undefine(value());
        else if (arg.starts_with("-L")) result.library_paths.push_back(short_value(2));
        else if (key == "--library-path") result.library_paths.push_back(value());
        else if (arg.starts_with("-l") && arg != "-link") result.libraries.push_back(short_value(2));
        else if (key == "--library") result.libraries.push_back(value());
        else if (key == "-O" || key == "--optimize") { auto level = value(); if (level.size() != 1 || level[0] < '0' || level[0] > '3') throw std::runtime_error("optimization requires 0 through 3"); result.optimization = unsigned(level[0] - '0'); }
        else if (arg.size() == 3 && arg.starts_with("-O") && arg[2] >= '0' && arg[2] <= '3') result.optimization = unsigned(arg[2] - '0');
        else if (arg == "-g" || arg == "--debug") result.debug = true;
        else if (arg == "-v" || arg == "--verbose") result.verbose = true;
        else if (arg == "-V" || arg == "--version") result.version = true;
        else if (arg == "-h" || arg == "--help") result.help = true;
        else if (arg == "--keep" || arg == "-keep") result.keep = true;
        else if (key == "--keep-dir" || key == "-keep-dir") { result.keep_directory = value(); result.keep = true; }
        else if (arg == "-shared" || arg == "--shared") result.shared = true;
        else if (arg == "--forward-unknown-to-host-compiler" || arg == "-forward-unknown-to-host-compiler") result.forward_unknown = true;
        else if (arg == "-m64" || arg == "--machine=64") { /* The selected host ABI is x64. */ }
        else if (arg == "-Wno-deprecated-gpu-targets") { /* sm_75 is supported by this profile; this warning category is dormant. */ }
        else if (arg == "-x") { auto language = value(); if (language != "cu") throw std::runtime_error("this profile supports -x cu"); result.cuda_input = true; }
        else if (!arg.empty() && arg[0] == '-') {
            if (result.forward_unknown) result.host_flags.push_back(arg);
            else throw std::runtime_error("unsupported option: " + arg);
        }
        else result.inputs.push_back(arg);
    }
    if (phases.size() > 1) throw std::runtime_error("conflicting compilation phases");
    if (result.standard != "c++17" && result.standard != "c++20" && result.standard != "c++23") throw std::runtime_error("supported standards are c++17, c++20 and c++23");
    if (result.runtime != "none" && result.runtime != "shared" && result.runtime != "static") throw std::runtime_error("cudart requires none, shared or static");
    return result;
}
}
