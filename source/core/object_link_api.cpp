#include "object_link_api.hpp"
#include "compiler.hpp"
#include "program.hpp"
#include "host_link.hpp"
#include <algorithm>
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/Object/Archive.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/raw_ostream.h>

namespace cudavm {
bool collect_object_device_inputs(compiler_options& values, llvm::MemoryBufferRef buffer, const std::string& origin, bool library) {
    if (llvm::identify_magic(buffer.getBuffer()) == llvm::file_magic::coff_import_library) return false;
    auto object = llvm::object::ObjectFile::createObjectFile(buffer);
    if (!object) throw std::runtime_error("invalid host object " + origin + ": " + llvm::toString(object.takeError()));
    bool completed = false;
    for (const auto& section : (*object)->sections()) {
        auto name = section.getName();
        if (!name) throw std::runtime_error(llvm::toString(name.takeError()));
        if (*name != ".cudavm") continue;
        auto content = section.getContents();
        if (!content) throw std::runtime_error(llvm::toString(content.takeError()));
        auto bytes = *content;
        // A COFF section may have alignment padding after the framed record.
        while (!bytes.empty()) {
            if (bytes.find_first_not_of('\0') == llvm::StringRef::npos) break;
            if (bytes.size() < 16) {
                if (bytes.find_first_not_of('\0') == llvm::StringRef::npos) break;
                throw std::runtime_error("truncated device object record: " + origin);
            }
            auto magic = bytes.take_front(8);
            if (magic != "CVOBJ001" && magic != "CVLNK001") throw std::runtime_error("unsupported device object record: " + origin);
            uint64_t size = 0;
            for (unsigned i = 0; i < 8; ++i) size |= uint64_t(static_cast<unsigned char>(bytes[8 + i])) << (8 * i);
            if (size > bytes.size() - 16 || (magic == "CVLNK001" && size)) throw std::runtime_error("invalid device object record extent: " + origin);
            if (magic == "CVLNK001") { if (!library) completed = true; }
            else {
                if (!size) throw std::runtime_error("empty device object payload: " + origin);
                values.ir_inputs.push_back({bytes.substr(16, size).str(), origin, library});
            }
            bytes = bytes.drop_front(16 + size);
        }
    }
    return completed;
}

extern "C" CUDAVM_CORE_API int cudavm_core_device_runtime_v1(cudavm::compiler_result** output) noexcept {
    using namespace cudavm;
    return run_compiler_file_api(output, [&](compiler_options& values, compiler_result& result) {
        std::lock_guard lock(compiler_mutex());
        const char* source = R"(
#undef __cudaCDP2RuntimeGetVersion
extern "C" __device__ __attribute__((used)) cudaError_t __cudaCDP2RuntimeGetVersion(int* output) {
    if (!output) return cudaErrorInvalidValue;
    *output = CUDART_VERSION;
    return cudaSuccess;
}
)";
        compiler_source_request settings{1, source, "cudadevrt.cu", "c++17", "sm_75", 1, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr};
        compiler_input_request request{1, compiler_input_kind::source, nullptr, 0, "cudadevrt.cu", &settings};
        compiler_result compilation;
        std::unique_ptr<compiler_program> program;
        try { program = import_compiler_program(request, compilation); }
        catch (...) { values.diagnostics += compilation.diagnostics; throw; }
        auto* version = program->module->getFunction("__cudaCDP2RuntimeGetVersion");
        require_compiler_request(version && !version->isDeclaration(), "device Runtime version helper is absent");
        version->addFnAttr("cudavm.device.runtime.version");
        std::vector<llvm::NamedMDNode*> private_metadata;
        for (auto& metadata : program->module->named_metadata()) if (metadata.getName().starts_with("cudavm.")) private_metadata.push_back(&metadata);
        for (auto* metadata : private_metadata) program->module->eraseNamedMetadata(metadata);
        llvm::raw_string_ostream stream(result.output); llvm::WriteBitcodeToFile(*program->module, stream);
    });
}
}

extern "C" CUDAVM_CORE_API int cudavm_core_link_objects_v1(const cudavm::compiler_object_link_request* request, cudavm::compiler_result** output) noexcept {
    using namespace cudavm;
    return run_compiler_file_api(output, [&](compiler_options& values, compiler_result&) {
        std::lock_guard lock(compiler_mutex());
        require_compiler_request(request && request->abi_version == 1 && request->module
            && request->optimization_level <= 3 && request->skip_completed <= 1, "invalid object-link request");
        append_compiler_option_strings(request->input_count, request->inputs, values.host_inputs);
        if (request->architecture) values.architecture = request->architecture;
        bool completed = false;
        for (const auto& path : values.host_inputs) {
            auto file = llvm::MemoryBuffer::getFile(path);
            if (!file) throw std::runtime_error("cannot read host input " + path + ": " + file.getError().message());
            if ((*file)->getBuffer().starts_with("!<thin>\n")) throw std::runtime_error("thin archives are unsupported: " + path);
            if (!(*file)->getBuffer().starts_with("!<arch>\n")) {
                completed |= collect_object_device_inputs(values, (*file)->getMemBufferRef(), path, false); continue;
            }
            auto archive = llvm::object::Archive::create((*file)->getMemBufferRef());
            if (!archive) throw std::runtime_error(llvm::toString(archive.takeError()));
            llvm::Error error = llvm::Error::success();
            for (const auto& child : (*archive)->children(error)) {
                auto name = child.getName(); if (!name) throw std::runtime_error(llvm::toString(name.takeError()));
                auto buffer = child.getMemoryBufferRef(); if (!buffer) throw std::runtime_error(llvm::toString(buffer.takeError()));
                collect_object_device_inputs(values, *buffer, path + "(" + name->str() + ")", true);
            }
            if (error) throw std::runtime_error(llvm::toString(std::move(error)));
        }
        if (completed && request->skip_completed) return;
        require_compiler_request(!completed, "completed device-link objects cannot be linked again");
        if (values.ir_inputs.empty()) return;
        if (std::none_of(values.ir_inputs.begin(), values.ir_inputs.end(), [](const auto& input) { return !input.library; })
            && host_link(values.host_inputs).select({}).empty()) return;
        compiler_program linked; linked.architecture = values.architecture;
        linked.module = link_device(linked.context, values);
        auto prepared = prepare_compiler_program(linked, {1, 0, 1, request->optimization_level});
        compiler_emit_request emit; emit.kind = compiler_output_kind::program_image; emit.environment = compiler_environment::none;
        compiler_result image; emit_compiler_program(*prepared, emit, image);
        write_file(request->module, embed_cuda(*prepared->module, image.output));
    });
}
