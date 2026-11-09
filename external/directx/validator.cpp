#include "../../source/cuda/directx_validation.hpp"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/DxilHash/DxilHash.h"
#include "dxc/DxilValidation/DxilValidation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <limits>
#include <mutex>

namespace cudavm {
namespace {
struct directx_validator_runtime {
    HRESULT status = DxcInitThreadMalloc();
    bool initialized = SUCCEEDED(status);
    directx_validator_runtime() {
        if (!initialized) return;
        DxcThreadMalloc allocator(nullptr);
        if (llvm::sys::fs::SetupPerThreadFileSystem()) status = E_FAIL;
    }
    ~directx_validator_runtime() {
        if (!initialized) return;
        {
            DxcThreadMalloc allocator(nullptr);
            llvm::sys::fs::CleanupPerThreadFileSystem();
            llvm::llvm_shutdown();
        }
        DxcCleanupThreadMalloc();
    }
    std::mutex mutex;
};
}

int32_t validate_directx_dxil(std::string& bytes, std::string& diagnostic) {
    diagnostic.clear();
    if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
        diagnostic = "DXIL container exceeds the validator size limit";
        return E_INVALIDARG;
    }
    static directx_validator_runtime runtime;
    if (FAILED(runtime.status)) {
        diagnostic = "DXIL validator initialization failed";
        return runtime.status;
    }
    std::lock_guard<std::mutex> lock(runtime.mutex);
    DxcThreadMalloc allocator(nullptr);
    try {
        llvm::raw_string_ostream errors(diagnostic);
        auto status = hlsl::ValidateDxilContainer(bytes.data(), static_cast<uint32_t>(bytes.size()), errors);
        errors.flush();
        if (FAILED(status)) {
            if (diagnostic.empty()) diagnostic = "DXIL validation failed (HRESULT " + std::to_string(status) + ")";
            return status;
        }
        auto* container = reinterpret_cast<hlsl::DxilContainerHeader*>(bytes.data());
        const auto* program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
        const auto version = program->ProgramVersion;
        if (hlsl::ShaderModel::IsPreReleaseShaderModel((version >> 4) & 0xf, version & 0xf)) {
            std::memcpy(container->Hash.Digest, hlsl::PreviewByPassHash.Digest, sizeof(container->Hash.Digest));
        } else {
            constexpr auto offset = offsetof(hlsl::DxilContainerHeader, Version);
            ComputeHashRetail(reinterpret_cast<const unsigned char*>(bytes.data()) + offset,
                container->ContainerSizeInBytes - static_cast<unsigned>(offset), container->Hash.Digest);
        }
        return S_OK;
    } catch (const std::bad_alloc&) {
        diagnostic = "DXIL validator allocation failed";
        return E_OUTOFMEMORY;
    } catch (const hlsl::Exception& error) {
        diagnostic = error.what();
        if (diagnostic.empty()) diagnostic = "DXIL validator failed (HRESULT " + std::to_string(error.hr) + ")";
        return error.hr;
    } catch (const std::exception& error) {
        diagnostic = error.what();
        return E_FAIL;
    } catch (...) {
        diagnostic = "DXIL validator failed";
        return E_FAIL;
    }
}
}
