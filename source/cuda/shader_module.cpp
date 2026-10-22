#include "backend.hpp"
#include "command_regions.hpp"
#include "shader_ptx.hpp"
#include "../cudavm/module_descriptor.hpp"
#include "../cudavm/shader_packet.hpp"
#include "../cudavm/spirv_artifact.hpp"
#include "../cudavm/dxil_artifact.hpp"
#include <cstdlib>
#include <array>
#include <cctype>
#include <fstream>
#include <iterator>
#include <list>
#include <limits>
#include <set>
#include <map>
#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace cudavm {
namespace {
struct shader_kernel : kernel {
    session& owner;
    cudavm::kernel_descriptor entry;
    std::vector<uint64_t> globals;
    cudavm::shader_artifact native;
    cudavm::shader_artifact dxil;
    std::vector<cudavm::dxil_variant> dxil_variants;
    struct variant {
        std::array<unsigned,4> key;
        std::shared_ptr<compute_pipeline> pipeline;
    };
    // Driver launches are serialized. Variants and transport belong to this
    // function and are released on module unload or context destruction.
    std::list<variant> variants;
    std::unique_ptr<buffer> arena, args;
    unsigned dynamic_shared_limit;
    int carveout = CU_SHAREDMEM_CARVEOUT_DEFAULT;
    const std::string& native_code(const launch_shape& shape) const {
        if (owner.kind() != cudavmBackendDirectX) return owner.kind() == cudavmBackendOpenCL ? native.opencl_code : native.code;
        for(const auto& variant:dxil_variants) if(std::equal(variant.block.begin(),variant.block.end(),shape.block)) return variant.code;
        static const std::string empty; return empty;
    }
    void validate_native_shape(const launch_shape& shape) const {
        if(owner.kind() == cudavmBackendDirectX) owner.validate_native_dxil_shape(shape);
        else owner.validate_native_spirv_shape(shape);
    }
    void require_native(const launch_shape& shape) const {
        const auto& bytes = native_code(shape);
        if (owner.kind() == cudavmBackendDirectX) {
            if (bytes.empty()) throw failure(CUDA_ERROR_NOT_SUPPORTED, "native DXIL lowering or block variant unavailable: " + (dxil.code.empty() ? dxil.diagnostic : "CUDA block shape has no emitted DXIL variant"));
            require(owner.supports_native_dxil(bytes), CUDA_ERROR_NOT_SUPPORTED, "native DXIL shader model or required device capabilities are unavailable on this adapter");
        } else {
            if (bytes.empty()) throw failure(CUDA_ERROR_NOT_SUPPORTED, "native SPIR-V lowering unavailable: " + (owner.kind() == cudavmBackendOpenCL ? native.opencl_diagnostic : native.diagnostic));
            require(owner.supports_native_spirv(cudavm::decode_spirv_words(bytes, nullptr, owner.kind() != cudavmBackendOpenCL)), CUDA_ERROR_NOT_SUPPORTED, "native SPIR-V environment, IL ingestion or required device capabilities are unavailable on this adapter");
        }
        validate_native_shape(shape);
    }
    std::shared_ptr<compute_pipeline> prepare(const launch_shape& shape) {
        std::array<unsigned,4> key{shape.block[0], shape.block[1], shape.block[2], shape.shared_bytes};
        auto found = std::find_if(variants.begin(), variants.end(), [&](const auto& v) { return v.key == key; });
        if (found != variants.end()) variants.splice(variants.begin(), variants, found);
        else {
            auto pipeline = owner.kind() == cudavmBackendDirectX ? owner.compile_native_dxil(native_code(shape), shape)
                : owner.compile_native_spirv(cudavm::decode_spirv_words(native_code(shape), owner.kind() == cudavmBackendOpenCL ? nullptr : shape.block, owner.kind() != cudavmBackendOpenCL), shape);
            variants.push_front({key, std::shared_ptr<compute_pipeline>(std::move(pipeline))});
            if (variants.size() > 16) variants.pop_back();
        }
        return variants.front().pipeline;
    }
    shader_kernel(session& s, const cudavm::kernel_descriptor& e, const std::vector<uint64_t>& g, const cudavm::shader_artifact& n,const cudavm::shader_artifact& d)
        : owner(s), entry(e), globals(g), native(n), dxil(d), dxil_variants(d.code.empty()?std::vector<cudavm::dxil_variant>{}:cudavm::decode_dxil_variants(d.code)),
          dynamic_shared_limit(std::min(32768u, s.shader_limits().shared_bytes) - e.shared_bytes) {}
    int attribute(CUfunction_attribute which) override {
        switch (which) {
        case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: return int(std::min(1024u, owner.native_block_limit()));
        case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: return int(entry.shared_bytes);
        case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: return int(entry.local_bytes);
        // Shader lowering uses registry global storage for CUDA constants and
        // exposes no NVIDIA physical registers, PTX ISA or SM binary version.
        case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: case CU_FUNC_ATTRIBUTE_NUM_REGS:
        case CU_FUNC_ATTRIBUTE_PTX_VERSION: case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
        case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
        case CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET:
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT:
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED:
        case CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: case CU_FUNC_ATTRIBUTE_DEVICE_NODE_UPDATE_SUPPORTED:
            return 0;
        case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: return int(dynamic_shared_limit);
        case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: return carveout;
        default: return kernel::attribute(which);
        }
    }
    void set_attribute(CUfunction_attribute which, int input) override {
        if (which == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
            require(input >= 0 && unsigned(input) <= std::min(32768u, owner.shader_limits().shared_bytes) - entry.shared_bytes, CUDA_ERROR_INVALID_VALUE, "dynamic shared memory exceeds shader transport capacity");
            dynamic_shared_limit = unsigned(input);
        } else if (which == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) {
            // A preference does not change the transport's shared-memory capacity.
            require(input >= CU_SHAREDMEM_CARVEOUT_DEFAULT && input <= 100, CUDA_ERROR_INVALID_VALUE, "invalid shared-memory carveout preference");
            carveout = input;
        } else kernel::set_attribute(which, input);
    }
    void parameter_info(size_t i,size_t& offset,size_t& size) override {
        require(i<entry.parameters.size(),CUDA_ERROR_INVALID_VALUE,"kernel parameter index is out of range");
        offset=entry.parameters[i].offset; size=entry.parameters[i].size;
    }
    std::vector<kernel_parameter> parameter_layout() override {
        std::vector<kernel_parameter> result;
        for (auto parameter : entry.parameters) result.push_back({parameter.offset, parameter.size});
        return result;
    }
    void validate_shape(const launch_shape& shape) {
        const auto& options = shape.options;
        if (options.access_policy) {
            validate_access_policy(*options.access_policy);
            require(!options.access_policy->bytes, CUDA_ERROR_NOT_SUPPORTED, "nonempty access policy windows require native resource information");
        }
        require(!options.cooperative && !options.cluster && !options.cluster_policy && !options.domain_map && !options.domain && !options.shared_memory_mode &&
            !options.preferred_cluster && !options.portable_cluster && !options.programmatic_serialization.value_or(false) &&
            !options.programmatic_event && !options.completion_event,
            CUDA_ERROR_NOT_SUPPORTED, "cooperative/cluster launch or memory synchronization domain is unavailable on this shader adapter");
        auto limits = owner.shader_limits();
        for (unsigned axis = 0; axis < 3; ++axis)
            require(shape.grid[axis] <= std::min(unsigned(std::numeric_limits<int>::max()), limits.grid[axis]) && shape.block[axis] <= limits.block[axis],
                CUDA_ERROR_INVALID_VALUE, "launch dimensions exceed shader dispatch limits");
        uint64_t threads=1;
        for(auto dimension:shape.block) {
            require(dimension && dimension<=1024 && threads<=std::min(1024u, limits.threads)/dimension,CUDA_ERROR_INVALID_VALUE,"block exceeds shader invocation limits");
            threads*=dimension;
        }
        require(shape.shared_bytes<=dynamic_shared_limit,CUDA_ERROR_INVALID_VALUE,"dynamic shared memory exceeds function limits");
        require(entry.local_bytes<=16384,CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,"shader local memory exceeds 16 KiB per invocation");
    }
    void validate_configuration(const launch_shape& shape) override {
        kernel::validate_configuration(shape); validate_shape(shape); require_native(shape);
    }
    void validate_launch(const launch_shape& shape, std::span<const std::byte> bytes, std::span<const allocation_binding> bindings) override {
        kernel::validate_launch(shape, bytes, bindings);
        for (auto parameter : entry.parameters) if (parameter.pointer) {
            uint64_t pointer = 0; std::memcpy(&pointer, bytes.data() + parameter.offset, sizeof(pointer));
            if (!pointer) continue;
            bool found = std::any_of(bindings.begin(), bindings.end(), [&](auto binding) {
                return pointer >= binding.address && (pointer - binding.address <= binding.native->size);
            });
            require(found, CUDA_ERROR_INVALID_VALUE, "shader pointer argument is outside the context's allocations");
        }
    }
    void launch(const launch_shape&,void**,void**) override { throw failure(CUDA_ERROR_INVALID_VALUE,"shader launch requires allocation bindings"); }
    void launch_bound(const launch_shape& shape,void** parameters,void** extra,std::span<const allocation_binding> bindings) override {
        wait_device(enqueue_launch(shape, parameters, extra, bindings, nullptr));
    }
    bool can_record_launch(std::span<const allocation_binding> bindings) const override {
        return owner.command_regions() && !owner.profile->enabled &&
            std::all_of(bindings.begin(), bindings.end(), [&](auto binding) {
                return !binding.surface_descriptor && owner.can_record_fill(*binding.native, 0, binding.native->size) &&
                    binding.native->size % 4 == 0 && binding.native->storage_offset() % 4 == 0;
            });
    }
    device_work enqueue_launch(const launch_shape& shape,void** parameters,void** extra,std::span<const allocation_binding> bindings, native_queue* queue) override {
        bool recording = queue && queue->recording();
        require(!recording || can_record_launch(bindings), CUDA_ERROR_NOT_SUPPORTED, "shader transport cannot be recorded");
        validate_shape(shape);
        size_t parameter_bytes=0;
        for(auto p:entry.parameters) parameter_bytes=std::max(parameter_bytes,size_t(p.offset)+p.size);
        const std::byte* packed=nullptr; size_t packed_size=0;
        if(extra) {
            bool ended=false; bool got_buffer=false,got_size=false;
            for(unsigned i=0;i<5;) {
                auto token=extra[i++]; if(token==CU_LAUNCH_PARAM_END) { ended=true;break; }
                require(i<5,CUDA_ERROR_INVALID_VALUE,"invalid launch extra list"); auto data=extra[i++];
                if(token==CU_LAUNCH_PARAM_BUFFER_POINTER && !got_buffer) {packed=static_cast<const std::byte*>(data);got_buffer=true;}
                else if(token==CU_LAUNCH_PARAM_BUFFER_SIZE && !got_size && data) {std::memcpy(&packed_size,data,sizeof(packed_size));got_size=true;}
                else throw failure(CUDA_ERROR_INVALID_VALUE,"invalid launch extra token");
            }
            require(ended&&got_buffer&&got_size&&packed_size>=parameter_bytes&&(!parameter_bytes||packed),CUDA_ERROR_INVALID_VALUE,"incomplete packed launch parameters");
        } else require(entry.parameters.empty()||parameters,CUDA_ERROR_INVALID_VALUE,"missing kernel parameters");
        struct mapping { allocation_binding binding; size_t offset; };
        require_native(shape);
        std::vector<mapping> mappings;
        size_t arena_size = 16;
        for (auto binding : bindings) {
            arena_size = (arena_size + 15) & ~size_t(15);
            require(arena_size < (1u << 30) && binding.native->size < (1u << 30) - arena_size, CUDA_ERROR_OUT_OF_MEMORY, "native address transport exceeds 1 GiB");
            mappings.push_back({binding, arena_size}); arena_size += binding.native->size;
        }
        arena_size = (arena_size + 15) & ~size_t(15);
        std::vector<std::byte> packet((32+parameter_bytes+15)&~size_t(15));
        std::memcpy(packet.data(),shape.grid,12);std::memcpy(packet.data()+12,shape.block,12);
        uint32_t table_offset = uint32_t(packet.size()), count = uint32_t(mappings.size());
        require(mappings.size() < ((1u << 30) - packet.size()) / sizeof(cudavm::packet_allocation), CUDA_ERROR_OUT_OF_MEMORY, "native allocation table exceeds transport limit");
        std::memcpy(packet.data() + 24, &table_offset, 4); std::memcpy(packet.data() + 28, &count, 4);
        packet.resize(packet.size() + mappings.size() * sizeof(cudavm::packet_allocation));
        for (size_t i = 0; i < mappings.size(); ++i) {
            auto m = mappings[i];
            cudavm::packet_allocation record{cudavm::split_packet_word64(m.binding.address), uint32_t(m.binding.native->size), uint32_t(m.offset)};
            std::memcpy(packet.data() + table_offset + i * sizeof(record), &record, sizeof(record));
        }
        auto globals_offset = packet.size(); packet.resize(globals_offset + globals.size() * 8);
        for (size_t i = 0; i < globals.size(); ++i) { auto address = cudavm::split_packet_word64(globals[i]); std::memcpy(packet.data() + globals_offset + i * 8, &address, 8); }
        for(size_t i=0;i<entry.parameters.size();++i) {
            auto p=entry.parameters[i]; auto* src=packed?packed+p.offset:static_cast<const std::byte*>(parameters[i]);
            require(src,CUDA_ERROR_INVALID_VALUE,"null kernel parameter storage");
            std::memcpy(packet.data()+32+p.offset,src,p.size);
            if(p.pointer) {
                require(p.size==8,CUDA_ERROR_INVALID_IMAGE,"shader pointer ABI requires 64-bit CUDA pointers");
                uint64_t pointer=0;std::memcpy(&pointer,src,8);
                if(pointer) {
                    bool found=false;
                    for(auto m:mappings) if(pointer>=m.binding.address && (pointer-m.binding.address<=m.binding.native->size)) {
                        found=true;break;
                    }
                    require(found,CUDA_ERROR_INVALID_VALUE,"kernel pointer is not a live allocation in the current context");
                }

            }
        }
        auto pipeline=prepare(shape);
        bool check_image_fault = std::any_of(bindings.begin(), bindings.end(), [](auto binding) { return binding.surface_descriptor; });
        // A recorded launch must not borrow the function's mutable transport.
        // Each region owns its arena and packet through every accepted replay.
        std::shared_ptr<buffer> recorded_arena, recorded_args;
        if (recording) { recorded_arena = owner.allocate(arena_size); recorded_args = owner.allocate(packet.size()); }
        else {
            if(!arena || arena->size<arena_size) arena=owner.allocate(arena_size);
            if(!args || args->size<packet.size()) args=owner.allocate(packet.size());
        }
        auto launch_arena = recording ? recorded_arena.get() : arena.get();
        auto launch_args = recording ? recorded_args.get() : args.get();
        // Native shaders use the arena ABI. Submit its gathers, dispatch and
        // writebacks in order on the same device queue, without a CPU wait
        // after each transfer. Byte staging and external/host allocations keep
        // the synchronous path, since those copies may involve another queue.
        bool queued_transport = (owner.kind()!=cudavmBackendOpenCL || queue)
            && std::all_of(bindings.begin(),bindings.end(),[&](auto binding) { return owner.can_copy_direct(*binding.native) && binding.native->size % 4 == 0 && binding.native->storage_offset() % 4 == 0; });
        if(queued_transport) {
            auto payload=std::make_shared<std::vector<std::byte>>(std::move(packet));
            auto zero=std::make_shared<std::array<std::byte,16>>();
            auto tasks=std::make_shared<device_group>();
            tasks->tasks.reserve(2 * mappings.size() + 3);
            // Retain host packet storage (OpenCL borrows it), pipelines and
            // every transfer fence until writeback completes, including errors.
            auto completion=std::make_shared<device_continuation>(tasks,[this,pipeline,payload,zero,recorded_arena,recorded_args,launch_arena,check_image_fault] {
                if (check_image_fault) {
                    uint32_t fault = 0; owner.read(*launch_arena, 8, std::as_writable_bytes(std::span(&fault, 1)));
                    require(!fault, CUDA_ERROR_ILLEGAL_ADDRESS, "surface access is outside the array bounds");
                }
            });
            if (recording) dynamic_cast<command_region_queue&>(*queue).resources.push_back(completion);
            tasks->tasks.push_back(owner.enqueue_write(*launch_arena,0,*zero,queue));
            for(size_t i=0;i<mappings.size();++i) {auto m=mappings[i];tasks->tasks.push_back(owner.enqueue_copy(*launch_arena,m.offset,*m.binding.native,0,m.binding.native->size,queue));}
            tasks->tasks.push_back(owner.enqueue_write(*launch_args,0,*payload,queue));
            tasks->tasks.push_back(owner.enqueue_dispatch_bound(*pipeline,shape,*launch_arena,*launch_args,{},queue));
            for(size_t i=0;i<mappings.size();++i) {auto m=mappings[i];tasks->tasks.push_back(owner.enqueue_copy(*m.binding.native,0,*launch_arena,m.offset,m.binding.native->size,queue));}
            return recording ? device_work{} : completion;
        }
        // Keep the reserved transport prefix initialized between launches.
        const std::array<std::byte,16> zero{};owner.write(*arena,0,zero);
        for(size_t i=0;i<mappings.size();++i) {auto m=mappings[i];owner.copy(*arena,m.offset,*m.binding.native,0,m.binding.native->size);}
        owner.write(*args,0,packet);
        // Context transport ordering prevents another command from changing
        // the shared heap before its complete writeback has been published.
        auto continuation = std::make_shared<device_continuation>(device_work{}, [this, mappings = std::move(mappings), pipeline,check_image_fault] {
            (void)pipeline; // Keep evicted pipeline variants alive through GPU completion.
            for(auto m:mappings) owner.copy(*m.binding.native,0,*arena,m.offset,m.binding.native->size);
            if (check_image_fault) {
                uint32_t fault = 0; owner.read(*arena, 8, std::as_writable_bytes(std::span(&fault, 1)));
                require(!fault, CUDA_ERROR_ILLEGAL_ADDRESS, "surface access is outside the array bounds");
            }
        });
        continuation->task = owner.enqueue_dispatch_bound(*pipeline,shape,*arena,*args,{},queue);
        return continuation;
    }
};
struct shader_executable : executable {
    std::vector<std::string> kernel_names() const override {
        std::vector<std::string> result; for (auto& entry : entries) result.push_back(entry.symbol); std::sort(result.begin(), result.end()); return result;
    }
    session& owner; std::vector<cudavm::kernel_descriptor> entries; bool live=true;
    std::vector<cudavm::global_descriptor> definitions; std::vector<uint64_t> addresses;
    std::map<std::string, cudavm::shader_artifact> native_entries;
    std::map<std::string, cudavm::shader_artifact> dxil_entries;
    shader_executable(session& s,cudavm::module_descriptor image):owner(s),entries(std::move(image.entries)),definitions(std::move(image.globals)) {
        std::set<std::string> symbols;
        for(const auto& entry:entries) {
            require(!entry.symbol.empty() && entry.symbol.find('\0')==entry.symbol.npos && symbols.insert(entry.symbol).second,CUDA_ERROR_INVALID_IMAGE,"invalid or duplicate shader entry");
            uint64_t end=0;
            for(auto p:entry.parameters) {
                require(p.size && p.alignment && !(p.alignment&(p.alignment-1)) && p.offset%p.alignment==0 && p.offset>=end && p.pointer<=1 && (!p.pointer || p.size==8),CUDA_ERROR_INVALID_IMAGE,"invalid shader parameter layout");
                end=uint64_t(p.offset)+p.size;require(end<=65536,CUDA_ERROR_INVALID_IMAGE,"shader parameter packet exceeds 64 KiB");
            }
            require(entry.shared_bytes<=std::min(32768u, owner.shader_limits().shared_bytes) && entry.local_bytes<=16384,CUDA_ERROR_INVALID_IMAGE,"shader static storage exceeds transport limits");
        }
        for(const auto& g:definitions) {
            require(!g.symbol.empty() && g.symbol.find('\0')==g.symbol.npos && symbols.insert(g.symbol).second,CUDA_ERROR_INVALID_IMAGE,"invalid or duplicate shader global");
            require(g.alignment && g.alignment<=256 && !(g.alignment&(g.alignment-1)) && !g.initializer.empty() && g.initializer.size()<(1u<<30),CUDA_ERROR_INVALID_IMAGE,"invalid shader global storage");
            uint64_t end=0;
            for(auto r:g.relocations) {
                require(r.target<definitions.size() && r.offset>=end && uint64_t(r.offset)+8<=g.initializer.size() && r.addend<=definitions[r.target].initializer.size(),CUDA_ERROR_INVALID_IMAGE,"invalid shader global relocation");
                end=uint64_t(r.offset)+8;
            }
        }
    }
    std::vector<std::string> global_names() const override { std::vector<std::string> result; for(const auto& g:definitions)result.push_back(g.symbol); return result; }
    std::unique_ptr<buffer> find_global(const char* name) override {
        require(live,CUDA_ERROR_INVALID_HANDLE,"unloaded shader module");
        for(const auto& g:definitions) if(g.symbol==name) return owner.allocate(g.initializer.size());
        throw failure(CUDA_ERROR_NOT_FOUND,"global symbol is absent from shader module");
    }
    void bind_globals(std::span<const allocation_binding> bindings) override {
        require(bindings.size()==definitions.size(),CUDA_ERROR_INVALID_IMAGE,"incomplete shader global bindings");
        for(auto b:bindings) addresses.push_back(b.address);
        for(size_t i=0;i<definitions.size();++i) {
            auto bytes=definitions[i].initializer;
            for(auto r:definitions[i].relocations) { auto address=cudavm::split_packet_word64(addresses[r.target]+r.addend); std::memcpy(bytes.data()+r.offset,&address,8); }
            owner.write(*bindings[i].native,0,{reinterpret_cast<const std::byte*>(bytes.data()),bytes.size()});
        }
    }
    std::unique_ptr<kernel> find(const char* name) override {
        require(live,CUDA_ERROR_INVALID_HANDLE,"unloaded shader module");
        for(const auto& e:entries) if(e.symbol==name) return std::make_unique<shader_kernel>(owner,e,addresses,
            native_entries.contains(e.symbol)?native_entries.at(e.symbol):cudavm::shader_artifact{e.symbol,{},"module has no native SPIR-V artifact",{},"module has no OpenCL SPIR-V artifact"},
            dxil_entries.contains(e.symbol)?dxil_entries.at(e.symbol):cudavm::shader_artifact{e.symbol,{},"module has no native DXIL artifact",{}, {}});
        throw failure(CUDA_ERROR_NOT_FOUND,"kernel symbol is absent from shader module");
    }
    void unload() override { live=false;entries.clear();definitions.clear();addresses.clear();native_entries.clear();dxil_entries.clear(); }
};
}
std::shared_ptr<executable> session::load_image(const void* image,unsigned count,CUjit_option* options,void** values) {
    require(image,CUDA_ERROR_INVALID_VALUE,"null module image");
    for(unsigned i=0;i<count;++i) {
        require(options && values,CUDA_ERROR_INVALID_VALUE,"missing JIT options");
        require(options[i]==CU_JIT_OPTIMIZATION_LEVEL,CUDA_ERROR_NOT_SUPPORTED,"shader JIT option is unavailable");
        require(reinterpret_cast<uintptr_t>(values[i])<=4,CUDA_ERROR_INVALID_VALUE,"invalid JIT optimization hint");
    }
    auto text=std::string_view(static_cast<const char*>(image));
    std::string compiled;
    if (!cudavm::has_module_descriptor(text) && text.find(".version") != text.npos) {
        compiled = compile_shader_ptx(text, kind()); text = compiled;
    }
    try {
        auto result = std::make_shared<shader_executable>(*this,cudavm::decode_module_descriptor(text));
        for (auto& entry : cudavm::decode_spirv_artifacts(text)) {
            require(std::any_of(result->entries.begin(), result->entries.end(), [&](const auto& e) { return e.symbol == entry.symbol; }) && !result->native_entries.contains(entry.symbol), CUDA_ERROR_INVALID_IMAGE, "native SPIR-V entry has no unique matching kernel");
            require(std::any_of(result->entries.begin(), result->entries.end(), [&](const auto& e) { return e.symbol == entry.symbol; }), CUDA_ERROR_INVALID_IMAGE, "native SPIR-V requires allocation-table shader ABI");
            if (!entry.code.empty()) cudavm::decode_spirv_words(entry.code);
            if (!entry.opencl_code.empty()) cudavm::decode_spirv_words(entry.opencl_code, nullptr, false);
            auto symbol = entry.symbol; result->native_entries.emplace(std::move(symbol), std::move(entry));
        }
        result->data_views = decode_data_views(text);
        for(auto& entry:cudavm::decode_dxil_artifacts(text)) {
            require(std::any_of(result->entries.begin(),result->entries.end(),[&](const auto& e){ return e.symbol==entry.symbol; }) && !result->dxil_entries.contains(entry.symbol),
                CUDA_ERROR_INVALID_IMAGE,"native DXIL entry has no unique allocation-table kernel");
            if(!entry.code.empty()) cudavm::decode_dxil_variants(entry.code);
            auto symbol=entry.symbol; result->dxil_entries.emplace(std::move(symbol),std::move(entry));
        }
        return result;
    }
    catch(const failure&) { throw; }
    catch(const std::exception& e) { throw failure(CUDA_ERROR_INVALID_IMAGE,e.what()); }
}
std::shared_ptr<executable> session::load_file(const char* path) {
    require(path,CUDA_ERROR_INVALID_VALUE,"null module filename");
    std::ifstream file(path,std::ios::binary);require(bool(file),CUDA_ERROR_FILE_NOT_FOUND,"module file cannot be opened");
    std::string image((std::istreambuf_iterator<char>(file)),{});return load_image(image.c_str(),0,nullptr,nullptr);
}
}
