#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_ENABLE_BETA_EXTENSIONS
#include <CL/cl.h>
#include <CL/cl_ext.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include "dynamic_library.hpp"
#include "device_resources.hpp"
#if defined(_WIN32)
#include <CL/cl_d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#endif
namespace cudavm {
namespace {
void check(cl_int result, const char* operation) {
    if (result == CL_SUCCESS) return;
    CUresult error = (result == CL_MEM_OBJECT_ALLOCATION_FAILURE || result == CL_OUT_OF_HOST_MEMORY || result == CL_OUT_OF_RESOURCES) ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    throw failure(error, std::string(operation) + " failed: " + std::to_string(result));
}
struct opencl_api {
#if defined(_WIN32)
    dynamic_library driver{{"OpenCL.dll"}};
#else
    dynamic_library driver{{"libOpenCL.so.1", "libOpenCL.so"}};
#endif
#define ENTRY(name) decltype(&name) name = driver.get<decltype(name)>(#name)
    ENTRY(clGetPlatformIDs); ENTRY(clGetDeviceIDs); ENTRY(clGetDeviceInfo);
    ENTRY(clCreateContext); ENTRY(clReleaseContext); ENTRY(clCreateCommandQueue); ENTRY(clReleaseCommandQueue);
    ENTRY(clCreateBuffer); ENTRY(clReleaseMemObject); ENTRY(clEnqueueWriteBuffer); ENTRY(clEnqueueReadBuffer);
    ENTRY(clEnqueueCopyBuffer); ENTRY(clEnqueueFillBuffer); ENTRY(clFinish);
    ENTRY(clCreateProgramWithSource);ENTRY(clBuildProgram);ENTRY(clGetProgramBuildInfo);ENTRY(clReleaseProgram);
    ENTRY(clCreateKernel);ENTRY(clReleaseKernel);ENTRY(clSetKernelArg);ENTRY(clEnqueueNDRangeKernel); ENTRY(clGetKernelWorkGroupInfo);
    ENTRY(clGetExtensionFunctionAddressForPlatform); ENTRY(clCreateSubBuffer); ENTRY(clFlush);
    ENTRY(clGetEventInfo); ENTRY(clReleaseEvent); ENTRY(clWaitForEvents); ENTRY(clGetEventProfilingInfo); ENTRY(clEnqueueMarkerWithWaitList);
#undef ENTRY
    decltype(&::clCreateProgramWithIL) create_program_il = reinterpret_cast<decltype(&::clCreateProgramWithIL)>(driver.find("clCreateProgramWithIL"));
};
shader_resource_limits opencl_shader_limits(opencl_api& api, cl_device_id device) {
    size_t maximum = 0; cl_uint dimensions = 0; cl_ulong shared = 0;
    check(api.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maximum), &maximum, nullptr), "CL_DEVICE_MAX_WORK_GROUP_SIZE");
    check(api.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dimensions), &dimensions, nullptr), "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    check(api.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(shared), &shared, nullptr), "CL_DEVICE_LOCAL_MEM_SIZE");
    require(dimensions >= 3 && dimensions <= 1024, CUDA_ERROR_NOT_SUPPORTED, "OpenCL shader profile requires three work-item dimensions");
    std::vector<size_t> sizes(dimensions);
    check(api.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizes.size() * sizeof(size_t), sizes.data(), nullptr), "CL_DEVICE_MAX_WORK_ITEM_SIZES");
    shader_resource_limits result; result.threads = unsigned(std::min(maximum, size_t(1024)));
    for (unsigned axis = 0; axis < 3; ++axis) result.block[axis] = unsigned(std::min(sizes[axis], size_t(1024)));
    // Dispatch products and group IDs use the compiler's unsigned 32-bit ABI.
    result.grid.fill(unsigned(std::numeric_limits<int>::max()));
    result.shared_bytes = unsigned(std::min(shared, cl_ulong(32768))); return result;
}
struct opencl_context {
    std::shared_ptr<opencl_api> api;
    cl_device_id device;
    uint32_t native_il_version = 0;
    bool native_il = false;
    bool native_int64_atomics = false;
    decltype(&::clCreateProgramWithIL) create_program_il = nullptr;
#include "opencl_interop_api.inl"
    opencl_context(std::shared_ptr<opencl_api> input, cl_device_id device) : api(std::move(input)), device(device) {
        cl_int status = CL_SUCCESS;
        context = api->clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        check(status, "clCreateContext");
        queue = api->clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        if (status != CL_SUCCESS) { api->clReleaseContext(context); context = nullptr; check(status, "clCreateCommandQueue"); }
        size_t bytes = 0; cl_uint address_bits = 0; cl_bool little = CL_FALSE;
        api->clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits), &address_bits, nullptr);
        api->clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(little), &little, nullptr);
        if (api->clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, 0, nullptr, &bytes) == CL_SUCCESS && bytes > 1 && bytes < 65536) {
            std::string versions(bytes, '\0');
            if (api->clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, bytes, versions.data(), nullptr) == CL_SUCCESS) {
                for (unsigned minor = 0; minor <= 6; ++minor)
                    if (versions.find("SPIR-V_1." + std::to_string(minor)) != versions.npos) native_il_version = 0x10000u + (minor << 8);
            }
        }
        create_program_il = api->create_program_il;
        if (!create_program_il) {
            cl_platform_id platform = nullptr;
            if (api->clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr) == CL_SUCCESS)
                create_program_il = reinterpret_cast<decltype(create_program_il)>(api->clGetExtensionFunctionAddressForPlatform(platform,"clCreateProgramWithILKHR"));
        }
        native_il = create_program_il && native_il_version && address_bits == 64 && little;
        bytes=0;
        if(api->clGetDeviceInfo(device,CL_DEVICE_EXTENSIONS,0,nullptr,&bytes)==CL_SUCCESS && bytes && bytes<65536) {
            std::string extensions(bytes,'\0');
            if(api->clGetDeviceInfo(device,CL_DEVICE_EXTENSIONS,bytes,extensions.data(),nullptr)==CL_SUCCESS) {
                extensions=std::string(" ")+extensions.c_str()+" ";
                native_int64_atomics=extensions.find(" cl_khr_int64_base_atomics ")!=extensions.npos && extensions.find(" cl_khr_int64_extended_atomics ")!=extensions.npos;
            }
        }
    }
    opencl_context(const opencl_context&) = delete;
    opencl_context& operator=(const opencl_context&) = delete;
    opencl_context(opencl_context&&) = delete;
    ~opencl_context() {
        if (queue) { api->clFinish(queue); api->clReleaseCommandQueue(queue); }
        if (context) api->clReleaseContext(context);
    }
    ::cl_context context = nullptr;
    cl_command_queue queue = nullptr;
};
struct opencl_buffer : buffer {
    std::shared_ptr<opencl_context> owner;
    cl_mem value = nullptr;
    std::shared_ptr<external_memory> imported;
    opencl_buffer(std::shared_ptr<opencl_context> input, size_t bytes, cl_mem memory, std::shared_ptr<external_memory> external)
        : buffer(bytes), owner(std::move(input)), value(memory), imported(std::move(external)) {}
    opencl_buffer(std::shared_ptr<opencl_context> context, size_t bytes) : buffer(bytes), owner(std::move(context)) {
        cl_int status = CL_SUCCESS;
        value = owner->api->clCreateBuffer(owner->context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
        check(status, "clCreateBuffer");
    }
    ~opencl_buffer() { if (value) owner->api->clReleaseMemObject(value); }
    size_t physical_size() const override { return imported ? 0 : size; }
};
struct opencl_queue : native_queue {
    std::shared_ptr<opencl_context> owner;
    cl_command_queue value = nullptr;
    explicit opencl_queue(std::shared_ptr<opencl_context> input) : owner(std::move(input)) {
        cl_int status = 0; value = owner->api->clCreateCommandQueue(owner->context, owner->device, CL_QUEUE_PROFILING_ENABLE, &status); check(status, "clCreateCommandQueue");
    }
    ~opencl_queue() { if (value) { owner->api->clFinish(value); owner->api->clReleaseCommandQueue(value); } }
};
struct opencl_task : device_task {
    std::shared_ptr<opencl_context> owner;
    cl_event value = nullptr;
    std::shared_ptr<device_timestamp> timestamp;
    std::shared_ptr<compute_profile> marker_profile,dispatch_profile;
    bool measured = false;
    explicit opencl_task(std::shared_ptr<opencl_context> input) : owner(std::move(input)) {}
    bool query() override {
        cl_int status = 0;
        check(owner->api->clGetEventInfo(value, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr), "clGetEventInfo");
        require(status >= 0, CUDA_ERROR_UNKNOWN, "OpenCL asynchronous command failed");
        if(status==CL_COMPLETE && timestamp) {
            cl_ulong ticks{};check(owner->api->clGetEventProfilingInfo(value,CL_PROFILING_COMMAND_END,sizeof(ticks),&ticks,nullptr),"clGetEventProfilingInfo");
            timestamp->ticks=ticks;timestamp->nanoseconds_per_tick=1;
            if(marker_profile) marker_profile->snapshot(*timestamp);
        }
        if(status==CL_COMPLETE && dispatch_profile && !measured) {
            cl_ulong begin{},end{};
            check(owner->api->clGetEventProfilingInfo(value,CL_PROFILING_COMMAND_START,sizeof(begin),&begin,nullptr),"clGetEventProfilingInfo start");
            check(owner->api->clGetEventProfilingInfo(value,CL_PROFILING_COMMAND_END,sizeof(end),&end,nullptr),"clGetEventProfilingInfo end");
            dispatch_profile->nanoseconds.fetch_add(end-begin);measured=true;
        }
        return status == CL_COMPLETE;
    }
    ~opencl_task() { if (value) { owner->api->clWaitForEvents(1, &value); owner->api->clReleaseEvent(value); } }
};
#include "opencl_interop.inl"
struct opencl_pipeline : compute_pipeline {
    std::shared_ptr<opencl_context> owner;
    cl_program program=nullptr;
    cl_kernel kernel=nullptr;
    explicit opencl_pipeline(std::shared_ptr<opencl_context> context) : owner(std::move(context)) {}
    ~opencl_pipeline() {
        if(kernel) owner->api->clReleaseKernel(kernel);
        if(program) owner->api->clReleaseProgram(program);
    }
};
struct opencl_session : session {
    std::shared_ptr<opencl_context> owner;
    explicit opencl_session(std::shared_ptr<opencl_context> input) : owner(std::move(input)) {}
    interop_capabilities interop() override { owner->negotiate_interop(); return owner->caps; }
    std::shared_ptr<external_memory> import_memory(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d, const cudavmExternalMemoryMetadata* = nullptr) override {
        require(interop().memory & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "OpenCL external memory extensions are unavailable");
        return std::make_shared<opencl_memory>(owner, d);
    }
    std::shared_ptr<external_semaphore> import_semaphore(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d) override {
        require(interop().semaphore & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "OpenCL external semaphore extensions are unavailable");
        return std::make_shared<opencl_semaphore>(owner, d);
    }
    std::unique_ptr<buffer> allocate(size_t size) override { return std::make_unique<opencl_buffer>(owner, size); }
    std::shared_ptr<native_queue> create_queue(int) override { return std::make_shared<opencl_queue>(owner); }
    device_work enqueue_timestamp(std::shared_ptr<device_timestamp> timestamp,native_queue* queue) override {
        auto task=std::make_shared<opencl_task>(owner);task->timestamp=std::move(timestamp);task->marker_profile=profile;
        auto lane=queue ? static_cast<opencl_queue*>(queue)->value : owner->queue;
        check(owner->api->clEnqueueMarkerWithWaitList(lane,0,nullptr,&task->value),"clEnqueueMarkerWithWaitList");
        check(owner->api->clFlush(lane),"clFlush");return task;
    }
    template<class F> device_work enqueue(native_queue* queue, F operation) {
        auto task = std::make_shared<opencl_task>(owner); auto value = static_cast<opencl_queue*>(queue)->value;
        operation(value, &task->value);
        check(owner->api->clFlush(value), "clFlush"); return task;
    }
    device_work enqueue_write(buffer& value, size_t offset, std::span<const std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_write(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address() || static_cast<opencl_buffer&>(value).imported) return session::enqueue_write(value, offset, bytes, queue);
        return enqueue(queue, [&](cl_command_queue q, cl_event* event) { check(owner->api->clEnqueueWriteBuffer(q, static_cast<opencl_buffer&>(value).value, CL_FALSE, offset, bytes.size(), bytes.data(), 0, nullptr, event), "clEnqueueWriteBuffer"); });
    }
    device_work enqueue_read(buffer& value, size_t offset, std::span<std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_read(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address() || static_cast<opencl_buffer&>(value).imported) return session::enqueue_read(value, offset, bytes, queue);
        return enqueue(queue, [&](cl_command_queue q, cl_event* event) { check(owner->api->clEnqueueReadBuffer(q, static_cast<opencl_buffer&>(value).value, CL_FALSE, offset, bytes.size(), bytes.data(), 0, nullptr, event), "clEnqueueReadBuffer"); });
    }
    device_work enqueue_copy(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes, native_queue* queue) override {
        if (&to.storage() != &to || &from.storage() != &from) return enqueue_copy(to.storage(), destination + to.storage_offset(), from.storage(), source + from.storage_offset(), bytes, queue);
        if (to.host_address() || from.host_address() || static_cast<opencl_buffer&>(to).imported || static_cast<opencl_buffer&>(from).imported)
            return session::enqueue_copy(to, destination, from, source, bytes, queue);
        return enqueue(queue, [&](cl_command_queue q, cl_event* event) { check(owner->api->clEnqueueCopyBuffer(q, static_cast<opencl_buffer&>(from).value, static_cast<opencl_buffer&>(to).value, source, destination, bytes, 0, nullptr, event), "clEnqueueCopyBuffer"); });
    }
    device_work enqueue_fill(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_fill(value.storage(), offset + value.storage_offset(), pattern, bytes, queue);
        if (value.host_address() || static_cast<opencl_buffer&>(value).imported) return session::enqueue_fill(value, offset, pattern, bytes, queue);
        return enqueue(queue, [&](cl_command_queue q, cl_event* event) { check(owner->api->clEnqueueFillBuffer(q, static_cast<opencl_buffer&>(value).value, pattern.data(), pattern.size(), offset, bytes, 0, nullptr, event), "clEnqueueFillBuffer"); });
    }
    void write_device(buffer& value, size_t offset, std::span<const std::byte> data) override {
        if (&value.storage() != &value) { write_device(value.storage(), offset + value.storage_offset(), data); return; }
        opencl_external_access access(static_cast<opencl_buffer&>(value));
        check(owner->api->clEnqueueWriteBuffer(owner->queue, static_cast<opencl_buffer&>(value).value, CL_TRUE, offset, data.size(), data.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    }
    void read_device(buffer& value, size_t offset, std::span<std::byte> data) override {
        if (&value.storage() != &value) { read_device(value.storage(), offset + value.storage_offset(), data); return; }
        opencl_external_access access(static_cast<opencl_buffer&>(value));
        check(owner->api->clEnqueueReadBuffer(owner->queue, static_cast<opencl_buffer&>(value).value, CL_TRUE, offset, data.size(), data.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    }
    void copy_device(buffer& to, size_t to_offset, buffer& from, size_t from_offset, size_t size) override {
        if (&to.storage() != &to || &from.storage() != &from) { copy_device(to.storage(), to_offset + to.storage_offset(), from.storage(), from_offset + from.storage_offset(), size); return; }
        opencl_external_access source(static_cast<opencl_buffer&>(from));
        opencl_external_access destination(static_cast<opencl_buffer&>(to), &to != &from && (!source.buffer.imported || source.buffer.imported != static_cast<opencl_buffer&>(to).imported));
        check(owner->api->clEnqueueCopyBuffer(owner->queue, static_cast<opencl_buffer&>(from).value, static_cast<opencl_buffer&>(to).value, from_offset, to_offset, size, 0, nullptr, nullptr), "clEnqueueCopyBuffer");
        synchronize();
    }
    void fill_device(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t size) override {
        if (&value.storage() != &value) { fill_device(value.storage(), offset + value.storage_offset(), pattern, size); return; }
        opencl_external_access access(static_cast<opencl_buffer&>(value));
        check(owner->api->clEnqueueFillBuffer(owner->queue, static_cast<opencl_buffer&>(value).value, pattern.data(), pattern.size(), offset, size, 0, nullptr, nullptr), "clEnqueueFillBuffer");
        synchronize();
    }
    void synchronize() override { check(owner->api->clFinish(owner->queue), "clFinish"); }
    cudavmBackend kind() const override { return cudavmBackendOpenCL; }
    unsigned direct_binding_limit() const override { return 6; }
    bool can_copy_direct(const buffer& value) const override {
        auto native = dynamic_cast<const opencl_buffer*>(&value.storage());
        return native && native->owner == owner && !native->imported;
    }
    bool can_bind_direct(const buffer& value) const override {
        auto native=dynamic_cast<const opencl_buffer*>(&value.storage());
        return native && native->owner==owner && !native->imported && value.size%4==0 && value.storage_offset()%4==0;
    }
    bool supports_native_spirv(std::span<const uint32_t> code) const override {
        if (!owner->native_il || code.size() < 5 || code[1] > owner->native_il_version) return false;
        for (size_t i = 5; i < code.size();) {
            auto count = code[i] >> 16, op = code[i] & 0xffffu;
            if (!count || count > code.size() - i) return false;
            if (op == 17) {
                if(count!=2) return false;
                if(code[i+1]==10) {
                    cl_device_fp_config fp64=0;
                    if(owner->api->clGetDeviceInfo(owner->device,CL_DEVICE_DOUBLE_FP_CONFIG,sizeof(fp64),&fp64,nullptr)!=CL_SUCCESS || !fp64) return false;
                } else if(code[i+1]==12) {if(!owner->native_int64_atomics)return false;}
                else if(code[i+1]!=4 && code[i+1]!=6 && code[i+1]!=11 && code[i+1]!=39) return false;
            }
            i += count;
        }
        return true;
    }
    unsigned native_block_limit() const override {
        size_t maximum = 0;
        check(owner->api->clGetDeviceInfo(owner->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maximum), &maximum, nullptr), "CL_DEVICE_MAX_WORK_GROUP_SIZE");
        return unsigned(std::min(maximum, size_t(std::numeric_limits<unsigned>::max())));
    }
    shader_resource_limits shader_limits() const override { return opencl_shader_limits(*owner->api, owner->device); }
    shader_device_resources execution_resources() const override {
        cl_uint units = 0; cl_ulong shared = 0;
        check(owner->api->clGetDeviceInfo(owner->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr), "CL_DEVICE_MAX_COMPUTE_UNITS");
        check(owner->api->clGetDeviceInfo(owner->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(shared), &shared, nullptr), "CL_DEVICE_LOCAL_MEM_SIZE");
        shader_device_resources result; result.compute_units = units; result.shared_bytes = shared; return result;
    }
    void validate_native_spirv_shape(const launch_shape& shape) const override {
        size_t maximum = 0; cl_uint dimensions = 0;
        check(owner->api->clGetDeviceInfo(owner->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maximum), &maximum, nullptr), "CL_DEVICE_MAX_WORK_GROUP_SIZE");
        check(owner->api->clGetDeviceInfo(owner->device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dimensions), &dimensions, nullptr), "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
        require(dimensions >= 3 && dimensions <= 1024, CUDA_ERROR_NOT_SUPPORTED, "OpenCL native profile requires three work-item dimensions");
        std::vector<size_t> sizes(dimensions);
        check(owner->api->clGetDeviceInfo(owner->device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizes.size() * sizeof(size_t), sizes.data(), nullptr), "CL_DEVICE_MAX_WORK_ITEM_SIZES");
        uint64_t threads = 1;
        for (unsigned axis = 0; axis < 3; ++axis) { require(shape.block[axis] <= sizes[axis], CUDA_ERROR_INVALID_VALUE, "native SPIR-V block exceeds OpenCL work-item limits"); threads *= shape.block[axis]; }
        require(threads <= maximum, CUDA_ERROR_INVALID_VALUE, "native SPIR-V block exceeds OpenCL workgroup limits");
    }
    std::unique_ptr<compute_pipeline> compile_native_spirv(std::span<const uint32_t> code, const launch_shape& shape) override {
        require(supports_native_spirv(code), CUDA_ERROR_NOT_SUPPORTED, "OpenCL SPIR-V IL ingestion is unavailable on this device");
        validate_native_spirv_shape(shape);
        auto compiled = std::make_unique<opencl_pipeline>(owner); compiled->direct_buffers=packet_direct_bindings; cl_int status = CL_SUCCESS;
        compiled->program = owner->create_program_il(owner->context, code.data(), code.size() * 4, &status); check(status,"clCreateProgramWithIL");
        status = owner->api->clBuildProgram(compiled->program, 1, &owner->device, "", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t bytes = 0; owner->api->clGetProgramBuildInfo(compiled->program,owner->device,CL_PROGRAM_BUILD_LOG,0,nullptr,&bytes);
            std::string log(std::max(size_t(1),bytes),'\0'); owner->api->clGetProgramBuildInfo(compiled->program,owner->device,CL_PROGRAM_BUILD_LOG,log.size(),log.data(),nullptr);
            throw failure(CUDA_ERROR_INVALID_IMAGE,"OpenCL native SPIR-V build failed: " + log);
        }
        compiled->kernel = owner->api->clCreateKernel(compiled->program, "main_kernel", &status); check(status,"clCreateKernel native SPIR-V");
        size_t kernel_threads = 0, preferred = 0; cl_ulong local_bytes = 0, private_bytes = 0, local_limit = 0;
        check(owner->api->clGetKernelWorkGroupInfo(compiled->kernel, owner->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_threads), &kernel_threads, nullptr), "CL_KERNEL_WORK_GROUP_SIZE");
        check(owner->api->clGetKernelWorkGroupInfo(compiled->kernel, owner->device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(local_bytes), &local_bytes, nullptr), "CL_KERNEL_LOCAL_MEM_SIZE");
        check(owner->api->clGetKernelWorkGroupInfo(compiled->kernel, owner->device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(private_bytes), &private_bytes, nullptr), "CL_KERNEL_PRIVATE_MEM_SIZE");
        check(owner->api->clGetKernelWorkGroupInfo(compiled->kernel, owner->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(preferred), &preferred, nullptr), "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
        compiled->resources.max_threads = kernel_threads; compiled->resources.preferred_thread_multiple = preferred;
        compiled->resources.shared_bytes = local_bytes; compiled->resources.private_bytes = private_bytes;
        check(owner->api->clGetDeviceInfo(owner->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_limit), &local_limit, nullptr), "CL_DEVICE_LOCAL_MEM_SIZE");
        require(local_bytes <= local_limit, CUDA_ERROR_NOT_SUPPORTED, "native SPIR-V static shared memory exceeds OpenCL device storage");
        require(uint64_t(shape.block[0]) * shape.block[1] * shape.block[2] <= kernel_threads, CUDA_ERROR_INVALID_VALUE, "native SPIR-V block exceeds compiled OpenCL kernel limits");
        return compiled;
    }
    void dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters) override {
        wait_device(enqueue_dispatch(pipeline, shape, memory, parameters, nullptr));
    }
    device_work enqueue_dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters, native_queue* queue) override {
        return enqueue_dispatch_bound(pipeline,shape,memory,parameters,{},queue);
    }
    device_work enqueue_dispatch_bound(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters,std::span<buffer* const> bindings,native_queue* queue) override {
        std::array<buffer*,packet_direct_bindings> unused; unused.fill(&memory);
        if (bindings.empty()) bindings = unused;
        require(bindings.size()==pipeline.direct_buffers,CUDA_ERROR_INVALID_VALUE,"OpenCL direct binding count mismatch");
        auto& api=*owner->api;auto kernel=static_cast<opencl_pipeline&>(pipeline).kernel;
        auto mem=static_cast<opencl_buffer&>(memory).value,args=static_cast<opencl_buffer&>(parameters).value;
        check(api.clSetKernelArg(kernel,0,sizeof(mem),&mem),"clSetKernelArg");check(api.clSetKernelArg(kernel,1,sizeof(args),&args),"clSetKernelArg");
        for(unsigned i=0;i<bindings.size();++i) {
            require(can_bind_direct(*bindings[i]),CUDA_ERROR_INVALID_VALUE,"incompatible OpenCL direct buffer");
            auto resource=static_cast<opencl_buffer&>(bindings[i]->storage()).value;
            check(api.clSetKernelArg(kernel,i+2,sizeof(resource),&resource),"clSetKernelArg direct buffer");
        }
        size_t global[3],local[3];for(unsigned i=0;i<3;++i) {local[i]=shape.block[i];global[i]=size_t(shape.grid[i])*local[i];}
        auto task = std::make_shared<opencl_task>(owner);
        auto lane = queue ? static_cast<opencl_queue*>(queue)->value : owner->queue;
        if(profile->enabled) task->dispatch_profile=profile;
        check(api.clEnqueueNDRangeKernel(lane,kernel,3,nullptr,global,local,0,nullptr,&task->value),"clEnqueueNDRangeKernel");
        check(api.clFlush(lane), "clFlush"); return task;
    }
};
struct opencl_backend : shader_backend {
    shader_resource_limits shader_limits(int ordinal) override { return opencl_shader_limits(*api, native_devices.at(ordinal)); }
    std::shared_ptr<opencl_api> api = std::make_shared<opencl_api>();
    std::vector<cl_device_id> native_devices;
    device_resources<opencl_context> resources;
    void device_extension(int ordinal, const char* name) {
        size_t bytes = 0; auto device = native_devices.at(ordinal);
        check(api->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &bytes), "clGetDeviceInfo");
        std::string extensions(bytes, '\0'); check(api->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, bytes, extensions.data(), nullptr), "clGetDeviceInfo");
        auto text = " " + extensions.substr(0, extensions.find('\0')) + " ";
        require(text.find(std::string(" ") + name + " ") != std::string::npos, CUDA_ERROR_NOT_SUPPORTED, "OpenCL device identity extension is unavailable");
    }
    std::array<char, 16> device_uuid(int ordinal, bool) override {
        device_extension(ordinal, "cl_khr_device_uuid"); std::array<char, 16> result;
        check(api->clGetDeviceInfo(native_devices.at(ordinal), CL_DEVICE_UUID_KHR, result.size(), result.data(), nullptr), "CL_DEVICE_UUID_KHR"); return result;
    }
    device_luid_info device_luid(int ordinal) override {
        try { device_extension(ordinal, "cl_khr_device_uuid"); }
        catch (const failure& error) {
            if (error.code != CUDA_ERROR_NOT_SUPPORTED) throw;
            return device_d3d11_luid(ordinal);
        }
        cl_bool valid = CL_FALSE; auto device = native_devices.at(ordinal);
        check(api->clGetDeviceInfo(device, CL_DEVICE_LUID_VALID_KHR, sizeof(valid), &valid, nullptr), "CL_DEVICE_LUID_VALID_KHR");
        if (!valid) return device_d3d11_luid(ordinal);
        device_luid_info result;
        check(api->clGetDeviceInfo(device, CL_DEVICE_LUID_KHR, result.bytes.size(), result.bytes.data(), nullptr), "CL_DEVICE_LUID_KHR");
        check(api->clGetDeviceInfo(device, CL_DEVICE_NODE_MASK_KHR, sizeof(result.node_mask), &result.node_mask, nullptr), "CL_DEVICE_NODE_MASK_KHR"); return result;
    }
    device_luid_info device_d3d11_luid(int ordinal) {
#if defined(_WIN32)
        // Older ICDs can identify their DXGI adapter through the standard
        // sharing query even when cl_khr_device_uuid is unavailable. No
        // context or shared allocation is needed to query the adapter.
        cl_platform_id platform = nullptr;
        check(api->clGetDeviceInfo(native_devices.at(ordinal), CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr), "CL_DEVICE_PLATFORM");
        auto query = reinterpret_cast<clGetDeviceIDsFromD3D11KHR_fn>(api->clGetExtensionFunctionAddressForPlatform(platform, "clGetDeviceIDsFromD3D11KHR"));
        require(query, CUDA_ERROR_NOT_SUPPORTED, "OpenCL device LUID is unavailable");
        dynamic_library dxgi{{"dxgi.dll"}};
        auto create = dxgi.get<decltype(&CreateDXGIFactory1)>("CreateDXGIFactory1");
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        require(SUCCEEDED(create(IID_PPV_ARGS(&factory))), CUDA_ERROR_NOT_SUPPORTED, "DXGI identity discovery is unavailable");
        for (UINT i = 0;; ++i) {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            if (!adapter) break;
            cl_uint count = 0;
            if (query(platform, CL_D3D11_DXGI_ADAPTER_KHR, adapter.Get(), CL_ALL_DEVICES_FOR_D3D11_KHR, 0, nullptr, &count) != CL_SUCCESS || count != 1) continue;
            cl_device_id matched = nullptr;
            if (query(platform, CL_D3D11_DXGI_ADAPTER_KHR, adapter.Get(), CL_ALL_DEVICES_FOR_D3D11_KHR, 1, &matched, nullptr) != CL_SUCCESS || matched != native_devices.at(ordinal)) continue;
            DXGI_ADAPTER_DESC1 description{};
            if (FAILED(adapter->GetDesc1(&description))) continue;
            device_luid_info result; std::memcpy(result.bytes.data(), &description.AdapterLuid, result.bytes.size()); result.node_mask = 1; return result;
        }
#else
        (void)ordinal;
#endif
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "OpenCL device LUID is unavailable");
    }
    pci_address device_pci(int ordinal) override {
        device_extension(ordinal, "cl_khr_pci_bus_info"); cl_device_pci_bus_info_khr info{};
        check(api->clGetDeviceInfo(native_devices.at(ordinal), CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(info), &info, nullptr), "CL_DEVICE_PCI_BUS_INFO_KHR");
        return {info.pci_domain, info.pci_bus, info.pci_device, info.pci_function};
    }
    std::vector<device_info> discover() override {
        cl_uint count = 0;
        cl_int result = api->clGetPlatformIDs(0, nullptr, &count);
        if (result == -1001) return {}; // CL_PLATFORM_NOT_FOUND_KHR
        check(result, "clGetPlatformIDs");
        std::vector<cl_platform_id> platforms(count);
        if (count) check(api->clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
        native_devices.clear();
        std::vector<device_info> devices;
        for (auto platform : platforms) {
            cl_uint device_count = 0;
            result = api->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
            if (result == CL_DEVICE_NOT_FOUND) continue;
            check(result, "clGetDeviceIDs");
            std::vector<cl_device_id> values(device_count);
            if (device_count) check(api->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, values.data(), nullptr), "clGetDeviceIDs");
            for (auto device : values) {
                size_t name_size = 0;
                cl_ulong bytes = 0;
                check(api->clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &name_size), "clGetDeviceInfo");
                std::vector<char> name(name_size);
                check(api->clGetDeviceInfo(device, CL_DEVICE_NAME, name.size(), name.data(), nullptr), "clGetDeviceInfo");
                check(api->clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(bytes), &bytes, nullptr), "clGetDeviceInfo");
                devices.push_back({name.data(), static_cast<size_t>(bytes)});
                native_devices.push_back(device);
            }
        }
        return devices;
    }
    std::unique_ptr<session> open(int ordinal) override {
        auto device = native_devices.at(ordinal);
        return std::make_unique<opencl_session>(resources.get(ordinal, [&] { return std::make_shared<opencl_context>(api, device); }));
    }
};
}
std::unique_ptr<backend> make_opencl_backend() { return std::make_unique<opencl_backend>(); }
}
