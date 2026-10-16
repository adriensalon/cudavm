#include "dynamic_library.hpp"
#include "device_resources.hpp"
#include "../cudavm/spirv_artifact.hpp"
#include "opengl_shader.hpp"
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <limits>
#include <atomic>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#if defined(_WIN32)
#define GL_CALL __stdcall
#else
#define GL_CALL
#endif
namespace cudavm {
namespace {
using uint = unsigned int;
using enum_type = unsigned int;
using size_type = std::ptrdiff_t;
constexpr enum_type copy_read = 0x8f36, copy_write = 0x8f37, dynamic_copy = 0x88ea;
struct opengl_context {
#if defined(_WIN32)
    dynamic_library gl{{"opengl32.dll"}};
    using create_type = HGLRC (WINAPI*)(HDC);
    using delete_type = BOOL (WINAPI*)(HGLRC);
    using current_type = BOOL (WINAPI*)(HDC, HGLRC);
    using proc_type = PROC (WINAPI*)(LPCSTR);
    create_type create = gl.get<create_type>("wglCreateContext");
    delete_type remove = gl.get<delete_type>("wglDeleteContext");
    current_type current = gl.get<current_type>("wglMakeCurrent");
    proc_type proc = gl.get<proc_type>("wglGetProcAddress");
    HWND window = nullptr;
    HDC dc = nullptr;
    HGLRC context = nullptr;
    explicit opengl_context(opengl_context* share = nullptr) {
        try {
            WNDCLASSA window_class{};
            window_class.style = CS_OWNDC;
            window_class.lpfnWndProc = DefWindowProcA;
            window_class.hInstance = GetModuleHandleA(nullptr);
            window_class.lpszClassName = "cudavm_private_opengl";
            require(RegisterClassA(&window_class) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS, CUDA_ERROR_NO_DEVICE, "OpenGL window class creation failed");
            window = CreateWindowExA(0, window_class.lpszClassName, "cudavm", WS_DISABLED, 0, 0, 1, 1, nullptr, nullptr, window_class.hInstance, nullptr);
            require(window != nullptr, CUDA_ERROR_NO_DEVICE, "OpenGL hidden window creation failed");
            dc = GetDC(window);
            require(dc != nullptr, CUDA_ERROR_NO_DEVICE, "OpenGL device context unavailable");
            PIXELFORMATDESCRIPTOR format{};
            format.nSize = sizeof(format);
            format.nVersion = 1;
            format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            format.iPixelType = PFD_TYPE_RGBA;
            format.cColorBits = 24;
            format.iLayerType = PFD_MAIN_PLANE;
            int pixel = ChoosePixelFormat(dc, &format);
            require(pixel > 0 && SetPixelFormat(dc, pixel, &format), CUDA_ERROR_NO_DEVICE, "OpenGL pixel format unavailable");
            context = create(dc);
            require(context != nullptr && current(dc, context), CUDA_ERROR_NO_DEVICE, "OpenGL bootstrap context unavailable");
            using modern_type = HGLRC (WINAPI*)(HDC, HGLRC, const int*);
            auto modern = reinterpret_cast<modern_type>(find("wglCreateContextAttribsARB"));
            require(modern != nullptr, CUDA_ERROR_NOT_SUPPORTED, "OpenGL 4.3 context creation unavailable");
            const int attributes[] = {0x2091, 4, 0x2092, 3, 0x9126, 1, 0};
            // The temporary legacy context is outside the requested share
            // group. Detach it before creating a shared core context.
            current(nullptr, nullptr);
            HGLRC replacement = modern(dc, share ? share->context : nullptr, attributes);
            if (!replacement) throw failure(CUDA_ERROR_NOT_SUPPORTED, "OpenGL 4.3 core context unavailable: " + std::to_string(GetLastError()));
            current(nullptr, nullptr);
            remove(context);
            context = replacement;
            require(current(dc, context) != 0, CUDA_ERROR_NO_DEVICE, "OpenGL context activation failed");
        } catch (...) { cleanup(); throw; }
    }
    void* find(const char* name) {
        auto value = reinterpret_cast<void*>(proc(name));
        auto address = reinterpret_cast<std::uintptr_t>(value);
        if (!value || address <= 3 || address == std::uintptr_t(-1)) return gl.find(name);
        return value;
    }
    void cleanup() {
        if (context) { current(nullptr, nullptr); remove(context); context = nullptr; }
        if (dc) { ReleaseDC(window, dc); dc = nullptr; }
        if (window) { DestroyWindow(window); window = nullptr; }
    }
#else
    dynamic_library egl{{"libEGL.so.1", "libEGL.so"}};
    dynamic_library gl{{"libOpenGL.so.0", "libGL.so.1"}};
    using boolean = unsigned int;
    void* display = nullptr;
    void* context = nullptr;
    void* surface = nullptr;
    using get_display_type = void* (*)(void*);
    using initialize_type = boolean (*)(void*, int*, int*);
    using bind_type = boolean (*)(unsigned);
    using choose_type = boolean (*)(void*, const int*, void**, int, int*);
    using surface_type = void* (*)(void*, void*, const int*);
    using create_type = void* (*)(void*, void*, void*, const int*);
    using current_type = boolean (*)(void*, void*, void*, void*);
    using destroy_type = boolean (*)(void*, void*);
    using terminate_type = boolean (*)(void*);
    using proc_type = void* (*)(const char*);
    get_display_type get_display = egl.get<get_display_type>("eglGetDisplay");
    initialize_type initialize = egl.get<initialize_type>("eglInitialize");
    bind_type bind = egl.get<bind_type>("eglBindAPI");
    choose_type choose = egl.get<choose_type>("eglChooseConfig");
    surface_type create_surface = egl.get<surface_type>("eglCreatePbufferSurface");
    create_type create = egl.get<create_type>("eglCreateContext");
    current_type current = egl.get<current_type>("eglMakeCurrent");
    destroy_type destroy_context = egl.get<destroy_type>("eglDestroyContext");
    destroy_type destroy_surface = egl.get<destroy_type>("eglDestroySurface");
    terminate_type terminate = egl.get<terminate_type>("eglTerminate");
    proc_type proc = egl.get<proc_type>("eglGetProcAddress");
    bool owns_display = true;
    explicit opengl_context(opengl_context* share = nullptr) {
        try {
            int major = 0, minor = 0;
            owns_display = !share;
            display = share ? share->display : get_display(nullptr);
            if (!share && (!display || !initialize(display, &major, &minor))) {
                using platform_type = void* (*)(unsigned, void*, const int*);
                auto platform = reinterpret_cast<platform_type>(proc("eglGetPlatformDisplayEXT"));
                display = platform ? platform(0x31dd, nullptr, nullptr) : nullptr;
                require(display && initialize(display, &major, &minor), CUDA_ERROR_NO_DEVICE, "EGL display unavailable");
            }
            require(bind(0x30a2) != 0, CUDA_ERROR_NOT_SUPPORTED, "EGL desktop OpenGL unavailable");
            int count = 0;
            void* config = nullptr;
            const int config_attributes[] = {0x3033, 1, 0x3040, 8, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3038};
            require(choose(display, config_attributes, &config, 1, &count) && count > 0, CUDA_ERROR_NOT_SUPPORTED, "EGL OpenGL pbuffer unavailable");
            const int dimensions[] = {0x3057, 1, 0x3056, 1, 0x3038};
            surface = create_surface(display, config, dimensions);
            const int attributes[] = {0x3098, 4, 0x30fb, 3, 0x30fd, 1, 0x3038};
            context = create(display, config, share ? share->context : nullptr, attributes);
            require(surface && context && current(display, surface, surface, context), CUDA_ERROR_NOT_SUPPORTED, "EGL OpenGL 4.3 context unavailable");
        } catch (...) { cleanup(); throw; }
    }
    void* find(const char* name) { auto value = proc(name); return value ? value : gl.find(name); }
    void cleanup() {
        if (display) {
            current(display, nullptr, nullptr, nullptr);
            if (context) destroy_context(display, context);
            if (surface) destroy_surface(display, surface);
            if (owns_display) terminate(display);
        }
        display = context = surface = nullptr;
    }
#endif
    ~opengl_context() { cleanup(); }
    void detach() {
#if defined(_WIN32)
        current(nullptr, nullptr);
#else
        current(display, nullptr, nullptr, nullptr);
#endif
    }
};
struct opengl_api {
    using gen_type = void (GL_CALL*)(int, uint*);
    using delete_type = void (GL_CALL*)(int, const uint*);
    using bind_type = void (GL_CALL*)(enum_type, uint);
    using data_type = void (GL_CALL*)(enum_type, size_type, const void*, enum_type);
    using write_type = void (GL_CALL*)(enum_type, size_type, size_type, const void*);
    using read_type = void (GL_CALL*)(enum_type, size_type, size_type, void*);
    using copy_type = void (GL_CALL*)(enum_type, enum_type, size_type, size_type, size_type);
    using finish_type = void (GL_CALL*)();
    using error_type = enum_type (GL_CALL*)();
    using string_type = const unsigned char* (GL_CALL*)(enum_type);
    using create_shader_type = uint (GL_CALL*)(enum_type);
    using shader_source_type = void (GL_CALL*)(uint,int,const char* const*,const int*);
    using shader_binary_type = void (GL_CALL*)(int,const uint*,enum_type,const void*,int);
    using specialize_type = void (GL_CALL*)(uint,const char*,uint,const uint*,const uint*);
    using shader_action_type = void (GL_CALL*)(uint);
    using query_type = void (GL_CALL*)(uint,enum_type,int*);
    using log_type = void (GL_CALL*)(uint,int,int*,char*);
    using create_program_type = uint (GL_CALL*)();
    using attach_type = void (GL_CALL*)(uint,uint);
    using bind_base_type = void (GL_CALL*)(enum_type,uint,uint);
    using dispatch_type = void (GL_CALL*)(uint,uint,uint);
    using barrier_type = void (GL_CALL*)(enum_type);
    using fence_type = void* (GL_CALL*)(enum_type, uint);
    using wait_sync_type = enum_type (GL_CALL*)(void*, uint, uint64_t);
    using delete_sync_type = void (GL_CALL*)(void*);
    using query_counter_type = void (GL_CALL*)(uint,enum_type);
    using query_result_type = void (GL_CALL*)(uint,enum_type,uint64_t*);
    gen_type gen_queries;delete_type delete_queries;query_counter_type query_counter;query_result_type query_result;
    fence_type fence; wait_sync_type wait_sync; delete_sync_type delete_sync; finish_type flush;
    gen_type gen; delete_type remove; bind_type bind; data_type data; write_type write; read_type read; copy_type copy;
    finish_type finish; error_type error; string_type string;
    create_shader_type create_shader;shader_source_type shader_source;shader_action_type compile_shader,delete_shader;
    query_type shader_query,program_query;log_type shader_log,program_log;
    create_program_type create_program;attach_type attach;shader_action_type link_program,use_program,delete_program;
    bind_base_type bind_base;dispatch_type dispatch;barrier_type memory_barrier;
    unsigned direct_buffers = 0;
    shader_binary_type shader_binary = nullptr;
    specialize_type specialize = nullptr;
    std::atomic<bool> native_spirv{false};
    bool native_int64 = false;
    unsigned timestamp_bits = 0;
    unsigned native_group_count[3]{}, native_group_size[3]{}, native_invocations = 0, native_shared_bytes = 0;
    shader_resource_limits shader_limits() const {
        shader_resource_limits result; result.threads = native_invocations;
        std::copy_n(native_group_count, 3, result.grid.begin()); std::copy_n(native_group_size, 3, result.block.begin());
        result.shared_bytes = native_shared_bytes; return result;
    }
    size_t memory_bytes = 0;
#include "opengl_interop_api.inl"
    template<class T> T load(opengl_context& context, const char* name) {
        auto value = reinterpret_cast<T>(context.find(name));
        require(value != nullptr, CUDA_ERROR_NOT_SUPPORTED, name);
        return value;
    }
    explicit opengl_api(opengl_context& context) {
        gen = load<gen_type>(context, "glGenBuffers"); remove = load<delete_type>(context, "glDeleteBuffers");
        bind = load<bind_type>(context, "glBindBuffer"); data = load<data_type>(context, "glBufferData");
        write = load<write_type>(context, "glBufferSubData"); read = load<read_type>(context, "glGetBufferSubData");
        copy = load<copy_type>(context, "glCopyBufferSubData"); finish = load<finish_type>(context, "glFinish");
        error = load<error_type>(context, "glGetError"); string = load<string_type>(context, "glGetString");
        fence = load<fence_type>(context, "glFenceSync"); wait_sync = load<wait_sync_type>(context, "glClientWaitSync");
        delete_sync = load<delete_sync_type>(context, "glDeleteSync"); flush = load<finish_type>(context, "glFlush");
        gen_queries=load<gen_type>(context,"glGenQueries");delete_queries=load<delete_type>(context,"glDeleteQueries");
        query_counter=load<query_counter_type>(context,"glQueryCounter");query_result=load<query_result_type>(context,"glGetQueryObjectui64v");
        auto get_query=load<query_type>(context,"glGetQueryiv");int clock_bits=0;get_query(0x8e28,0x8864,&clock_bits);
        timestamp_bits=unsigned(std::clamp(clock_bits,0,64));
        create_shader=load<create_shader_type>(context,"glCreateShader");shader_source=load<shader_source_type>(context,"glShaderSource");
        compile_shader=load<shader_action_type>(context,"glCompileShader");delete_shader=load<shader_action_type>(context,"glDeleteShader");
        shader_query=load<query_type>(context,"glGetShaderiv");shader_log=load<log_type>(context,"glGetShaderInfoLog");
        create_program=load<create_program_type>(context,"glCreateProgram");attach=load<attach_type>(context,"glAttachShader");
        link_program=load<shader_action_type>(context,"glLinkProgram");use_program=load<shader_action_type>(context,"glUseProgram");
        delete_program=load<shader_action_type>(context,"glDeleteProgram");program_query=load<query_type>(context,"glGetProgramiv");program_log=load<log_type>(context,"glGetProgramInfoLog");
        bind_base=load<bind_base_type>(context,"glBindBufferBase");dispatch=load<dispatch_type>(context,"glDispatchCompute");memory_barrier=load<barrier_type>(context,"glMemoryBarrier");
        auto get_integer=load<void (GL_CALL*)(enum_type,int*)>(context,"glGetIntegerv");
        int blocks=0,bindings=0;get_integer(0x90db,&blocks);get_integer(0x90dd,&bindings);
        direct_buffers=unsigned(std::max(0,std::min({6,blocks-2,bindings-2})));
        auto indexed = load<void (GL_CALL*)(enum_type,uint,int*)>(context,"glGetIntegeri_v");
        for (unsigned axis = 0; axis < 3; ++axis) {
            int count = 0, size = 0; indexed(0x91be, axis, &count); indexed(0x91bf, axis, &size);
            native_group_count[axis] = unsigned(std::max(0,count)); native_group_size[axis] = unsigned(std::max(0,size));
        }
        int invocations = 0; get_integer(0x90eb, &invocations); native_invocations = unsigned(std::max(0,invocations));
        int shared = 0; get_integer(0x8262, &shared); native_shared_bytes = unsigned(std::max(0,shared));
        shader_binary = reinterpret_cast<shader_binary_type>(context.find("glShaderBinary"));
        specialize = reinterpret_cast<specialize_type>(context.find("glSpecializeShader"));
        if (!specialize) specialize = reinterpret_cast<specialize_type>(context.find("glSpecializeShaderARB"));
        initialize_interop(context);
    }
    void check(const char* operation) {
        auto value = error();
        if (value) throw failure(value == 0x0505 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN, std::string(operation) + " failed: " + std::to_string(value));
    }
};
// A dedicated owner thread keeps contexts usable by arbitrary CUDA host threads
// and creates/destroys hidden Windows resources on their owning thread.
struct opengl_worker {
    // A share-group root stays alive until every member's buffers and fences
    // retire. Each member keeps its own native context and execution thread.
    std::shared_ptr<opengl_worker> root;
    opengl_context* native = nullptr;
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
    bool stopping = false;
    std::unique_ptr<opengl_api> api;
    std::thread thread;
    device_info info;
    std::vector<uint> external_buffers;
    explicit opengl_worker(std::shared_ptr<opengl_worker> share = {}) : root(std::move(share)) {
        auto initialized = std::make_shared<std::promise<device_info>>();
        auto future = initialized->get_future();
        thread = std::thread([this, initialized] {
            bool published = false;
            try {
                opengl_context context(root ? root->native : nullptr);
                native = &context;
                api = std::make_unique<opengl_api>(context);
                const char* renderer = reinterpret_cast<const char*>(api->string(0x1f01));
                require(renderer != nullptr, CUDA_ERROR_NO_DEVICE, "OpenGL renderer unavailable");
                std::string name(renderer);
                require(name.find("llvmpipe") == std::string::npos && name.find("softpipe") == std::string::npos && name.find("GDI Generic") == std::string::npos, CUDA_ERROR_NO_DEVICE, "software OpenGL renderer is not a GPU backend");
                // The resource root owns no execution queue. WGL requires it
                // to be detached while another thread creates a shared member.
                if (!root) context.detach();
                initialized->set_value({name, api->memory_bytes});
                published = true;
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mutex);
                        ready.wait(lock, [&] { return stopping || !tasks.empty(); });
                        if (tasks.empty() && stopping) break;
                        task = std::move(tasks.front()); tasks.pop_front();
                    }
                    task();
                }
                api.reset();
                native = nullptr;
            } catch (...) {
                // Invoked tasks use packaged_task and carry their own errors.
                // This path reports initialization failures before publication.
                if (!published) initialized->set_exception(std::current_exception());
            }
        });
        try { info = future.get(); }
        catch (...) { thread.join(); throw; }
    }
    ~opengl_worker() {
        { std::lock_guard lock(mutex); stopping = true; }
        ready.notify_one();
        if (thread.joinable()) thread.join();
    }
    template<class F> auto invoke(F function) {
        using result_type = decltype(function());
        auto task = std::make_shared<std::packaged_task<result_type()>>(std::move(function));
        auto future = task->get_future();
        { std::lock_guard lock(mutex); tasks.emplace_back([task] { (*task)(); }); }
        ready.notify_one();
        return future.get();
    }
    const opengl_worker* resource_owner() const { return root ? root.get() : this; }
};
struct opengl_buffer : buffer {
    std::shared_ptr<opengl_worker> owner;
    uint value = 0;
    std::shared_ptr<external_memory> imported;
    opengl_buffer(std::shared_ptr<opengl_worker> input, size_t bytes, uint object, std::shared_ptr<external_memory> memory)
        : buffer(bytes), owner(std::move(input)), value(object), imported(std::move(memory)) {}
    opengl_buffer(std::shared_ptr<opengl_worker> input, size_t bytes) : buffer(bytes), owner(std::move(input)) {
        value = owner->invoke([&] {
            uint result = 0;
            owner->api->gen(1, &result);
            owner->api->bind(copy_write, result);
            owner->api->data(copy_write, static_cast<size_type>(bytes), nullptr, dynamic_copy);
            try { owner->api->check("glBufferData"); }
            catch (...) { owner->api->remove(1, &result); throw; }
            return result;
        });
    }
    ~opengl_buffer() { if (value) { try { owner->invoke([&] { std::erase(owner->external_buffers, value); owner->api->remove(1, &value); }); } catch (...) {} } }
    size_t physical_size() const override { return imported ? 0 : size; }
};
struct opengl_task : device_task {
    std::shared_ptr<opengl_worker> owner;
    void* fence = nullptr;
    uint timestamp_query = 0;
    std::shared_ptr<device_timestamp> timestamp;
    uint compute_queries[2]{};
    std::shared_ptr<compute_profile> marker_profile,dispatch_profile;
    bool measured = false;
    explicit opengl_task(std::shared_ptr<opengl_worker> input) : owner(std::move(input)) {}
    void record() {
        fence = owner->api->fence(0x9117, 0); require(fence, CUDA_ERROR_UNKNOWN, "glFenceSync failed"); owner->api->flush();
    }
    bool query() override {
        return owner->invoke([&] {
            auto status = owner->api->wait_sync(fence, 0, 0);
            require(status != 0x911d, CUDA_ERROR_UNKNOWN, "glClientWaitSync failed");
            bool ready=status == 0x911a || status == 0x911c;
            if(ready && timestamp) {
                owner->api->query_result(timestamp_query,0x8866,&timestamp->ticks);owner->api->check("glGetQueryObjectui64v");
                timestamp->nanoseconds_per_tick=1;
                timestamp->valid_bits=owner->api->timestamp_bits;
                if(marker_profile) marker_profile->snapshot(*timestamp);
            }
            if(ready && dispatch_profile && !measured) {
                uint64_t begin{},end{};owner->api->query_result(compute_queries[0],0x8866,&begin);owner->api->query_result(compute_queries[1],0x8866,&end);
                owner->api->check("glGetQueryObjectui64v dispatch");auto delta=end-begin;
                if(owner->api->timestamp_bits<64)delta&=(uint64_t(1)<<owner->api->timestamp_bits)-1;
                dispatch_profile->nanoseconds.fetch_add(delta);measured=true;
            }
            return ready;
        });
    }
    ~opengl_task() {
        if (fence) try { owner->invoke([&] { auto status = owner->api->wait_sync(fence, 0, 0); if (status == 0x911b) owner->api->finish(); owner->api->delete_sync(fence); }); } catch (...) {}
        if(timestamp_query) try { owner->invoke([&] {owner->api->delete_queries(1,&timestamp_query);}); } catch(...) {}
        if(compute_queries[0]) try { owner->invoke([&] {owner->api->delete_queries(2,compute_queries);}); } catch(...) {}
    }
};
#include "opengl_interop.inl"
struct opengl_pipeline : compute_pipeline {
    std::shared_ptr<opengl_worker> owner;
    uint program=0;
    explicit opengl_pipeline(std::shared_ptr<opengl_worker> worker) : owner(std::move(worker)) {}
    ~opengl_pipeline() {
        if(program) { try { owner->invoke([&] { owner->api->delete_program(program); }); } catch(...) {} }
    }
};
struct opengl_session : session {
    std::shared_ptr<opengl_worker> owner;
    explicit opengl_session(std::shared_ptr<opengl_worker> input) : owner(std::move(input)) {}
    interop_capabilities interop() override { return owner->invoke([&] { return owner->api->caps; }); }
    std::shared_ptr<external_memory> import_memory(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d, const cudavmExternalMemoryMetadata* = nullptr) override {
        require(interop().memory & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "GL external memory extensions are unavailable");
        return std::make_shared<opengl_memory>(owner, d);
    }
    std::shared_ptr<external_semaphore> import_semaphore(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d) override {
        require(interop().semaphore & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "GL external semaphore extensions are unavailable");
        return std::make_shared<opengl_semaphore>(owner, d);
    }
    std::unique_ptr<buffer> allocate(size_t bytes) override { return std::make_unique<opengl_buffer>(owner, bytes); }
    device_work enqueue_write(buffer& value, size_t offset, std::span<const std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_write(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address()) return session::enqueue_write(value, offset, bytes, queue);
        auto task = std::make_shared<opengl_task>(owner);
        owner->invoke([&] {
            owner->api->bind(copy_write, static_cast<opengl_buffer&>(value).value);
            owner->api->write(copy_write, size_type(offset), size_type(bytes.size()), bytes.data());
            owner->api->check("glBufferSubData"); task->record();
        }); return task;
    }
    device_work enqueue_copy(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes, native_queue* queue) override {
        if (&to.storage() != &to || &from.storage() != &from) return enqueue_copy(to.storage(), destination + to.storage_offset(), from.storage(), source + from.storage_offset(), bytes, queue);
        if (to.host_address() || from.host_address()) return session::enqueue_copy(to, destination, from, source, bytes, queue);
        auto task = std::make_shared<opengl_task>(owner);
        owner->invoke([&] {
            owner->api->bind(copy_read, static_cast<opengl_buffer&>(from).value);
            owner->api->bind(copy_write, static_cast<opengl_buffer&>(to).value);
            owner->api->copy(copy_read, copy_write, size_type(source), size_type(destination), size_type(bytes));
            owner->api->check("glCopyBufferSubData"); task->record();
        }); return task;
    }
    device_work enqueue_read(buffer& value, size_t offset, std::span<std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_read(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address()) return session::enqueue_read(value, offset, bytes, queue);
        std::shared_ptr<buffer> staging = allocate(bytes.size());
        auto continuation = std::make_shared<device_continuation>(device_work{}, [this, staging, bytes] { read(*staging, 0, bytes); });
        continuation->task = enqueue_copy(*staging, 0, value, offset, bytes.size(), queue); return continuation;
    }
    void write_device(buffer& value, size_t offset, std::span<const std::byte> data) override {
        if (&value.storage() != &value) { write_device(value.storage(), offset + value.storage_offset(), data); return; }
        owner->invoke([&] {
            owner->api->bind(copy_write, static_cast<opengl_buffer&>(value).value);
            owner->api->write(copy_write, static_cast<size_type>(offset), static_cast<size_type>(data.size()), data.data());
            owner->api->finish(); owner->api->check("glBufferSubData");
        });
    }
    void read_device(buffer& value, size_t offset, std::span<std::byte> data) override {
        if (&value.storage() != &value) { read_device(value.storage(), offset + value.storage_offset(), data); return; }
        owner->invoke([&] {
            owner->api->bind(copy_read, static_cast<opengl_buffer&>(value).value);
            owner->api->read(copy_read, static_cast<size_type>(offset), static_cast<size_type>(data.size()), data.data());
            owner->api->check("glGetBufferSubData");
        });
    }
    void copy_device(buffer& to, size_t to_offset, buffer& from, size_t from_offset, size_t bytes) override {
        if (&to.storage() != &to || &from.storage() != &from) { copy_device(to.storage(), to_offset + to.storage_offset(), from.storage(), from_offset + from.storage_offset(), bytes); return; }
        owner->invoke([&] {
            owner->api->bind(copy_read, static_cast<opengl_buffer&>(from).value);
            owner->api->bind(copy_write, static_cast<opengl_buffer&>(to).value);
            owner->api->copy(copy_read, copy_write, static_cast<size_type>(from_offset), static_cast<size_type>(to_offset), static_cast<size_type>(bytes));
            owner->api->finish(); owner->api->check("glCopyBufferSubData");
        });
    }
    void synchronize() override { owner->invoke([&] { owner->api->finish(); owner->api->check("glFinish"); }); }
    cudavmBackend kind() const override { return cudavmBackendOpenGL; }
    device_work enqueue_timestamp(std::shared_ptr<device_timestamp> timestamp,native_queue*) override {
        if(!owner->api->timestamp_bits) return {};
        auto task=std::make_shared<opengl_task>(owner);task->timestamp=std::move(timestamp);task->marker_profile=profile;
        owner->invoke([&] {
            owner->api->gen_queries(1,&task->timestamp_query);owner->api->query_counter(task->timestamp_query,0x8e28);
            owner->api->check("glQueryCounter");task->record();
        });return task;
    }
    unsigned direct_binding_limit() const override { return owner->api->direct_buffers; }
    bool can_copy_direct(const buffer& value) const override {
        auto native = dynamic_cast<const opengl_buffer*>(&value.storage());
        return native && native->owner->resource_owner() == owner->resource_owner() && !native->imported;
    }
    bool can_bind_direct(const buffer& value) const override {
        auto native=dynamic_cast<const opengl_buffer*>(&value.storage());
        return native && native->owner->resource_owner()==owner->resource_owner() && !native->imported && value.size%4==0 && value.storage_offset()%4==0 && value.size<=UINT32_MAX && value.storage_offset()<=UINT32_MAX-value.size;
    }
    bool supports_native_spirv(std::span<const uint32_t> code) const override {
        if ((!owner->api->native_spirv && !native_glsl_configured) || !owner->api->native_int64) return false;
        try {
            auto gl = cudavm::adapt_spirv_for_opengl(code);
            for (size_t i = 5; i < gl.size(); i += gl[i] >> 16)
                if ((gl[i] & 0xffffu) == 17 && gl[i + 1] != 1 && gl[i + 1] != 10 && gl[i + 1] != 11) return false;
            return true;
        } catch (const std::exception&) { return false; }
    }
    unsigned native_block_limit() const override { return owner->api->native_invocations; }
    shader_resource_limits shader_limits() const override { return owner->api->shader_limits(); }
    void validate_native_spirv_shape(const launch_shape& shape) const override {
        uint64_t threads = 1;
        for (unsigned axis = 0; axis < 3; ++axis) {
            require(shape.grid[axis] <= owner->api->native_group_count[axis] && shape.block[axis] <= owner->api->native_group_size[axis], CUDA_ERROR_INVALID_VALUE, "native SPIR-V dimensions exceed OpenGL device limits");
            threads *= shape.block[axis];
        }
        require(threads <= owner->api->native_invocations, CUDA_ERROR_INVALID_VALUE, "native SPIR-V block exceeds OpenGL invocation limits");
    }
    std::unique_ptr<compute_pipeline> compile_native_spirv(std::span<const uint32_t> input, const launch_shape& shape) override {
        require(supports_native_spirv(input), CUDA_ERROR_NOT_SUPPORTED, "OpenGL SPIR-V ingestion or required device capabilities are unavailable");
        validate_native_spirv_shape(shape); auto code = cudavm::adapt_spirv_for_opengl(input);
        // Binary specialization is unreliable on some otherwise conforming GL
        // drivers. GLSL ingestion of the same native IR avoids that vendor JIT.
        if(native_glsl_configured) return compile_native_glsl(native_glsl(code));
        auto compiled = std::make_unique<opengl_pipeline>(owner);
        compiled->program = owner->invoke([&] {
            auto& api = *owner->api; api.check("OpenGL native SPIR-V preflight");
            uint shader = api.create_shader(0x91b9), program = 0;
            auto reject = [&](const char* stage, const std::string& log) {
                if (log.find("C9999") != log.npos) {
                    api.native_spirv = false;
                    // Specialization can also set a GL error. Consume errors
                    // belonging to this rejected compile before auto fallback.
                    for (unsigned i = 0; i < 16 && api.error(); ++i) {}
                    throw failure(CUDA_ERROR_NOT_SUPPORTED,"OpenGL driver SPIR-V compiler rejected the native profile with an internal error: " + log);
                }
                throw failure(CUDA_ERROR_INVALID_IMAGE,std::string("OpenGL native SPIR-V ") + stage + " failed: " + log);
            };
            try {
                api.shader_binary(1, &shader, 0x9551, code.data(), int(code.size() * 4));
                api.specialize(shader, "main", 0, nullptr, nullptr);
                int status = 0; api.shader_query(shader, 0x8b81, &status);
                if (!status) { int size = 0; api.shader_query(shader, 0x8b84, &size); std::string log(std::max(1,size),'\0'); api.shader_log(shader,int(log.size()),nullptr,log.data()); reject("specialization",log); }
                program = api.create_program(); api.attach(program, shader); api.link_program(program); api.program_query(program, 0x8b82, &status);
                if (!status) {
                    int size = 0; api.program_query(program,0x8b84,&size); std::string log(std::max(1,size),'\0'); api.program_log(program,int(log.size()),nullptr,log.data());
                    reject("link",log);
                }
                api.check("OpenGL native SPIR-V ingestion");
            } catch (...) { if (program) api.delete_program(program); if (shader) api.delete_shader(shader); throw; }
            api.delete_shader(shader); return program;
        });
        return compiled;
    }
    std::unique_ptr<compute_pipeline> compile_native_glsl(const std::string& source) {
        auto compiled=std::make_unique<opengl_pipeline>(owner);
        compiled->direct_buffers=0;
        compiled->program=owner->invoke([&] {
            auto& api=*owner->api;uint shader=api.create_shader(0x91b9),program=0;
            try {
                auto* text=source.c_str();api.shader_source(shader,1,&text,nullptr);api.compile_shader(shader);
                int status=0;api.shader_query(shader,0x8b81,&status);
                if(!status) { int length=0;api.shader_query(shader,0x8b84,&length);std::string log(std::max(1,length),'\0');api.shader_log(shader,int(log.size()),nullptr,log.data());throw failure(CUDA_ERROR_INVALID_IMAGE,"OpenGL shader compilation failed: "+log); }
                program=api.create_program();api.attach(program,shader);api.link_program(program);api.program_query(program,0x8b82,&status);
                if(!status) {int length=0;api.program_query(program,0x8b84,&length);std::string log(std::max(1,length),'\0');api.program_log(program,int(log.size()),nullptr,log.data());throw failure(CUDA_ERROR_INVALID_IMAGE,"OpenGL shader link failed: "+log);}
            } catch(...) { if(program) api.delete_program(program);if(shader) api.delete_shader(shader);throw; }
            api.delete_shader(shader);
            return program;
        });
        return compiled;
    }
    void dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters) override {
        wait_device(enqueue_dispatch(pipeline, shape, memory, parameters, nullptr));
    }
    device_work enqueue_dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters, native_queue*) override {
        return enqueue_dispatch_bound(pipeline,shape,memory,parameters,{},nullptr);
    }
    device_work enqueue_dispatch_bound(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters,std::span<buffer* const> bindings,native_queue*) override {
        require(bindings.size()==pipeline.direct_buffers,CUDA_ERROR_INVALID_VALUE,"OpenGL direct binding count mismatch");
        for(auto resource:bindings) require(can_bind_direct(*resource),CUDA_ERROR_INVALID_VALUE,"incompatible OpenGL direct buffer");
        auto task = std::make_shared<opengl_task>(owner);
        auto program=static_cast<opengl_pipeline&>(pipeline).program;
        owner->invoke([&] {
            auto& api=*owner->api;
            auto unbind=[&] { api.use_program(0);for(unsigned i=0;i<bindings.size()+2;++i)api.bind_base(0x90d2,i,0); };
            try {
                api.use_program(program);api.bind_base(0x90d2,0,static_cast<opengl_buffer&>(memory).value);api.bind_base(0x90d2,1,static_cast<opengl_buffer&>(parameters).value);
                for(unsigned i=0;i<bindings.size();++i)api.bind_base(0x90d2,i+2,static_cast<opengl_buffer&>(bindings[i]->storage()).value);
                api.memory_barrier(0x2000|0x200);
                if(profile->enabled && api.timestamp_bits) {task->dispatch_profile=profile;api.gen_queries(2,task->compute_queries);api.query_counter(task->compute_queries[0],0x8e28);}
                api.dispatch(shape.grid[0],shape.grid[1],shape.grid[2]);
                if(task->dispatch_profile) api.query_counter(task->compute_queries[1],0x8e28);
                api.memory_barrier(0x2000|0x200);api.check("glDispatchCompute");task->record();
            } catch(...) { unbind();throw; }
            unbind();
        }); return task;
    }
};
struct opengl_backend : shader_backend {
    shader_resource_limits limits;
    shader_resource_limits shader_limits(int) override { return limits; }
    device_resources<opengl_worker> resources;
    std::optional<std::array<char, 16>> uuid;
    std::optional<device_luid_info> luid;
    std::vector<device_info> discover() override {
        auto probe = std::make_shared<opengl_worker>();
        probe->invoke([&] { uuid = probe->api->identity_uuid; luid = probe->api->identity_luid; limits = probe->api->shader_limits(); }); return {probe->info};
    }
    std::array<char, 16> device_uuid(int, bool) override { require(bool(uuid), CUDA_ERROR_NOT_SUPPORTED, "OpenGL device UUID is unavailable"); return *uuid; }
    device_luid_info device_luid(int) override { require(bool(luid), CUDA_ERROR_NOT_SUPPORTED, "OpenGL device LUID is unavailable"); return *luid; }
    std::unique_ptr<session> open(int device) override {
        require(device == 0, CUDA_ERROR_INVALID_DEVICE, "OpenGL adapter ordinal out of range");
        auto root = resources.get(0, [] { return std::make_shared<opengl_worker>(); });
        return std::make_unique<opengl_session>(std::make_shared<opengl_worker>(std::move(root)));
    }
};
}
std::unique_ptr<backend> make_opengl_backend() { return std::make_unique<opengl_backend>(); }
}
