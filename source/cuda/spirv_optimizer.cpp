#include <spirv-tools/optimizer.hpp>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#if defined(_WIN32)
#define NATIVE_OPT_EXPORT __declspec(dllexport)
#else
#define NATIVE_OPT_EXPORT __attribute__((visibility("default")))
#endif
// Keep the SPIRV-Tools SDK's CRT behind a C boundary. Output is copied by a
// caller-owned callback; neither allocation nor C++ objects cross the DLL ABI.
extern "C" NATIVE_OPT_EXPORT int cudavmOptimizeNativeSpirv(const uint32_t* input,size_t count,
    void (*output)(const uint32_t*,size_t,void*),void* user,char* diagnostic,size_t capacity) noexcept {
    auto report=[&](const std::string& text) {
        if(capacity) {auto length=std::min(text.size(),capacity-1);std::memcpy(diagnostic,text.data(),length);diagnostic[length]=0;}
    };
    try {
        spvtools::Optimizer optimizer(SPV_ENV_OPENGL_4_5);std::string message;
        optimizer.SetMessageConsumer([&](spv_message_level_t,const char*,const spv_position_t&,const char* text) {if(text)message+=std::string(text)+"\n";});
        optimizer.RegisterPerformancePasses();std::vector<uint32_t> words;
        if(!optimizer.Run(input,count,&words)) {report(message);return 1;}
        output(words.data(),words.size(),user);return 0;
    } catch(const std::bad_alloc&) {report("native SPIR-V optimization allocation failed");return 2;}
      catch(const std::exception& error) {report(error.what());return 1;}
}
