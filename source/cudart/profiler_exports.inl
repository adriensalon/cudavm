RUNTIME_API(cudaProfilerStart, (), runtime_current(); check(runtime_driver_wait([&] { return cuProfilerStart(); })))
RUNTIME_API(cudaProfilerStop, (), runtime_current(); check(runtime_driver_wait([&] { return cuProfilerStop(); })))
