DRIVER_API(cuProfilerStart, (), driver_ready(); auto owner = current();
    native_call(owner, [&] { owner->native->set_profiling(true); }))
DRIVER_API(cuProfilerStop, (), driver_ready(); auto owner = current();
    native_call(owner, [&] { owner->native->set_profiling(false); }))
