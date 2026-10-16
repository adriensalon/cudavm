void runtime_stream_callback(cudaStream_t stream, cudaStream_t reported, cudaStreamCallback_t callback, void* data, unsigned flags) {
    require(callback && flags == 0, CUDA_ERROR_INVALID_VALUE, "invalid Runtime stream callback or flags"); runtime_current();
    struct callback_state {
        cudaStream_t stream; cudaStreamCallback_t callback; void* data;
        static void CUDA_CB invoke(CUstream, CUresult status, void* pointer) {
            std::unique_ptr<callback_state> state(static_cast<callback_state*>(pointer));
            state->callback(state->stream, runtime_error(status), state->data);
        }
    };
    auto state = std::make_unique<callback_state>(callback_state{reported, callback, data});
    check(cuStreamAddCallback(reinterpret_cast<CUstream>(stream), callback_state::invoke, state.get(), flags)); state.release();
}
