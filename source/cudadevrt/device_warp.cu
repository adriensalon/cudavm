extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_idx(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_up(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_down(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_bfly(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) bool __cudavm_rtc_vote_all(unsigned,bool);
extern "C" __device__ __attribute__((convergent)) bool __cudavm_rtc_vote_any(unsigned,bool);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_vote_ballot(unsigned,bool);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_activemask();
extern "C" __device__ __attribute__((convergent)) void __cudavm_rtc_syncwarp(unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_any32(unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_any64(unsigned,unsigned long long);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_all32(unsigned,unsigned,int*);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_all64(unsigned,unsigned long long,int*);
__device__ inline void __syncwarp(unsigned mask=0xffffffffu) {__cudavm_rtc_syncwarp(mask);}
__device__ inline int __all_sync(unsigned mask,int predicate) {return __cudavm_rtc_vote_all(mask,predicate!=0);}
__device__ inline int __any_sync(unsigned mask,int predicate) {return __cudavm_rtc_vote_any(mask,predicate!=0);}
__device__ inline unsigned __ballot_sync(unsigned mask,int predicate) {return __cudavm_rtc_vote_ballot(mask,predicate!=0);}
__device__ inline unsigned __activemask() {return __cudavm_rtc_activemask();}
