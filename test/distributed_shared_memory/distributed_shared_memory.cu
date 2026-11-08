extern "C" __global__ void cluster_exchange(unsigned* output, const unsigned* input, unsigned count) {
    __shared__ unsigned mailbox[128];
    const unsigned lane = threadIdx.x, row = blockIdx.x * 128 + lane;
    mailbox[lane] = input[row];
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
    unsigned long long address;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(reinterpret_cast<unsigned long long>(mailbox + lane)));
    unsigned remote, remote_rank, value;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(unsigned(address)), "r"(rank ^ 1));
    asm("getctarank.shared::cluster.u32 %0, %1;" : "=r"(remote_rank) : "r"(remote));
    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(value) : "r"(remote) : "memory");
    // Every CTA stays alive until every remote read has completed, including
    // launches with no active output rows. No thread exits before the barriers.
    asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
    if (lane < count) {
        output[row * 4] = value; output[row * 4 + 1] = mailbox[lane];
        output[row * 4 + 2] = rank; output[row * 4 + 3] = remote_rank;
    }
}
