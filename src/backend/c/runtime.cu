__shared__ uint3 __shady_make_thread_local(RealGlobalInvocationId);
__shared__ uint3 __shady_make_thread_local(RealLocalInvocationId);

#define GlobalInvocationId __shady_thread_local_access(RealGlobalInvocationId)
#define LocalInvocationId __shady_thread_local_access(RealLocalInvocationId)

__device__ void __shady_prepare_builtins() {
    LocalInvocationId.x = threadIdx.x;
    LocalInvocationId.y = threadIdx.y;
    LocalInvocationId.z = threadIdx.z;
    GlobalInvocationId.x = threadIdx.x + blockDim.x * blockIdx.x;
    GlobalInvocationId.y = threadIdx.y + blockDim.y * blockIdx.y;
    GlobalInvocationId.z = threadIdx.z + blockDim.z * blockIdx.z;
}

__device__ bool __shady_elect_first() {
    unsigned int writemask = __activemask();
    // Find the lowest-numbered active lane
    int elected_lane = __ffs(writemask) - 1;
    return threadIdx.x == __shfl_sync(writemask, threadIdx.x, elected_lane)
        && threadIdx.y == __shfl_sync(writemask, threadIdx.y, elected_lane)
        && threadIdx.z == __shfl_sync(writemask, threadIdx.z, elected_lane);
}

template<typename T>
__device__ T __shady_broadcast_first(T t) {
    unsigned int writemask = __activemask();
    // Find the lowest-numbered active lane
    int elected_lane = __ffs(writemask) - 1;
    return __shfl_sync(writemask, t, elected_lane);
}

__device__ static inline float sign(float f) {
    return copysignf(1.0f, f);
}