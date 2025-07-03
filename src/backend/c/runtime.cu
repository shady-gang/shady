#define GlobalInvocationId (make_uint3(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y, threadIdx.z + blockIdx.z * blockDim.z))
#define LocalInvocationId (make_uint3(threadIdx.x, threadIdx.y, threadIdx.z))
#define NumWorkgroups (make_uint3(gridDim.x, gridDim.y, gridDim.z))
#define WorkgroupSize (make_uint3(blockDim.x, blockDim.y, blockDim.z))
#define SubgroupSize (warpSize)
#define SubgroupLocalInvocationId (threadIdx.x % warpSize)
#define LinearInvocationId ((((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x) + threadIdx.x)
#define SubgroupId (LinearInvocationId / warpSize)

__device__ void __shady_cuda_init() {
	/* do nothing */
}

__device__ bool __shady_elect_first() {
    unsigned int writemask = __activemask();
	unsigned int id = SubgroupLocalInvocationId;
	unsigned int minid = __reduce_min_sync(writemask, id);
	//printf("__shady_elect_first(), mask=%x, id = %d, minid = %d\\n", writemask, id, minid);
	return id == minid;
    // Find the lowest-numbered active lane
    //int elected_lane = __ffs(writemask) - 1;
    //return threadIdx.x == __shfl_sync(writemask, threadIdx.x, elected_lane)
    //    && threadIdx.y == __shfl_sync(writemask, threadIdx.y, elected_lane)
    //    && threadIdx.z == __shfl_sync(writemask, threadIdx.z, elected_lane);
}

__device__ unsigned __shady_ballot(bool pred) {
    return __ballot_sync(__activemask(), pred);
}

__device__ unsigned __shady_iadd_reduce(unsigned v) {
    return __reduce_add_sync(__activemask(), v);
}

__device__ int __shady_iadd_reduce(int v) {
    return __reduce_add_sync(__activemask(), v);
}

template<typename T>
__device__ T __shady_broadcast_first(T t) {
    unsigned int writemask = __activemask();
    // Find the lowest-numbered active lane
    int elected_lane = __ffs(writemask) - 1;
    return __shfl_sync(writemask, t, elected_lane);
}

__device__ float fract(float x) {
	return x - floorf(x);
}

__device__ static inline float sign(float f) {
    return copysignf(1.0f, f);
}