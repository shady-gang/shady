__shared__ uvec3 __shady_make_thread_local(RealGlobalInvocationId);
__shared__ uvec3 __shady_make_thread_local(RealLocalInvocationId);

#define GlobalInvocationId __shady_thread_local_access(RealGlobalInvocationId)
#define LocalInvocationId __shady_thread_local_access(RealLocalInvocationId)

__device__ void __shady_prepare_builtins() {
    LocalInvocationId.arr[0] = threadIdx.x;
    LocalInvocationId.arr[1] = threadIdx.y;
    LocalInvocationId.arr[2] = threadIdx.z;
    GlobalInvocationId.arr[0] = threadIdx.x + blockDim.x * blockIdx.x;
    GlobalInvocationId.arr[1] = threadIdx.y + blockDim.y * blockIdx.y;
    GlobalInvocationId.arr[2] = threadIdx.z + blockDim.z * blockIdx.z;
}
