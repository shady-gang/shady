#define __shady_make_thread_local(var) var[__shady_workgroup_size]
#define __shady_thread_local_access(var) (var[((threadIdx.x * blockDim.y + threadIdx.y) * blockDim.z + threadIdx.z)])

#define offsetof(T, f) (long long int) &((T*)nullptr)->f