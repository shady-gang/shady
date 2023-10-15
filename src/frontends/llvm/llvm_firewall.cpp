#include <dlfcn.h>
#include <assert.h>

static void* object;

static __attribute__((constructor)) void my_init() {
    object = dlmopen(LM_ID_BASE, "libLLVM-14.so", RTLD_LOCAL);
    assert(object);
}

void LLVMGetPointerAddressSpace() {
    void* args =  __builtin_apply_args();
    void* fn = dlsym(object, "LLVMGetPointerAddressSpace");
    assert(fn);
    
}