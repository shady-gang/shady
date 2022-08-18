#include "runtime_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <stdlib.h>

typedef enum { DispatchCompute } DispatchType;

struct Dispatch_ {
    DispatchType type;
    SpecProgram* src;
};

Dispatch* launch_kernel(Program* program, Device* device, int dimx, int dimy, int dimz, int extra_args_count, void** extra_args) {
    assert(extra_args_count == 0 && "TODO");

    Dispatch* dispatch = calloc(1, sizeof(Dispatch));
    dispatch->type = DispatchCompute;
    dispatch->src = get_specialized_program(program, device);

    error("TODO");
    // vkCmdDispatch()
}

bool wait_completion(Dispatch* dispatch) {
    free(dispatch);
}
