#include "shady/ir.h"

#include "portability.h"
#include "log.h"
#include <assert.h>

#include "primops_generated.c"

String shd_get_primop_name(Op op) {
    return primop_names[op];
}

