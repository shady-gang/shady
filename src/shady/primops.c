#include "shady/ir.h"

#include "primops_generated.c"

bool has_primop_got_side_effects(Op op) {
    return primop_side_effects[op];
}
