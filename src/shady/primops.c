#include "shady/ir.h"

#include "primops_generated.c"

String get_primop_name(Op op) {
    return primop_names[op];
}

bool has_primop_got_side_effects(Op op) {
    return primop_side_effects[op];
}
