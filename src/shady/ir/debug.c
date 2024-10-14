#include "shady/ir/debug.h"
#include "shady/ir/grammar.h"

#include <string.h>
#include <assert.h>

String shd_get_value_name_unsafe(const Node* v) {
    assert(v && is_value(v));
    if (v->tag == Param_TAG)
        return v->payload.param.name;
    return NULL;
}

String shd_get_value_name_safe(const Node* v) {
    String name = shd_get_value_name_unsafe(v);
    if (name && strlen(name) > 0)
        return name;
    //if (v->tag == Variable_TAG)
    return shd_fmt_string_irarena(v->arena, "%%%d", v->id);
    //return node_tags[v->tag];
}

void shd_set_value_name(const Node* var, String name) {
    // TODO: annotations
    // if (var->tag == Variablez_TAG)
    //     var->payload.varz.name = string(var->arena, name);
}

void shd_bld_comment(BodyBuilder* bb, String str) {
    shd_bld_add_instruction_extract(bb, comment(shd_get_bb_arena(bb), (Comment) { .string = str, .mem = shd_bb_mem(bb) }));
}

void shd_bld_debug_printf(BodyBuilder* bb, String pattern, Nodes args) {
    shd_bld_add_instruction(bb, debug_printf(shd_get_bb_arena(bb), (DebugPrintf) { .string = pattern, .args = args, .mem = shd_bb_mem(bb) }));
}
