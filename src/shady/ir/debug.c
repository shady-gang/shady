#include "shady/ir/debug.h"
#include "shady/ir/grammar.h"
#include "shady/ir/module.h"
#include "shady/ir/annotation.h"
#include "shady/analysis/literal.h"

#include <string.h>
#include <assert.h>

String shd_get_debug_name(const Node* node) {
    IrArena* a = node->arena;
    const Node* ea = shd_lookup_annotation(node, "Name");
    if (ea) {
        assert(ea->tag == AnnotationValue_TAG);
        AnnotationValue payload = ea->payload.annotation_value;
        return shd_get_string_literal(a, payload.value);
    }
    return NULL;
}

String shd_get_node_name_unsafe(const Node* node) {
    assert(node);
    String exported_name = shd_get_exported_name(node);
    if (exported_name) return exported_name;
    String debug_name = shd_get_debug_name(node);
    if (debug_name) return debug_name;
    if (node->tag == Param_TAG)
        return node->payload.param.name;
    return NULL;
}

String shd_get_node_name_safe(const Node* v) {
    String name = shd_get_node_name_unsafe(v);
    if (name && strlen(name) > 0)
        return name;
    return shd_fmt_string_irarena(v->arena, "%%%d", v->id);
}

void shd_set_debug_name(const Node* var, String name) {
    if (shd_get_debug_name(var))
        shd_remove_annotation_by_name(var, "Name");
    shd_add_annotation(var, annotation_value_helper(var->arena, "Name", string_lit_helper(var->arena, name)));
}

void shd_bld_comment(BodyBuilder* bb, String str) {
    shd_bld_add_instruction(bb, comment(shd_get_bb_arena(bb), (Comment) { .string = str, .mem = shd_bld_mem(bb) }));
}

void shd_bld_debug_printf(BodyBuilder* bb, String pattern, Nodes args) {
    shd_bld_add_instruction(bb, debug_printf(shd_get_bb_arena(bb), (DebugPrintf) { .string = pattern, .args = args, .mem = shd_bld_mem(bb) }));
}
