#ifndef SHADY_IR_DECL_H
#define SHADY_IR_DECL_H

#include "shady/ir/base.h"
#include "shady/ir/grammar.h"
#include "shady/ir/builtin.h"
#include "shady/ir/module.h"
#include "shady/ir/arena.h"

Node* shd_constant(Module*, Constant payload);
Node* shd_global_var(Module*, GlobalVariable payload);

static inline Node* constant_helper(Module* m, const Type* t) {
    Node* c = shd_constant(m, (Constant) {
        .type_hint = t,
    });
    return c;
}

static inline Node* global_variable_helper(Module* m, const Type* t, AddressSpace as) {
    Node* g = shd_global_var(m, (GlobalVariable) {
        .type = t,
        .address_space = as,
        .is_ref = !shd_ir_arena_get_config(shd_module_get_arena(m))->target.memory.address_spaces[as].physical,
    });
    return g;
}

/// temporary to ease refactoring
static inline NodeTag is_declaration(const Node* n) { switch (n->tag) {
    case Function_TAG:
    case GlobalVariable_TAG:
    case NominalType_TAG:
    case Constant_TAG: return n->tag;
    default: return InvalidNode_TAG;
}}

typedef struct Rewriter_ Rewriter;
const Node* shd_find_or_process_decl(Rewriter* rewriter, const char* name);

#endif
