#ifndef SHADY_IR_DECL_H
#define SHADY_IR_DECL_H

#include "shady/ir/base.h"
#include "shady/ir/grammar.h"
#include "shady/ir/builtin.h"
#include "shady/ir/module.h"
#include "shady/ir/arena.h"

Node* shd_constant(Module*, Constant payload);
Node* shd_global_var(Module*, GlobalVariable payload);

static inline Node* constant_helper(Module* m, const Type* t, const char* name) {
    Node* c = shd_constant(m, (Constant) {
        .name = name,
        .type_hint = t,
    });
    return c;
}

static inline Node* global_variable_helper(Module* m, const Type* t, String name, AddressSpace as) {
    Node* g = shd_global_var(m, (GlobalVariable) {
        .name = name,
        .type = t,
        .address_space = as,
        .is_ref = !shd_ir_arena_get_config(shd_module_get_arena(m))->target.address_spaces[as].physical,
    });
    return g;
}

typedef struct Rewriter_ Rewriter;
const Node* shd_find_or_process_decl(Rewriter* rewriter, const char* name);

#endif
