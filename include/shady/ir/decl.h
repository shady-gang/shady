#ifndef SHADY_IR_DECL_H
#define SHADY_IR_DECL_H

#include "shady/ir/base.h"
#include "shady/ir/grammar.h"
#include "shady/ir/builtin.h"

Node* shd_constant(Module*, Constant payload);
Node* shd_global_var(Module*, GlobalVariable payload);

static inline Node* constant_helper(Module* m, Nodes annotations, const Type* t, const char* name) {
    return shd_constant(m, (Constant) {
        .name = name,
        .annotations = annotations,
        .type_hint = t,
    });
}

static inline Node* global_variable_helper(Module* m, Nodes annotations, const Type* t, String name, AddressSpace as, bool is_ref) {
    return shd_global_var(m, (GlobalVariable) {
        .name = name,
        .annotations = annotations,
        .type = t,
        .address_space = as,
        .is_ref = is_ref
    });
}

typedef struct Rewriter_ Rewriter;
const Node* shd_find_or_process_decl(Rewriter* rewriter, const char* name);

#endif
