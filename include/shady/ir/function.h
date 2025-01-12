#ifndef SHADY_IR_FUNCTION_H
#define SHADY_IR_FUNCTION_H

#include "shady/ir/grammar.h"
#include "shady/ir/type.h"

Node* shd_function(Module*, Function);
Node* shd_basic_block(IrArena*, BasicBlock);

static inline Node* function_helper(Module* m, Nodes params, const char* name, Nodes return_types) {
    Node* f = shd_function(m, (Function) {
        .params = params,
        .return_types = return_types,
        .name = name,
    });
    return f;
}

static inline Node* basic_block_helper(IrArena* a, Nodes params, const char* name) {
    return shd_basic_block(a, (BasicBlock) {
        .params = params,
        .name = name,
    });
}

static inline Node* case_(IrArena* a, Nodes params) { return basic_block_helper(a, params, NULL); }

/// For typing instructions that return nothing (equivalent to C's void f())
static inline const Type* empty_multiple_return_type(IrArena* arena) {
    return record_type(arena, (RecordType) {
        .members = shd_empty(arena),
        .special = MultipleReturn,
    });
}

inline static bool is_function(const Node* node) { return node->tag == Function_TAG; }

const Node* shd_get_abstraction_mem(const Node* abs);
String shd_get_abstraction_name(const Node* abs);
String shd_get_abstraction_name_unsafe(const Node* abs);
String shd_get_abstraction_name_safe(const Node* abs);

void shd_set_abstraction_body(Node* abs, const Node* body);

Nodes shd_bld_call(BodyBuilder* bb, const Node* callee, Nodes args);

#endif
