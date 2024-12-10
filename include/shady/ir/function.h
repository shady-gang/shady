#ifndef SHADY_IR_FUNCTION_H
#define SHADY_IR_FUNCTION_H

#include "shady/ir/grammar.h"
#include "shady/ir/type.h"

Node* _shd_param(IrArena*, const Type* type, const char* name);
Node* _shd_function(Module*, Nodes params, const char* name, Nodes annotations, Nodes return_types);
Node* _shd_basic_block(IrArena*, Nodes params, const char* name);

static inline Node* param(IrArena* a, const Type* type, const char* name) { return _shd_param(a, type, name); }
static inline Node* function(Module* m, Nodes params, const char* name, Nodes annotations, Nodes return_types) { return _shd_function(m, params, name, annotations, return_types); }
static inline Node* basic_block(IrArena* a, Nodes params, const char* name) { return _shd_basic_block(a, params, name); }
static inline Node* case_(IrArena* a, Nodes params) { return basic_block(a, params, NULL); }

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
