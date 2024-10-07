#ifndef SHADY_IR_FUNCTION_H
#define SHADY_IR_FUNCTION_H

#include "shady/ir/grammar.h"

Node* param(IrArena*, const Type* type, const char* name);

Node* function(Module*, Nodes params, const char* name, Nodes annotations, Nodes return_types);

// basic blocks
Node* basic_block(IrArena*, Nodes params, const char* name);
static inline Node* case_(IrArena* a, Nodes params) {
    return basic_block(a, params, NULL);
}

/// Empty type: there are no values of this type.
/// Useful for the codomain of things that don't return at all
const Type* noret_type(IrArena*);

/// For typing instructions that return nothing (equivalent to C's void f())
const Type* empty_multiple_return_type(IrArena*);

inline static bool is_function(const Node* node) { return node->tag == Function_TAG; }

const Node* shd_get_abstraction_mem(const Node* abs);
String shd_get_abstraction_name(const Node* abs);
String shd_get_abstraction_name_unsafe(const Node* abs);
String shd_get_abstraction_name_safe(const Node* abs);

void set_abstraction_body(Node* abs, const Node* body);

#endif
