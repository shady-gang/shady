#ifndef SHADY_IR_H
#define SHADY_IR_H

#include "shady/ir/base.h"
#include "shady/ir/module.h"

//////////////////////////////// Grammar ////////////////////////////////

// The bulk of the language grammar is defined through json files.
// We define some support enums here.

#include "shady/ir/int.h"
#include "shady/ir/float.h"
#include "shady/ir/execution_model.h"
#include "shady/ir/primop.h"
#include "shady/ir/grammar.h"

//////////////////////////////// Node categories ////////////////////////////////

bool is_nominal(const Node* node);

inline static bool is_function(const Node* node) { return node->tag == Function_TAG; }

//////////////////////////////// IR Arena ////////////////////////////////

#include "shady/ir/arena.h"

//////////////////////////////// Getters ////////////////////////////////

/// Get the name out of a global variable, function or constant
String get_value_name_safe(const Node*);
String get_value_name_unsafe(const Node*);
void set_value_name(const Node* var, String name);

String get_address_space_name(AddressSpace);
/// Returns false iff pointers in that address space can contain different data at the same address
/// (amongst threads in the same subgroup)
bool is_addr_space_uniform(IrArena*, AddressSpace);

#include "shady/ir/annotation.h"

const Node* maybe_tuple_helper(IrArena* a, Nodes values);
const Node* extract_helper(const Node* composite, const Node* index);

const Node* maybe_tuple_helper(IrArena* a, Nodes values);

const Node* tuple_helper(IrArena*, Nodes contents);
const Node* lea_helper(IrArena*, const Node*, const Node*, Nodes);

#include "shady/ir/mem.h"
#include "shady/ir/type.h"

const IntLiteral* resolve_to_int_literal(const Node* node);
int64_t get_int_literal_value(IntLiteral, bool sign_extend);
const FloatLiteral* resolve_to_float_literal(const Node* node);
double get_float_literal_value(FloatLiteral);
const char* get_string_literal(IrArena*, const Node*);

typedef struct {
    bool enter_loads;
    bool allow_incompatible_types;
    bool assume_globals_immutability;
} NodeResolveConfig;
NodeResolveConfig default_node_resolve_config(void);
const Node* chase_ptr_to_source(const Node*, NodeResolveConfig config);
const Node* resolve_ptr_to_value(const Node* node, NodeResolveConfig config);

const Node* resolve_node_to_definition(const Node* node, NodeResolveConfig config);

//////////////////////////////// Constructors ////////////////////////////////

/// Empty type: there are no values of this type.
/// Useful for the codomain of things that don't return at all
const Type* noret_type(IrArena*);
/// For typing instructions that return nothing (equivalent to C's void f())
const Node* empty_multiple_return_type(IrArena*);

Type* nominal_type(Module*, Nodes annotations, String name);

// values
Node* param(IrArena*, const Type* type, const char* name);

// decl ctors
Node* function    (Module*, Nodes params, const char* name, Nodes annotations, Nodes return_types);
Node* constant    (Module*, Nodes annotations, const Type*, const char* name);
Node* global_var  (Module*, Nodes annotations, const Type*, String, AddressSpace);

const Node* get_abstraction_mem(const Node* abs);
String      get_abstraction_name  (const Node* abs);
String      get_abstraction_name_unsafe(const Node* abs);
String      get_abstraction_name_safe(const Node* abs);

void        set_abstraction_body  (Node* abs, const Node* body);

// basic blocks
Node* basic_block(IrArena*, Nodes params, const char* name);
static inline Node* case_(IrArena* a, Nodes params) {
    return basic_block(a, params, NULL);
}

#include "shady/body_builder.h"

#endif
