#ifndef SHADY_IR_H
#define SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct IrArena_ IrArena;
typedef struct Node_ Node;
typedef struct Node_ Type;
typedef uint32_t NodeId;
typedef const char* String;

//////////////////////////////// Lists & Strings ////////////////////////////////

typedef struct Nodes_ {
    size_t count;
    const Node** nodes;
} Nodes;

typedef struct Strings_ {
    size_t count;
    String* strings;
} Strings;

Nodes     nodes(IrArena*, size_t count, const Node*[]);
Strings strings(IrArena*, size_t count, const char*[]);

Nodes empty(IrArena*);
Nodes singleton(const Node*);
#define mk_nodes(arena, ...) nodes(arena, sizeof((const Node*[]) { __VA_ARGS__ }) / sizeof(const Node*), (const Node*[]) { __VA_ARGS__ })

const Node* first(Nodes nodes);

Nodes append_nodes(IrArena*, Nodes, const Node*);
Nodes prepend_nodes(IrArena*, Nodes, const Node*);
Nodes concat_nodes(IrArena*, Nodes, Nodes);
Nodes change_node_at_index(IrArena*, Nodes, size_t, const Node*);
bool find_in_nodes(Nodes nodes, const Node* n);

String string_sized(IrArena*, size_t size, const char* start);
String string(IrArena*, const char*);
// see also: format_string in util.h
String format_string_interned(IrArena*, const char* str, ...);
String unique_name(IrArena*, const char* base_name);
String name_type_safe(IrArena*, const Type*);

//////////////////////////////// Modules ////////////////////////////////

typedef struct Module_ Module;

Module* new_module(IrArena*, String name);

IrArena* get_module_arena(const Module*);
String get_module_name(const Module*);
Nodes get_module_declarations(const Module*);
Node* get_declaration(const Module*, String);

void link_module(Module* dst, Module* src);

//////////////////////////////// Grammar ////////////////////////////////

// The bulk of the language grammar is defined through json files.
// We define some support enums here.

typedef enum {
    IntTy8,
    IntTy16,
    IntTy32,
    IntTy64,
} IntSizes;

enum {
    IntSizeMin = IntTy8,
    IntSizeMax = IntTy64,
};

static inline int int_size_in_bytes(IntSizes s) {
    switch (s) {
        case IntTy8: return 1;
        case IntTy16: return 2;
        case IntTy32: return 4;
        case IntTy64: return 8;
    }
}

typedef enum {
    FloatTy16,
    FloatTy32,
    FloatTy64
} FloatSizes;

static inline int float_size_in_bytes(FloatSizes s) {
    switch (s) {
        case FloatTy16: return 2;
        case FloatTy32: return 4;
        case FloatTy64: return 8;
    }
}

#define EXECUTION_MODELS(EM) \
EM(Compute,  1) \
EM(Fragment, 0) \
EM(Vertex,   0) \

typedef enum {
    EmNone,
#define EM(name, _) Em##name,
EXECUTION_MODELS(EM)
#undef EM
} ExecutionModel;

ExecutionModel execution_model_from_string(const char*);

typedef enum {
    NotSpecial,
    /// for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else
    MultipleReturn,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    DecorateBlock
} RecordSpecialFlag;

// see primops.json
#include "primops_generated.h"

String get_primop_name(Op op);
bool has_primop_got_side_effects(Op op);

// see grammar.json
#include "grammar_generated.h"

extern const char* node_tags[];
extern const bool node_type_has_payload[];

//////////////////////////////// Node categories ////////////////////////////////

bool is_nominal(const Node* node);

inline static bool is_function(const Node* node) { return node->tag == Function_TAG; }

//////////////////////////////// IR Arena ////////////////////////////////

/// See config.h for definition of ArenaConfig
typedef struct ArenaConfig_ ArenaConfig;

IrArena* new_ir_arena(const ArenaConfig*);
void destroy_ir_arena(IrArena*);
const ArenaConfig* get_arena_config(const IrArena*);
const Node* get_node_by_id(const IrArena*, NodeId);

//////////////////////////////// Getters ////////////////////////////////

/// Get the name out of a global variable, function or constant
String get_value_name_safe(const Node*);
String get_value_name_unsafe(const Node*);
void set_value_name(const Node* var, String name);

const IntLiteral* resolve_to_int_literal(const Node* node);
int64_t get_int_literal_value(IntLiteral, bool sign_extend);
const FloatLiteral* resolve_to_float_literal(const Node* node);
double get_float_literal_value(FloatLiteral);
const char* get_string_literal(IrArena*, const Node*);

String get_address_space_name(AddressSpace);
/// Returns false iff pointers in that address space can contain different data at the same address
/// (amongst threads in the same subgroup)
bool is_addr_space_uniform(IrArena*, AddressSpace);

const Node* lookup_annotation(const Node* decl, const char* name);
const Node* lookup_annotation_list(Nodes, const char* name);
const Node* get_annotation_value(const Node* annotation);
Nodes get_annotation_values(const Node* annotation);
/// Gets the string literal attached to an annotation, if present.
const char* get_annotation_string_payload(const Node* annotation);
bool lookup_annotation_with_string_payload(const Node* decl, const char* annotation_name, const char* expected_payload);
Nodes filter_out_annotation(IrArena*, Nodes, const char* name);

bool        is_abstraction        (const Node*);
String      get_abstraction_name  (const Node* abs);
String      get_abstraction_name_unsafe(const Node* abs);
String      get_abstraction_name_safe(const Node* abs);
const Node* get_abstraction_body  (const Node* abs);
Nodes       get_abstraction_params(const Node* abs);

void        set_abstraction_body  (Node* abs, const Node* body);

const Node* get_let_instruction(const Node* let);
const Node* get_let_chain_end(const Node* terminator);

const Node* maybe_tuple_helper(IrArena* a, Nodes values);
const Node* extract_helper(const Node* composite, const Node* index);
const Node* extract_multiple_ret_types_helper(const Node* composite, int index);

typedef struct {
    bool enter_loads;
    bool allow_incompatible_types;
    bool assume_globals_immutability;
} NodeResolveConfig;
NodeResolveConfig default_node_resolve_config();
const Node* chase_ptr_to_source(const Node*, NodeResolveConfig config);
const Node* resolve_ptr_to_value(const Node* node, NodeResolveConfig config);

const Node* resolve_node_to_definition(const Node* node, NodeResolveConfig config);

//////////////////////////////// Constructors ////////////////////////////////

/// For typing things that don't return at all
const Type* noret_type(IrArena*);
/// For making pointers to nothing in particular (equivalent to C's void*)
const Node* unit_type(IrArena*);
/// For typing instructions that return nothing (equivalent to C's void f())
const Node* empty_multiple_return_type(IrArena*);

const Type* int_type_helper(IrArena*, bool, IntSizes);

const Type* int8_type(IrArena*);
const Type* int16_type(IrArena*);
const Type* int32_type(IrArena*);
const Type* int64_type(IrArena*);

const Type* uint8_type(IrArena*);
const Type* uint16_type(IrArena*);
const Type* uint32_type(IrArena*);
const Type* uint64_type(IrArena*);

const Type* int8_literal(IrArena*,  int8_t i);
const Type* int16_literal(IrArena*, int16_t i);
const Type* int32_literal(IrArena*, int32_t i);
const Type* int64_literal(IrArena*, int64_t i);

const Type* uint8_literal(IrArena*,  uint8_t i);
const Type* uint16_literal(IrArena*, uint16_t i);
const Type* uint32_literal(IrArena*, uint32_t i);
const Type* uint64_literal(IrArena*, uint64_t i);

const Type* fp16_type(IrArena*);
const Type* fp32_type(IrArena*);
const Type* fp64_type(IrArena*);

const Node* fp_literal_helper(IrArena*, FloatSizes, double);

const Node* type_decl_ref_helper(IrArena*, const Node* decl);

// values
Node* param(IrArena*, const Type* type, const char* name);

const Node* tuple_helper(IrArena*, Nodes contents);
const Node* composite_helper(IrArena*, const Type*, Nodes contents);
const Node* fn_addr_helper(IrArena*, const Node* fn);
const Node* ref_decl_helper(IrArena*, const Node* decl);
const Node* string_lit_helper(IrArena* a, String s);
const Node* annotation_value_helper(IrArena* a, String n, const Node* v);

// instructions
const Node* prim_op_helper(IrArena*, Op, Nodes, Nodes);
const Node* compound_instruction(IrArena* arena, Nodes instructions, Nodes results);

// terminators
const Node* let(IrArena*, const Node* instruction, const Node* tail);
const Node* jump_helper(IrArena* a, const Node* dst, Nodes args);

// decl ctors
Node* function    (Module*, Nodes params, const char* name, Nodes annotations, Nodes return_types);
Node* constant    (Module*, Nodes annotations, const Type*, const char* name);
Node* global_var  (Module*, Nodes annotations, const Type*, String, AddressSpace);
Type* nominal_type(Module*, Nodes annotations, String name);

// basic blocks, lambdas and their helpers
Node* basic_block(IrArena*, Nodes params, const char* name);
Node* case_(IrArena* a, Nodes params);

/// Used to build a chain of let
typedef struct BodyBuilder_ BodyBuilder;
BodyBuilder* begin_body(IrArena*);

/// Appends an instruction to the builder, may apply optimisations.
/// If the arena is typed, returns a list of variables bound to the values yielded by that instruction
Nodes bind_instruction(BodyBuilder*, const Node* instruction);
Nodes bind_instruction_named(BodyBuilder*, const Node* instruction, String const output_names[]);

Nodes gen_if(BodyBuilder*, Nodes, const Node*, const Node*, Node*);
Nodes gen_match(BodyBuilder*, Nodes, const Node*, Nodes, Nodes, Node*);
Nodes gen_loop(BodyBuilder*, Nodes, Nodes, Node*);
Nodes gen_control(BodyBuilder*, Nodes, Node*);

/// Like append bind_instruction, but you explicitly give it information about any yielded values
/// ! In untyped arenas, you need to call this because we can't guess how many things are returned without typing info !
Nodes bind_instruction_outputs_count(BodyBuilder*, const Node* initial_value, size_t outputs_count);

const Node* finish_body(BodyBuilder*, const Node* terminator);
void cancel_body(BodyBuilder*);
const Node* yield_values_and_wrap_in_block_explicit_return_types(BodyBuilder*, Nodes, const Nodes);
const Node* yield_values_and_wrap_in_block(BodyBuilder*, Nodes);
const Node* bind_last_instruction_and_wrap_in_block_explicit_return_types(BodyBuilder*, const Node*, const Nodes);
const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder*, const Node*);

const Node* yield_values_and_wrap_in_compound_instruction_explicit_return_types(BodyBuilder*, Nodes, const Nodes);
const Node* yield_values_and_wrap_in_compound_instruction(BodyBuilder*, Nodes);
const Node* bind_last_instruction_and_wrap_in_compound_instruction_explicit_return_types(BodyBuilder*, const Node*, const Nodes);
const Node* bind_last_instruction_and_wrap_in_compound_instruction(BodyBuilder*, const Node*);

#endif
