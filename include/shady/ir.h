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

Nodes shd_nodes(IrArena*, size_t count, const Node*[]);
Strings shd_strings(IrArena* arena, size_t count, const char** in_strs);

Nodes shd_empty(IrArena* a);
Nodes shd_singleton(const Node* n);
#define mk_nodes(arena, ...) shd_nodes(arena, sizeof((const Node*[]) { __VA_ARGS__ }) / sizeof(const Node*), (const Node*[]) { __VA_ARGS__ })

const Node* shd_first(Nodes nodes);

Nodes shd_nodes_append(IrArena*, Nodes, const Node*);
Nodes shd_nodes_prepend(IrArena*, Nodes, const Node*);
Nodes shd_concat_nodes(IrArena* arena, Nodes a, Nodes b);
Nodes shd_change_node_at_index(IrArena* arena, Nodes old, size_t i, const Node* n);
bool shd_find_in_nodes(Nodes nodes, const Node* n);

String string_sized(IrArena*, size_t size, const char* start);
String string(IrArena*, const char*);
// see also: format_string in util.h
String shd_fmt_string_irarena(IrArena* arena, const char* str, ...);
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

#ifdef __GNUC__
#define SHADY_DESIGNATED_INIT __attribute__((designated_init))
#else
#define SHADY_DESIGNATED_INIT
#endif

// see primops.json
#include "primops_generated.h"

String get_primop_name(Op op);
bool has_primop_got_side_effects(Op op);

typedef struct BodyBuilder_ BodyBuilder;

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

IrArena* shd_new_ir_arena(const ArenaConfig* config);
void shd_destroy_ir_arena(IrArena* arena);
const ArenaConfig* shd_get_arena_config(const IrArena* a);
const Node* shd_get_node_by_id(const IrArena* a, NodeId id);

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

const Node* get_abstraction_mem(const Node* abs);
String      get_abstraction_name  (const Node* abs);
String      get_abstraction_name_unsafe(const Node* abs);
String      get_abstraction_name_safe(const Node* abs);

void        set_abstraction_body  (Node* abs, const Node* body);

const Node* maybe_tuple_helper(IrArena* a, Nodes values);
const Node* extract_helper(const Node* composite, const Node* index);

const Node* maybe_tuple_helper(IrArena* a, Nodes values);

const Node* get_parent_mem(const Node* mem);
const Node* get_original_mem(const Node* mem);

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

/// Empty type: there are no values of this type.
/// Useful for the codomain of things that don't return at all
const Type* noret_type(IrArena*);
/// Unit type, carries no information (equivalent to C's void)
/// There is exactly one possible value of this type: ()
const Node* unit_type(IrArena*);
/// For typing instructions that return nothing (equivalent to C's void f())
const Node* empty_multiple_return_type(IrArena*);

const Type* shd_int_type_helper(IrArena* a, bool s, IntSizes w);

const Type* shd_int8_type(IrArena* arena);
const Type* shd_int16_type(IrArena* arena);
const Type* shd_int32_type(IrArena* arena);
const Type* shd_int64_type(IrArena* arena);

const Type* shd_uint8_type(IrArena* arena);
const Type* shd_uint16_type(IrArena* arena);
const Type* shd_uint32_type(IrArena* arena);
const Type* shd_uint64_type(IrArena* arena);

const Type* shd_int8_literal(IrArena* arena, int8_t i);
const Type* shd_int16_literal(IrArena* arena, int16_t i);
const Type* shd_int32_literal(IrArena* arena, int32_t i);
const Type* shd_int64_literal(IrArena* arena, int64_t i);

const Type* shd_uint8_literal(IrArena* arena, uint8_t i);
const Type* shd_uint16_literal(IrArena* arena, uint16_t i);
const Type* shd_uint32_literal(IrArena* arena, uint32_t i);
const Type* shd_uint64_literal(IrArena* arena, uint64_t i);

const Type* shd_fp16_type(IrArena* arena);
const Type* shd_fp32_type(IrArena* arena);
const Type* shd_fp64_type(IrArena* arena);

const Node* shd_fp_literal_helper(IrArena* a, FloatSizes size, double value);

// values
Node* param(IrArena*, const Type* type, const char* name);

const Node* tuple_helper(IrArena*, Nodes contents);
const Node* lea_helper(IrArena*, const Node*, const Node*, Nodes);

// decl ctors
Node* function    (Module*, Nodes params, const char* name, Nodes annotations, Nodes return_types);
Node* constant    (Module*, Nodes annotations, const Type*, const char* name);
Node* global_var  (Module*, Nodes annotations, const Type*, String, AddressSpace);
Type* nominal_type(Module*, Nodes annotations, String name);

// basic blocks
Node* basic_block(IrArena*, Nodes params, const char* name);
static inline Node* case_(IrArena* a, Nodes params) {
    return basic_block(a, params, NULL);
}

/// Used to build a chain of let
BodyBuilder* begin_body_with_mem(IrArena*, const Node*);
BodyBuilder* begin_block_pure(IrArena*);
BodyBuilder* begin_block_with_side_effects(IrArena*, const Node*);

/// Appends an instruction to the builder, may apply optimisations.
/// If the arena is typed, returns a list of variables bound to the values yielded by that instruction
Nodes bind_instruction(BodyBuilder*, const Node* instruction);
const Node* bind_instruction_single(BodyBuilder*, const Node* instruction);
Nodes bind_instruction_named(BodyBuilder*, const Node* instruction, String const output_names[]);

Nodes deconstruct_composite(IrArena* a, BodyBuilder* bb, const Node* value, size_t outputs_count);

Nodes gen_if(BodyBuilder*, Nodes, const Node*, const Node*, Node*);
Nodes gen_match(BodyBuilder*, Nodes, const Node*, Nodes, Nodes, Node*);
Nodes gen_loop(BodyBuilder*, Nodes, Nodes, Node*);

typedef struct {
    Nodes results;
    Node* case_;
    const Node* jp;
} begin_control_t;
begin_control_t begin_control(BodyBuilder*, Nodes);

typedef struct {
    Nodes results;
    Node* loop_body;
    Nodes params;
    const Node* continue_jp;
    const Node* break_jp;
} begin_loop_helper_t;
begin_loop_helper_t begin_loop_helper(BodyBuilder*, Nodes, Nodes, Nodes);

Nodes gen_control(BodyBuilder*, Nodes, Node*);

const Node* bb_mem(BodyBuilder*);

/// Like append bind_instruction, but you explicitly give it information about any yielded values
/// ! In untyped arenas, you need to call this because we can't guess how many things are returned without typing info !
Nodes bind_instruction_outputs_count(BodyBuilder*, const Node* initial_value, size_t outputs_count);

const Node* finish_body(BodyBuilder*, const Node* terminator);
const Node* finish_body_with_return(BodyBuilder*, Nodes args);
const Node* finish_body_with_unreachable(BodyBuilder*);
const Node* finish_body_with_selection_merge(BodyBuilder*, Nodes args);
const Node* finish_body_with_loop_continue(BodyBuilder*, Nodes args);
const Node* finish_body_with_loop_break(BodyBuilder*, Nodes args);
const Node* finish_body_with_join(BodyBuilder*, const Node* jp, Nodes args);
const Node* finish_body_with_jump(BodyBuilder*, const Node* target, Nodes args);

void cancel_body(BodyBuilder*);

const Node* yield_value_and_wrap_in_block(BodyBuilder*, const Node*);
const Node* yield_values_and_wrap_in_block(BodyBuilder*, Nodes);
const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder*, const Node*);

const Node* yield_values_and_wrap_in_compound_instruction(BodyBuilder*, Nodes);

#endif
