#ifndef SHADY_IR_H
#define SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct IrArena_ IrArena;
typedef struct Node_ Node;
typedef struct Node_ Type;
typedef unsigned int VarId;
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
const Node* get_declaration(const Module*, String);

//////////////////////////////// Grammar ////////////////////////////////

// The language grammar is big enough that it deserve its own files

#include "primops.h"
#include "grammar.h"

//////////////////////////////// IR Arena ////////////////////////////////

typedef struct {
    bool name_bound;
    bool check_op_classes;
    bool check_types;
    bool allow_fold;
    bool untyped_ptrs;
    bool validate_builtin_types; // do @Builtins variables need to match their type in builtins.h ?
    bool is_simt;

    bool allow_subgroup_memory;
    bool allow_shared_memory;

    struct {
        /// Selects which type the subgroup intrinsic primops use to manipulate masks
        enum {
            /// Uses the MaskType
            SubgroupMaskAbstract,
            /// Uses a 64-bit integer
            SubgroupMaskInt64
        } subgroup_mask_representation;

        uint32_t subgroup_size;
        uint32_t workgroup_size[3];
    } specializations;

    struct {
        IntSizes ptr_size;
        /// The base type for emulated memory
        IntSizes word_size;
    } memory;

    /// 'folding' optimisations - happen in the constructors directly
    struct {
        bool delete_unreachable_structured_cases;
    } optimisations;
} ArenaConfig;

typedef struct CompilerConfig_ CompilerConfig;
ArenaConfig default_arena_config();

IrArena* new_ir_arena(ArenaConfig);
void destroy_ir_arena(IrArena*);
ArenaConfig get_arena_config(const IrArena*);

//////////////////////////////// Getters ////////////////////////////////

/// Get the name out of a global variable, function or constant
String get_decl_name(const Node*);
String get_value_name(const Node*);
String get_value_name_safe(const Node*);

const Node* get_quoted_value(const Node* instruction);
const IntLiteral* resolve_to_literal(const Node*);

int64_t get_int_literal_value(const Node*, bool sign_extend);
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
String get_annotation_name(const Node* node);
Nodes filter_out_annotation(IrArena*, Nodes, const char* name);

bool        is_abstraction        (const Node*);
String      get_abstraction_name  (const Node* abs);
const Node* get_abstraction_body  (const Node* abs);
Nodes       get_abstraction_params(const Node* abs);

void        set_abstraction_body  (Node* abs, const Node* body);

const Node* get_let_instruction(const Node* let);
const Node* get_let_tail(const Node* let);

//////////////////////////////// Constructors ////////////////////////////////

// type ctor helpers
const Type* noret_type(IrArena*);

const Node* unit_type(IrArena*);

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

const Node* type_decl_ref_helper(IrArena*, const Node* decl);

// values
const Node* var(IrArena*, const Type* type, const char* name);

const Node* tuple_helper(IrArena*, Nodes contents);
const Node* composite_helper(IrArena*, const Type*, Nodes contents);
const Node* fn_addr_helper(IrArena*, const Node* fn);
const Node* ref_decl_helper(IrArena*, const Node* decl);
const Node* string_lit_helper(IrArena* a, String s);
const Node* annotation_value_helper(IrArena* a, String n, const Node* v);

// instructions
/// Turns a value into an 'instruction' (the enclosing let will be folded away later)
/// Useful for local rewrites
const Node* quote_helper(IrArena*, Nodes values);
const Node* prim_op_helper(IrArena*, Op, Nodes, Nodes);

// terminators
const Node* let(IrArena*, const Node* instruction, const Node* tail);
const Node* let_mut(IrArena*, const Node* instruction, const Node* tail);
const Node* jump_helper(IrArena* a, const Node* dst, Nodes args);

// decl ctors
Node* function    (Module*, Nodes params, const char* name, Nodes annotations, Nodes return_types);
Node* constant    (Module*, Nodes annotations, const Type*, const char* name);
Node* global_var  (Module*, Nodes annotations, const Type*, String, AddressSpace);
Type* nominal_type(Module*, Nodes annotations, String name);

// basic blocks, lambdas and their helpers
Node* basic_block(IrArena*, Node* function, Nodes params, const char* name);
const Node* case_(IrArena* a, Nodes params, const Node* body);

/// Used to build a chain of let
typedef struct BodyBuilder_ BodyBuilder;
BodyBuilder* begin_body(IrArena*);

/// Appends an instruction to the builder, may apply optimisations.
/// If the arena is typed, returns a list of variables bound to the values yielded by that instruction
Nodes bind_instruction(BodyBuilder*, const Node* instruction);
Nodes bind_instruction_named(BodyBuilder*, const Node* instruction, String const output_names[]);

/// Like append instruction, but you explicitly give it information about any yielded values
/// ! In untyped arenas, you need to call this because we can't guess how many things are returned without typing info !
Nodes bind_instruction_explicit_result_types(BodyBuilder*, const Node* initial_value, Nodes provided_types, String const output_names[], bool mut);
Nodes bind_instruction_outputs_count(BodyBuilder*, const Node* initial_value, size_t outputs_count, String const output_names[], bool mut);

void bind_variables(BodyBuilder*, Nodes vars, Nodes values);

const Node* finish_body(BodyBuilder*, const Node* terminator);
void cancel_body(BodyBuilder*);
const Node* yield_values_and_wrap_in_block_explicit_return_types(BodyBuilder*, Nodes, const Nodes*);
const Node* yield_values_and_wrap_in_block(BodyBuilder*, Nodes);
const Node* bind_last_instruction_and_wrap_in_block_explicit_return_types(BodyBuilder*, const Node*, const Nodes*);
const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder*, const Node*);

//////////////////////////////// Compilation ////////////////////////////////

struct CompilerConfig_ {
    bool dynamic_scheduling;
    uint32_t per_thread_stack_size;

    struct {
        uint8_t major;
        uint8_t minor;
    } target_spirv_version;

    struct {
        bool emulate_subgroup_ops;
        bool emulate_subgroup_ops_extended_types;
        bool simt_to_explicit_simd;
        bool int64;
        bool decay_ptrs;
    } lower;

    struct {
        bool spv_shuffle_instead_of_broadcast_first;
        bool force_join_point_lifting;
        bool no_physical_global_ptrs;
    } hacks;

    struct {
        struct {
            bool after_every_pass;
            bool delete_unused_instructions;
        } cleanup;
    } optimisations;

    struct {
        bool memory_accesses;
        bool stack_accesses;
        bool god_function;
        bool stack_size;
        bool subgroup_ops;
    } printf_trace;

    struct {
        int max_top_iterations;
    } shader_diagnostics;

    struct {
        bool skip_generated, skip_builtin, skip_internal;
    } logging;

    struct {
        String entry_point;
        ExecutionModel execution_model;
        uint32_t subgroup_size;
    } specialization;

    struct {
        struct { void* uptr; void (*fn)(void*, String, Module*); } after_pass;
    } hooks;
};

CompilerConfig default_compiler_config();

typedef enum CompilationResult_ {
    CompilationNoError
} CompilationResult;

CompilationResult run_compiler_passes(CompilerConfig* config, Module** mod);

//////////////////////////////// Emission ////////////////////////////////

void emit_spirv(CompilerConfig* config, Module*, size_t* output_size, char** output, Module** new_mod);

typedef enum {
    C,
    GLSL,
    ISPC
} CDialect;

typedef struct {
    CDialect dialect;
    bool explicitly_sized_types;
    bool allow_compound_literals;
} CEmitterConfig;

void emit_c(CompilerConfig compiler_config, CEmitterConfig emitter_config, Module*, size_t* output_size, char** output, Module** new_mod);

void dump_cfg(FILE* file, Module*);
void dump_loop_trees(FILE* output, Module* mod);
void dump_module(Module*);
void print_module_into_str(Module*, char** str_ptr, size_t*);
void dump_node(const Node* node);

#endif
