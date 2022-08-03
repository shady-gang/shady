#ifndef SHADY_IR_H
#define SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct IrArena_ IrArena;
typedef struct Node_ Node;
typedef struct Node_ Type;
typedef int VarId;
typedef const char* String;

typedef enum DivergenceQualifier_ {
    Unknown,
    Uniform,
    Varying
} DivergenceQualifier;

typedef enum AddressSpace_ {
    AsGeneric,

    // used for lowering various nonsense, does not have a known hardware meaning
    AsSubgroupPhysical,

    AsPrivatePhysical,
    AsSharedPhysical,
    AsGlobalPhysical,

    AsFunctionLogical,
    AsPrivateLogical,
    AsSharedLogical,
    AsGlobalLogical,

    /// special addressing spaces for only global variables
    AsInput,
    AsOutput,
    AsExternal,

    // "fake" address space for function pointers
    AsProgramCode,
} AddressSpace;

static inline bool is_physical_as(AddressSpace as) { return as <= AsGlobalLogical; }

typedef enum EntryPointType_ {
    NotAnEntryPoint,
    Compute,
    Fragment,
    Vertex
} EntryPointType;

//////////////////////////////// Node Types Enumeration ////////////////////////////////
// NODEDEF(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name)

#define TYPE_NODES() \
NODEDEF(1, 0, 0, MaskType, mask_type) \
NODEDEF(1, 0, 0, NoRet, noret_type) \
NODEDEF(1, 0, 0, Unit, unit_type) \
NODEDEF(1, 0, 1, Int, int_type) \
NODEDEF(1, 0, 0, Float, float_type) \
NODEDEF(1, 0, 0, Bool, bool_type) \
NODEDEF(1, 0, 1, RecordType, record_type) \
NODEDEF(1, 0, 1, FnType, fn_type) \
NODEDEF(1, 0, 1, PtrType, ptr_type) \
NODEDEF(1, 1, 1, QualifiedType, qualified_type) \
NODEDEF(1, 0, 1, ArrType, arr_type) \
NODEDEF(1, 0, 1, PackType, pack_type) \

#define VALUE_NODES() \
NODEDEF(0, 1, 1, Variable, var) \
NODEDEF(1, 0, 1, Unbound, unbound) \
NODEDEF(1, 1, 1, UntypedNumber, untyped_number) \
NODEDEF(1, 1, 1, IntLiteral, int_literal) \
NODEDEF(1, 1, 0, True, true_lit) \
NODEDEF(1, 1, 0, False, false_lit) \
NODEDEF(1, 0, 1, StringLiteral, string_lit) \
NODEDEF(0, 1, 1, Tuple, tuple) \
NODEDEF(1, 1, 1, FnAddr, fn_addr) \

#define INSTRUCTION_NODES() \
NODEDEF(0, 1, 1, Let, let) \
NODEDEF(1, 1, 1, PrimOp, prim_op)  \
NODEDEF(1, 1, 1, Call, call_instr)  \
NODEDEF(1, 1, 1, If, if_instr) \
NODEDEF(1, 1, 1, Match, match_instr) \
NODEDEF(1, 1, 1, Loop, loop_instr) \

#define TERMINATOR_NODES() \
NODEDEF(1, 1, 1, Branch, branch) \
NODEDEF(1, 1, 1, Join, join) \
NODEDEF(1, 1, 1, Callc, callc) \
NODEDEF(1, 1, 1, Return, fn_ret) \
NODEDEF(1, 0, 1, MergeConstruct, merge_construct) \
NODEDEF(1, 0, 0, Unreachable, unreachable) \

#define NODES() \
VALUE_NODES() \
INSTRUCTION_NODES() \
TERMINATOR_NODES() \
TYPE_NODES() \
NODEDEF(0, 1, 1, Function, fn) \
NODEDEF(0, 0, 1, Constant, constant) \
NODEDEF(0, 1, 1, GlobalVariable, global_variable) \
NODEDEF(1, 0, 1, Block, block) \
NODEDEF(1, 0, 1, ParsedBlock, parsed_block) \
NODEDEF(1, 0, 1, Annotation, annotation) \
NODEDEF(1, 0, 1, Root, root) \

//////////////////////////////// Lists & Strings ////////////////////////////////

typedef struct Nodes_ {
    size_t count;
    const Node** nodes;
} Nodes;

typedef struct Strings_ {
    size_t count;
    String* strings;
} Strings;

Nodes         nodes(IrArena*, size_t count, const Node*[]);
Strings     strings(IrArena*, size_t count, const char*[]);

Nodes append_nodes(IrArena*, Nodes, const Node*);

String string_sized(IrArena* arena, size_t size, const char* start);
String string(IrArena* arena, const char* start);
String format_string(IrArena* arena, const char* str, ...);
String unique_name(IrArena* arena, const char* start);

//////////////////////////////// Types ////////////////////////////////

bool is_type(const Node*);

typedef struct QualifiedType_ {
    bool is_uniform;
    const Type* type;
} QualifiedType;

typedef struct RecordType_ {
    Nodes members;
    /// Can be empty (no names are given) or has to match the number of members
    Strings names;
    /// Set to 'true' for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else.
    bool must_be_deconstructed;
} RecordType;

typedef struct FnType_ {
    bool is_basic_block;
    Nodes param_types;
    Nodes return_types;
} FnType;

typedef struct PtrType_ {
    AddressSpace address_space;
    const Type* pointed_type;
} PtrType;

typedef struct ArrType_ {
    const Type* element_type;
    const Node* size;
} ArrType;

typedef enum {
    IntTy8,
    IntTy16,
    IntTy32,
    IntTy64,
} IntSizes;

typedef struct Int_ {
    IntSizes width;
} Int;

typedef struct PackType_ {
    const Type* element_type;
    int width;
} PackType;

//////////////////////////////// Values ////////////////////////////////

bool is_value(const Node*);

typedef struct Variable_ {
    const Type* type;
    VarId id;
    String name;

    // Set if this is a let-bound node, otherwise it's NULL and this is a parameter
    const Node* instruction;
    unsigned output;
} Variable;

typedef struct Unbound_ {
    String name;
} Unbound;

typedef struct UntypedNumber_ {
    String plaintext;
} UntypedNumber;

typedef struct IntLiteral_ {
    IntSizes width;
    union {
        int64_t  value_i64;
        int32_t  value_i32;
        int16_t  value_i16;
        int8_t    value_i8;
        uint64_t value_u64;
        uint32_t value_u32;
        uint16_t value_u16;
        uint8_t   value_u8;
    };
} IntLiteral;

int64_t extract_int_literal_value(const Node*, bool sign_extend);

typedef struct StringLiteral_ {
    const char* string;
} StringLiteral;

typedef struct Tuple_ {
    Nodes contents;
} Tuple;

typedef struct FnAddr_ {
    const Node* fn;
} FnAddr;

//////////////////////////////// Other ////////////////////////////////

typedef struct Annotation_ {
    const char* name;
    enum {
        AnPayloadNone,
        AnPayloadValue,
        AnPayloadValues,
        AnPayloadMap,
    } payload_type;
    Strings labels;
    union {
        const Node* value;
        Nodes values;
    };
} Annotation;

const Node*  lookup_annotation(const Node* decl, const char* name);
const Node*  extract_annotation_payload(const Node* annotation);
const Nodes* extract_annotation_payloads(const Node* annotation);
/// Gets the string literal attached to an annotation, if present.
const char*  extract_annotation_string_payload(const Node* annotation);

bool lookup_annotation_with_string_payload(const Node* decl, const char* annotation_name, const char* expected_payload);

typedef struct Function_ {
    Nodes annotations;
    String name;
    bool is_basic_block;
    Nodes params;
    const Node* block;
    Nodes return_types;
} Function;

typedef struct Constant_ {
    Nodes annotations;
    String name;
    const Node* value;
    const Node* type_hint;
} Constant;

typedef struct GlobalVariable_ {
    Nodes annotations;
    const Type* type;
    String name;
    AddressSpace address_space;
    const Node* init;
} GlobalVariable;

/// The body inside functions, continuations, if branches ...
typedef struct Block_ {
    Nodes instructions;
    const Node* terminator;
} Block;

/// used for the front-end to hold continuations before name binding
typedef struct ParsedBlock_ {
    Nodes instructions;
    const Node* terminator;

    Nodes continuations_vars;
    Nodes continuations;
} ParsedBlock;

typedef struct Root_ {
    Nodes declarations;
} Root;

//////////////////////////////// Instructions ////////////////////////////////

bool is_instruction(const Node*);

typedef struct Let_ {
    Nodes variables;
    const Node* instruction;
    bool is_mutable;
} Let;

// PRIMOP(has_side_effects, name)

#define PRIMOPS()                   \
PRIMOP(0, add)                      \
PRIMOP(0, sub)                      \
PRIMOP(0, mul)                      \
PRIMOP(0, div)                      \
PRIMOP(0, mod)                      \
PRIMOP(0, neg)                      \
PRIMOP(0, gt)                       \
PRIMOP(0, gte)                      \
PRIMOP(0, lt)                       \
PRIMOP(0, lte)                      \
PRIMOP(0, eq)                       \
PRIMOP(0, neq)                      \
PRIMOP(0, and)                      \
PRIMOP(0, or)                       \
PRIMOP(0, xor)                      \
PRIMOP(0, not)                      \
PRIMOP(0, rshift_logical)           \
PRIMOP(0, rshift_arithm)            \
PRIMOP(0, lshift)                   \
PRIMOP(1, assign)                   \
PRIMOP(1, subscript)                \
PRIMOP(1, alloca)                   \
PRIMOP(0, load)                     \
PRIMOP(1, store)                    \
PRIMOP(0, lea)                      \
PRIMOP(0, select)                   \
PRIMOP(0, convert)                  \
PRIMOP(0, reinterpret)              \
PRIMOP(0, extract)                  \
PRIMOP(1, push_stack)               \
PRIMOP(1, pop_stack)                \
PRIMOP(1, push_stack_uniform)       \
PRIMOP(1, pop_stack_uniform)        \
PRIMOP(0, subgroup_elect_first)     \
PRIMOP(0, subgroup_broadcast_first) \
PRIMOP(0, subgroup_active_mask)     \
PRIMOP(0, subgroup_ballot)          \
PRIMOP(0, subgroup_local_id)        \
PRIMOP(0, empty_mask)               \
PRIMOP(0, mask_is_thread_active)    \

typedef enum Op_ {
#define PRIMOP(has_side_effects, name) name##_op,
PRIMOPS()
#undef PRIMOP
    PRIMOPS_COUNT
} Op;

extern const char* primop_names[];
bool has_primop_got_side_effects(Op op);

typedef struct PrimOp_ {
    Op op;
    Nodes operands;
} PrimOp;

typedef struct Call_ {
    const Node* callee;
    Nodes args;
} Call;

// Those things are "meta" instructions, they contain other instructions.
// they map to SPIR-V structured control flow constructs directly
// they don't need merge blocks because they are instructions and so that is taken care of by the containing node

/// Structured "if" construct
typedef struct If_ {
    Nodes yield_types;
    const Node* condition;
    const Node* if_true;
    const Node* if_false;
} If;

/// Structured "match" construct
typedef struct Match_ {
    Nodes yield_types;
    const Node* inspect;
    Nodes literals;
    Nodes cases;
    const Node* default_case;
} Match;

/// Structured "loop" construct
typedef struct Loop_ {
    Nodes yield_types;
    Nodes params;
    const Node* body;
    Nodes initial_args;
} Loop;

//////////////////////////////// Terminators ////////////////////////////////

bool is_terminator(const Node*);

/// A branch. Branches can cause divergence, but they can never cause re-convergence.
/// @n @p BrJump is guaranteed to not cause divergence, but all the other forms may cause it.
typedef struct Branch_ {
    bool yield;
    enum {
        /// Uses the @p target field, it must be a value of a function pointer type matching the arguments. It may be varying.
        BrTailcall = 1,
        /// Uses the @p target field, it must point directly to a function, not a function pointer.
        BrJump,
        /// Uses the @p branch_condition and true/false targets, like for @p BrJump, the targets have to point directly to functions
        BrIfElse,
        /// Uses the @p switch_value and default_target, cases_values, case_targets, like for @p BrJump, the targets have to point directly to functions
        /// @todo This is unimplemented at this stage
        BrSwitch
    } branch_mode;
    union {
        const Node* target;
        struct {
            const Node* branch_condition;
            const Node* true_target;
            const Node* false_target;
        };
        struct {
            const Node* switch_value;
            const Node* default_target;
            Nodes case_values;
            Nodes case_targets;
        };
    };
    Nodes args;
} Branch;

/// Join nodes are used to undo the divergence caused by branches. At join nodes, an explicit mask is used to force a number of divergent execution paths to resume.
/// If @p is_indirect is set, the target must be a function pointer. Otherwise, the target must be a function directly.
/// @p join_at _must_ be uniform.
typedef struct Join_ {
    bool is_indirect;
    const Node* join_at;
    Nodes args;
    const Node* desired_mask;
} Join;

typedef struct Return_ {
    // set to NULL after typing
    const Node* fn;
    Nodes values;
} Return;

/// Calls to a function, and mentions the basic block/continuation where execution should resume.
/// NOTE: Since most targets do not allow entering a function from multiple entry points, it is necessary to split functions containing callc.
/// See lower_callc.c
typedef struct Callc_ {
    bool is_return_indirect;
    const Node* ret_cont;
    const Node* callee;
    Nodes args;
} Callc;

extern String merge_what_string[];

/// These terminators are used in conjunction with structured constructs, they go at the end of structured blocks
/// Using those terminators outside of an appropriate structured construct is undefined behaviour, and should probably be validated against
typedef struct MergeConstruct_ {
    enum { Selection, Continue, Break } construct;
    Nodes args;
} MergeConstruct;

//////////////////////////////// Nodes util ////////////////////////////////

extern const char* node_tags[];
extern const bool node_type_has_payload[];

/// Get the name out of a global variable, function or constant
String get_decl_name(const Node*);

typedef enum NodeTag_ {
#define NODEDEF(_, _2, _3, struct_name, short_name) struct_name##_TAG,
NODES()
#undef NODEDEF
} NodeTag;

inline static bool is_nominal(NodeTag tag) {
    return tag == Function_TAG || tag == Root_TAG || tag == Constant_TAG || tag == Variable_TAG || tag == GlobalVariable_TAG;
}

inline static bool is_declaration(NodeTag tag) {
    return tag == Function_TAG || tag == GlobalVariable_TAG || tag == Constant_TAG;
}

const IntLiteral* resolve_to_literal(const Node*);

struct Node_ {
    const IrArena* arena;
    const Type* type;
    NodeTag tag;
    union NodesUnion {
#define NODE_PAYLOAD_1(u, o) u o;
#define NODE_PAYLOAD_0(u, o)
#define NODEDEF(_, _2, has_payload, struct_name, short_name) NODE_PAYLOAD_##has_payload(struct_name, short_name)
        NODES()
#undef NODEDEF
    } payload;
};

// autogenerated node ctors
#define NODE_CTOR_DECL_1(struct_name, short_name) const Node* short_name(IrArena*, struct_name);
#define NODE_CTOR_DECL_0(struct_name, short_name) const Node* short_name(IrArena*);
#define NODE_CTOR_1(has_payload, struct_name, short_name) NODE_CTOR_DECL_##has_payload(struct_name, short_name)
#define NODE_CTOR_0(has_payload, struct_name, short_name)
#define NODEDEF(autogen_ctor, _, has_payload, struct_name, short_name) NODE_CTOR_##autogen_ctor(has_payload, struct_name, short_name)
NODES()
#undef NODEDEF
#undef NODE_CTOR_0
#undef NODE_CTOR_1
#undef NODE_CTOR_DECL_0
#undef NODE_CTOR_DECL_1

const Node* var(IrArena* arena, const Type* type, const char* name);
/// Wraps an instruction and binds the outputs to variables we can use
/// Should not be used if the instruction have no outputs !
const Node* let(IrArena* arena, const Node* instruction, size_t variables_count, const char* variable_names[]);

/// Not meant to be valid IR, useful for the builtin frontend desugaring
const Node* let_mut(IrArena* arena, const Node* instruction, Nodes types, size_t variables_count, const char* variable_names[]);

const Node* tuple(IrArena* arena, Nodes contents);

Node* fn(IrArena*,         Nodes annotations, const char* name, bool, Nodes params, Nodes return_types);
Node* constant(IrArena*,   Nodes annotations, const char* name);
Node* global_var(IrArena*, Nodes annotations, const Type*, String, AddressSpace);

typedef struct BlockBuilder_ BlockBuilder;

BlockBuilder* begin_block(IrArena*);

/// Appends an instruction to the block, and may apply optimisations.
/// If you are interested in the result of one operation, you should obtain it from the return of this function, as it might get optimised out and in such cases this function will account for that
void append_block(BlockBuilder*, const Node* instruction);

void copy_instrs(BlockBuilder*, Nodes);
const Node* finish_block(BlockBuilder*, const Node* terminator);

inline static const Type* int8_type(IrArena* arena)  { return int_type(arena, (Int) { .width = IntTy8  }); }
inline static const Type* int16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16 }); }
inline static const Type* int32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32 }); }
inline static const Type* int64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64 }); }

//////////////////////////////// IR management ////////////////////////////////

typedef struct {
    bool check_types;
    bool allow_fold;
    /// Selects which type the subgroup intrinsic primops use to manipulate masks
    enum {
        /// Uses the MaskType
        SubgroupMaskAbstract,
        /// Uses four packed 32-bit integers
        SubgroupMaskSpvKHRBallot
    } subgroup_mask_representation;
} ArenaConfig;

IrArena* new_arena(ArenaConfig);
void destroy_arena(IrArena*);

typedef struct CompilerConfig_ {
    bool use_loop_for_fn_body;
    bool use_loop_for_fn_calls;
} CompilerConfig;

CompilerConfig default_compiler_config();

typedef enum CompilationResult_ {
    CompilationNoError
} CompilationResult;

CompilationResult run_compiler_passes(CompilerConfig* config, IrArena** arena, const Node** program);
void emit_spirv(CompilerConfig* config, IrArena*, const Node* root, FILE* output);
void dump_cfg(FILE* file, const Node* root);
void print_node(const Node* node);

#endif
