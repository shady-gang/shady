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
    AsInput,
    AsOutput,
    AsGeneric,
    AsPrivate,
    AsShared,
    AsGlobal,

    /// special addressing space for top-level variables
    AsExternal,
} AddressSpace;

typedef enum EntryPointType_ {
    NotAnEntryPoint,
    Compute,
    Fragment,
    Vertex
} EntryPointType;

//////////////////////////////// Node Types Enumeration ////////////////////////////////
// NODEDEF(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name)

#define INSTRUCTION_NODES() \
NODEDEF(1, 0, 1, PrimOp, prim_op)  \
NODEDEF(1, 0, 1, Call, call_instr)  \
NODEDEF(1, 0, 1, If, if_instr) \
NODEDEF(1, 0, 1, Match, match_instr) \
NODEDEF(1, 0, 1, Loop, loop_instr) \

#define TERMINATOR_NODES() \
NODEDEF(1, 1, 1, Return, fn_ret) \
NODEDEF(1, 1, 1, Jump, jump) \
NODEDEF(1, 1, 1, Branch, branch) \
NODEDEF(1, 1, 1, Callc, callc) \
NODEDEF(1, 1, 1, Callf, callf) \
NODEDEF(1, 0, 1, Merge, merge) \
NODEDEF(1, 0, 0, Unreachable, unreachable) \

#define TYPE_NODES() \
NODEDEF(1, 0, 0, NoRet, noret_type) \
NODEDEF(1, 0, 0, Int, int_type) \
NODEDEF(1, 0, 0, Float, float_type) \
NODEDEF(1, 0, 0, Bool, bool_type) \
NODEDEF(1, 0, 1, RecordType, record_type) \
NODEDEF(1, 0, 1, FnType, fn_type) \
NODEDEF(1, 0, 1, PtrType, ptr_type) \
NODEDEF(1, 0, 1, QualifiedType, qualified_type) \

#define NODES() \
INSTRUCTION_NODES() \
TERMINATOR_NODES() \
TYPE_NODES() \
NODEDEF(1, 1, 1, Let, let)  \
NODEDEF(0, 1, 1, Variable, var) \
NODEDEF(1, 0, 1, Unbound, unbound) \
NODEDEF(1, 1, 1, UntypedNumber, untyped_number) \
NODEDEF(1, 1, 1, IntLiteral, int_literal) \
NODEDEF(1, 1, 0, True, true_lit) \
NODEDEF(1, 1, 0, False, false_lit) \
NODEDEF(0, 1, 1, Function, fn) \
NODEDEF(0, 0, 1, Constant, constant) \
NODEDEF(1, 0, 1, Block, block) \
NODEDEF(1, 0, 1, ParsedBlock, parsed_block) \
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

String string_sized(IrArena* arena, size_t size, const char* start);
String string(IrArena* arena, const char* start);
String unique_name(IrArena* arena, const char* start);

//////////////////////////////// Values ////////////////////////////////

typedef struct Variable_ {
    const Type* type;
    VarId id;
    String name;

    // Set if this is a let-bound variable, otherwise it's NULL and this is a parameter
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
    int64_t value;
} IntLiteral;

typedef struct Constant_ {
    String name;
    const Node* value;
    const Node* type_hint;
} Constant;

//////////////////////////////// Functions ////////////////////////////////

typedef struct FnAttributes_ {
    bool is_continuation;
    EntryPointType entry_point_type;
} FnAttributes;

typedef struct Function_ {
    String name;
    FnAttributes atttributes;
    Nodes params;
    const Node* block;
    Nodes return_types;
} Function;

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

typedef struct Let_ {
    Nodes variables;
    const Node* instruction;
} Let;

#define PRIMOPS()          \
PRIMOP(add)                \
PRIMOP(sub)                \
PRIMOP(mul)                \
PRIMOP(div)                \
PRIMOP(mod)                \
PRIMOP(gt)                \
PRIMOP(gte)                \
PRIMOP(lt)                \
PRIMOP(lte)                \
PRIMOP(eq)                \
PRIMOP(neq)                \
PRIMOP(and)                \
PRIMOP(or)                \
PRIMOP(xor)                \
PRIMOP(not)                \
PRIMOP(alloca)             \
PRIMOP(load)               \
PRIMOP(store)              \
PRIMOP(push_stack)         \
PRIMOP(pop_stack)          \
PRIMOP(push_stack_uniform) \
PRIMOP(pop_stack_uniform)  \

typedef enum Op_ {
#define PRIMOP(name) name##_op,
PRIMOPS()
#undef PRIMOP
} Op;

typedef struct PrimOp_ {
    Op op;
    Nodes operands;
} PrimOp;

extern const char* primop_names[];

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

// Structured "match" construct
typedef struct Match_ {
    Nodes yield_types;
    const Node* inspect;
    Nodes literals;
    Nodes cases;
    const Node* default_case;
} Match;

// Structured "loop" construct
typedef struct Loop_ {
    Nodes yield_types;
    Nodes params;
    const Node* body;
    Nodes initial_args;
} Loop;

//////////////////////////////// Terminators ////////////////////////////////

typedef struct Return_ {
    // set to NULL after typing
    const Node* fn;
    Nodes values;
} Return;

typedef struct Jump_ {
    const Node* target;
    Nodes args;
} Jump;

typedef struct Branch_ {
    const Node* condition;
    const Node* true_target;
    const Node* false_target;
    Nodes args;
} Branch;

typedef enum {
    Join,
    Continue,
    Break,
} MergeWhat;

static String merge_what_string[] = { "join", "continue", "break" };

typedef struct Merge_ {
    MergeWhat what;
    Nodes args;
} Merge;

typedef struct Callf_ {
    const Node* ret_fn;
    const Node* callee;
    Nodes args;
} Callf;

typedef struct Callc_ {
    const Node* ret_cont;
    const Node* callee;
    Nodes args;
} Callc;

//////////////////////////////// Types ////////////////////////////////

bool is_type(const Node*);

typedef struct QualifiedType_ {
    bool is_uniform;
    const Type* type;
} QualifiedType;

typedef struct RecordType_ {
    Nodes members;
} RecordType;

typedef struct FnType_ {
    bool is_continuation;
    Nodes param_types;
    Nodes return_types;
} FnType;

typedef struct PtrType_ {
    AddressSpace address_space;
    const Type* pointed_type;
} PtrType;

//////////////////////////////// Nodes util ////////////////////////////////

typedef enum NodeTag_ {
#define NODEDEF(_, _2, _3, struct_name, short_name) struct_name##_TAG,
NODES()
#undef NODEDEF
} NodeTag;

inline static bool is_nominal(NodeTag tag) {
    return tag == Function_TAG || tag == Root_TAG || tag == Constant_TAG;
}

struct Node_ {
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

// Node constructors
#define NODE_CTOR_DECL_1(struct_name, short_name) const Node* short_name(IrArena*, struct_name);
#define NODE_CTOR_DECL_0(struct_name, short_name) const Node* short_name(IrArena*);
#define NODE_CTOR_1(has_payload, struct_name, short_name) NODE_CTOR_DECL_##has_payload(struct_name, short_name)
#define NODE_CTOR_0(has_payload, struct_name, short_name)

// autogenerated ctors
#define NODEDEF(autogen_ctor, _, has_payload, struct_name, short_name) NODE_CTOR_##autogen_ctor(has_payload, struct_name, short_name)
NODES()
#undef NODEDEF
const Node* var(IrArena* arena, const Type* type, const char* name);
const Node* var_with_id(IrArena* arena, const Type* type, const char* name, VarId);
Node* fn(IrArena* arena, FnAttributes, const char* name, Nodes params, Nodes return_types);
Node* constant(IrArena* arena, const char* name);

#undef NODE_CTOR_0
#undef NODE_CTOR_1
#undef NODE_CTOR_DECL_0
#undef NODE_CTOR_DECL_1

extern const char* node_tags[];
extern const bool node_type_has_payload[];

//////////////////////////////// IR management ////////////////////////////////

typedef struct IrConfig_ {
    bool check_types;
} IrConfig;

IrArena* new_arena(IrConfig);
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

//////////////////////////////// IR processing ////////////////////////////////

typedef struct Rewriter_ Rewriter;
typedef const Node* (*RewriteFn)(Rewriter*, const Node*);

/// Applies the rewriter to all nodes in the collection
Nodes rewrite_nodes(Rewriter* rewriter, Nodes old_nodes);

/// bring in a node unmodified into a new arena
const Node* import_node   (IrArena*, const Node*);
Nodes       import_nodes  (IrArena*, Nodes);
Strings     import_strings(IrArena*, Strings);

typedef struct Rewriter_ {
    IrArena* src_arena;
    IrArena* dst_arena;

    RewriteFn rewrite_fn;
} Rewriter_;

const Node* rewrite_node(Rewriter*, const Node*);

/// Rewrites a node using the rewriter to provide the node and type operands
const Node* recreate_node_identity(Rewriter*, const Node*);

/// Rewrites a whole program, starting at the root
typedef const Node* (RewritePass)(IrArena* src_arena, IrArena* dst_arena, const Node* src_root);

typedef struct Visitor_ Visitor;
typedef void (*VisitFn)(Visitor*, const Node*);

struct Visitor_ {
   VisitFn visit_fn;
   // Enabling this will make visit_children build the scope of functions and look at their continuations in RPO
   bool visit_fn_scope_rpo;
   // Enabling this will make visit_children visit targets of control flow terminators, be wary this could cause infinite loops
   bool visit_cf_targets;
   bool visit_return_fn_annotation;
   bool visit_callf_return_fn_annotation;
};

void visit_children(Visitor*, const Node*);
void visit_fn_blocks_except_head(Visitor*, const Node*);

#endif
