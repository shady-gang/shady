#ifndef SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>

typedef struct IrArena_ IrArena;

struct Node_;
typedef struct Node_ Node;
typedef struct Node_ Type;

typedef const char* String;

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

typedef int VarId;

// NODEDEF(has_type_check_fn, has_payload, StructName, short_name)

#define INSTRUCTION_NODES() \
NODEDEF(1, 1, 1, VariableDecl, var_decl) \
NODEDEF(1, 1, 1, ExpressionEval, expr_eval) \
NODEDEF(1, 1, 1, Call, call) \
NODEDEF(1, 1, 1, Let, let)  \
NODEDEF(1, 1, 1, Return, fn_ret) \
NODEDEF(1, 1, 1, PrimOp, primop) \

#define TYPE_NODES() \
NODEDEF(1, 0, 0, NoRet, noret_type) \
NODEDEF(1, 0, 0, Int, int_type) \
NODEDEF(1, 0, 0, Float, float_type) \
NODEDEF(1, 0, 1, RecordType, record_type) \
NODEDEF(1, 0, 1, FnType, fn_type) \
NODEDEF(1, 0, 1, ContType, cont_type) \
NODEDEF(1, 0, 1, PtrType, ptr_type) \
NODEDEF(1, 0, 1, QualifiedType, qualified_type) \

#define NODES() \
INSTRUCTION_NODES() \
TYPE_NODES() \
NODEDEF(0, 1, 1, Variable, var) \
NODEDEF(1, 0, 1, Unbound, unbound) \
NODEDEF(1, 1, 1, UntypedNumber, untyped_number) \
NODEDEF(1, 1, 1, IntLiteral, int_literal) \
NODEDEF(1, 1, 1, Function, fn) \
NODEDEF(1, 1, 1, Root, root) \

typedef struct Nodes_ {
    size_t count;
    const Node** nodes;
} Nodes;

typedef struct Strings_ {
    size_t count;
    String* strings;
} Strings;

typedef struct Variable_ {
    const Type* type;
    VarId id;
    String name;
} Variable;

typedef struct Unbound_ {
    String name;
} Unbound;

typedef struct UntypedNumber_ {
    String plaintext;
} UntypedNumber;

typedef struct IntLiteral_ {
    int value;
} IntLiteral;

/// Function with _structured_ control flow
typedef struct Function_ {
    Nodes params;
    Nodes instructions;
    Nodes return_types;
} Function;

typedef enum TerminatorTag_ {
    Jump, Branch, Die
} TerminatorTag;

typedef struct Terminator_ {
    TerminatorTag tag;
    union {
        struct Jump {
            const Node* target;
            const Nodes args;
        } jump;
        //struct Branch {} branch;
    } payload;
} Terminator;

typedef struct Continuation_ {
    Nodes params;
    Nodes instructions;
    Terminator terminator;
} Continuation;

// Nodes

typedef struct VariableDecl_ {
    AddressSpace address_space;
    const Node* variable;
    const Node* init;
} VariableDecl;

typedef struct Let_ {
    Nodes variables;
    const Node* target;
} Let;

typedef struct ExpressionEval_ {
    const Node* expr;
} ExpressionEval;

typedef struct Call_ {
    const Node* callee;
    Nodes args;
} Call;

#define PRIMOPS() \
PRIMOP(add) \
PRIMOP(sub)       \

typedef enum Op_ {
#define PRIMOP(name) name##_op,
PRIMOPS()
#undef PRIMOP
} Op;

extern const char* primop_names[];

typedef struct PrimOp_ {
    Op op;
    Nodes args;
} PrimOp;

// Those things are "meta" instructions, they contain other instructions.
// they map to SPIR-V structured control flow constructs directly
// they don't need merge blocks because they are instructions and so that is taken care of by the containing node

typedef struct StructuredSelection_ {
    const Node* condition;
    Nodes ifTrue;
    Nodes ifFalse;
} StructuredSelection;

typedef struct StructuredSwitch_ {
    const Node* condition;
    Nodes ifTrue;
    Nodes ifFalse;
} StructuredSwitch;

typedef struct StructuredLoop_ {
    Node* condition;
    Nodes bodyInstructions;
} StructuredLoop;

typedef struct Return_ {
    Nodes values;
} Return;

typedef struct Root_ {
    Nodes variables;
    Nodes definitions;
} Root;

typedef enum DivergenceQualifier_ {
    Unknown,
    Uniform,
    Varying
} DivergenceQualifier;

typedef struct QualifiedType_ {
    bool is_uniform;
    const Type* type;
} QualifiedType;

typedef struct RecordType_ {
    Nodes members;
} RecordType;

typedef struct ContType_ {
    Nodes param_types;
} ContType;

typedef struct FnType_ {
    Nodes param_types;
    Nodes return_types;
} FnType;

typedef struct PtrType_ {
    AddressSpace address_space;
    const Type* pointed_type;
} PtrType;

typedef enum NodeTag_ {
#define NODEDEF(_, _2, _3, struct_name, short_name) struct_name##_TAG,
NODES()
#undef NODEDEF
} NodeTag;

struct Node_ {
    Nodes yields;
    NodeTag tag;
    union NodesUnion {
#define NODE_PAYLOAD_1(u, o) u o;
#define NODE_PAYLOAD_0(u, o)
#define NODEDEF(_, _2, has_payload, struct_name, short_name) NODE_PAYLOAD_##has_payload(struct_name, short_name)
        NODES()
#undef NODEDEF
    } payload;
};

typedef struct IrConfig_ {
    bool check_types;
} IrConfig;

IrArena* new_arena(IrConfig);
void destroy_arena(IrArena*);

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

Nodes         nodes(IrArena*, size_t count, const Node*[]);
Strings     strings(IrArena*, size_t count, const char*[]);

#define NODE_CTOR_DECL_1(struct_name, short_name) const Node* short_name(IrArena*, struct_name);
#define NODE_CTOR_DECL_0(struct_name, short_name) const Node* short_name(IrArena*);

#define NODE_CTOR_1(has_payload, struct_name, short_name) NODE_CTOR_DECL_##has_payload(struct_name, short_name)
#define NODE_CTOR_0(has_payload, struct_name, short_name)

#define NODEDEF(autogen_ctor, _, has_payload, struct_name, short_name) NODE_CTOR_##autogen_ctor(has_payload, struct_name, short_name)
NODES()
#undef NODEDEF
const Node* var(IrArena* arena, const Type* type, const char* name);

bool is_type(const Node*);

String string_sized(IrArena* arena, size_t size, const char* start);
String string(IrArena* arena, const char* start);

void print_node(const Node* node);

#define SHADY_IR_H

#endif
