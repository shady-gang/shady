#ifndef SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>

struct IrArena;
struct Node;

#define Type Node

typedef const char* String;

enum AddressSpace {
    AsGeneric,
    AsPrivate,
    AsShared,
    AsGlobal
};

#define INSTRUCTION_NODES() \
  NODEDEF(true, true, VariableDecl, var_decl) \
  NODEDEF(true, true, ExpressionEval, expr_eval) \
  NODEDEF(true, true, Call, call) \
  NODEDEF(true, true, Let, let)  \
  NODEDEF(true, true, Return, fn_ret) \
  NODEDEF(true, true, PrimOp, primop) \

#define TYPE_NODES() \
NODEDEF(false, false, NoRet, noret_type) \
NODEDEF(false, false, Void, void_type) \
NODEDEF(false, false, Int, int_type) \
NODEDEF(false, false, Float, float_type) \
NODEDEF(false, true, RecordType, record_type) \
NODEDEF(false, true, FnType, fn_type) \
NODEDEF(false, true, ContType, cont_type) \
NODEDEF(false, true, PtrType, ptr_type) \
NODEDEF(false, true, QualifiedType, qualified_type) \

#define NODES() \
  INSTRUCTION_NODES() \
  TYPE_NODES() \
  NODEDEF(true, true, Variable, var) \
  NODEDEF(true, true, UntypedNumber, untyped_number) \
  NODEDEF(true, true, Function, fn) \
  NODEDEF(true, true, Root, root) \

struct Nodes {
    size_t count;
    const struct Node** nodes;
};

struct Strings {
    size_t count;
    String* strings;
};

struct Variable {
    const struct Type* type;
    String name;
};

struct UntypedNumber {
    String plaintext;
};

/// Function with _structured_ control flow
struct Function {
    struct Nodes params;
    struct Nodes instructions;
    const struct Type* return_type;
};

struct Continuation;

enum TerminatorTag {
    Jump, Branch, Die
};

struct Terminator {
    enum TerminatorTag tag;
    union {
        struct Jump {
            const struct Continuation* target;
            const struct Nodes args;
        } jump;
        //struct Branch {} branch;
    } payload;
};

struct Continuation {
    struct Nodes params;
    struct Nodes instructions;
    struct Terminator terminator;
};

// Nodes

struct VariableDecl {
    enum AddressSpace address_space;
    const struct Node* variable;
    const struct Node* init;
};

struct Let {
    struct Nodes variables;
    const struct Node* target;
};

struct ExpressionEval {
    const struct Node* expr;
};

struct Call {
    const struct Node* callee;
    struct Nodes args;
};

#define PRIMOPS() \
PRIMOP(add) \
PRIMOP(sub)       \

enum Op {
#define PRIMOP(name) name##_op,
PRIMOPS()
#undef PRIMOP
};

extern const char* primop_names[];

struct PrimOp {
    enum Op op;
    struct Nodes args;
};

// Those things are "meta" instructions, they contain other instructions.
// they map to SPIR-V structured control flow constructs directly
// they don't need merge blocks because they are instructions and so that is taken care of by the containing node

struct StructuredSelection {
    const struct Node* condition;
    struct Nodes ifTrue;
    struct Nodes ifFalse;
};

struct StructuredSwitch {
    const struct Node* condition;
    struct Nodes ifTrue;
    struct Nodes ifFalse;
};

struct StructuredLoop {
    struct Node* condition;
    struct Nodes bodyInstructions;
};

struct Return {
    struct Nodes values;
};

struct Root {
    struct Nodes variables;
    struct Nodes definitions;
};

struct Types {
    size_t count;
    const struct Type** types;
};

enum DivergenceQualifier {
    Unknown,
    Uniform,
    Varying
};

struct QualifiedType {
    bool is_uniform;
    const struct Type* type;
};

struct RecordType {
    String name;
    struct Types members;
};

struct ContType {
    struct Types param_types;
};

struct FnType {
    struct Types param_types;
    const struct Type* return_type;
};

struct PtrType {
    enum AddressSpace address_space;
    const struct Type* pointed_type;
};

enum NodeTag {
#define NODEDEF(_, _2, struct_name, short_name) struct_name##_TAG,
NODES()
#undef NODEDEF
};

struct Node {
    const struct Type* type;
    enum NodeTag tag;
    union NodesUnion {
#define NODE_PAYLOAD_true(u, o) struct u o;
#define NODE_PAYLOAD_false(u, o)
#define NODEDEF(_, has_payload, struct_name, short_name) NODE_PAYLOAD_##has_payload(struct_name, short_name)
        NODES()
#undef NODEDEF
    } payload;
};

struct IrConfig {
    bool check_types;
};

struct IrArena* new_arena(struct IrConfig);
void destroy_arena(struct IrArena*);

struct Rewriter;

typedef struct Node* (*NodeRewriteFn)(struct Rewriter*, const struct Node*);
typedef struct Type* (*TypeRewriteFn)(struct Rewriter*, const struct Type*);

/// Applies the rewriter to all nodes in the collection
struct Nodes rewrite_nodes(struct Rewriter* rewriter, struct Nodes old_nodes);
/// Applies the rewriter to all types in the collection
struct Types rewrite_types(struct Rewriter* rewriter, struct Types old_types);

struct Strings import_strings(struct Rewriter* rewriter, struct Strings old_strings);

struct Rewriter {
    struct IrArena* src_arena;
    struct IrArena* dst_arena;

    NodeRewriteFn rewrite_node_fn;
    TypeRewriteFn rewrite_type_fn;
};

const struct Node* rewrite_node(struct Rewriter*, const struct Node*);
const struct Type* rewrite_type(struct Rewriter*, const struct Type*);

/// Rewrites a node using the rewriter to provide the node and type operands
const struct Node* recreate_node_identity(struct Rewriter*, const struct Node*);
/// Rewrites a type using the rewriter to provide the type operands
const struct Type* recreate_type_identity(struct Rewriter*, const struct Type*);

/// Rewrites a whole program, starting at the root
typedef const struct Node* (RewritePass)(struct IrArena* src_arena, struct IrArena* dst_arena, const struct Node* src_root);

struct Nodes         nodes(struct IrArena*, size_t count, const struct Node*[]);
struct Types         types(struct IrArena*, size_t count, const struct Type*[]);
struct Strings     strings(struct IrArena*, size_t count, const char*[]);

#define NODE_CTOR_true(struct_name, short_name) const struct Node* short_name(struct IrArena*, struct struct_name);
#define NODE_CTOR_false(struct_name, short_name) const struct Node* short_name(struct IrArena*);
#define NODEDEF(_, has_payload, struct_name, short_name) NODE_CTOR_##has_payload(struct_name, short_name)
NODES()
#undef NODEDEF

String string_sized(struct IrArena* arena, size_t size, const char* start);
String string(struct IrArena* arena, const char* start);

void print_node(const struct Node* node);

#define SHADY_IR_H

#endif
