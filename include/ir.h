#ifndef SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>

struct IrArena;
struct Node;
struct Type;

typedef const char* String;

enum AddressSpace {
    AsGeneric,
    AsPrivate,
    AsShared,
    AsGlobal
};

#define INSTRUCTIONS() \
  NODEDEF(VariableDecl, var_decl) \
  NODEDEF(ExpressionEval, expr_eval) \
  NODEDEF(Call, call) \
  NODEDEF(Let, let)  \
  NODEDEF(Return, fn_ret) \
  NODEDEF(PrimOp, primop) \

#define NODES() \
  INSTRUCTIONS() \
  NODEDEF(Variable, var) \
  NODEDEF(UntypedNumber, untyped_number) \
  NODEDEF(Function, fn) \
  NODEDEF(Root, root) \

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

#define OPS() \
OP(add) \
OP(sub)       \

enum Op {
#define OP(name) name##_op,
OPS()
#undef OP
};

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

//struct Continue {};
//struct Break {};

// types

enum TypeTag {
    NoRet,
    Void,
    Int,
    Float,
    RecordType,
    ContType,
    FnType,
    PtrType,
    QualType,
    NeedsInfer,
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

struct Type {
    enum TypeTag tag;
    union TypesUnion {
        struct QualifiedType {
            bool is_uniform;
            const struct Type* type;
        } qualified;
        struct RecordType {
            String name;
            struct Types members;
        } record;
        struct ContType {
            struct Types param_types;
        } cont;
        struct FnType {
            struct Types param_types;
            const struct Type* return_type;
        } fn;
        struct PtrType {
            struct Type* pointed_type;
            enum AddressSpace address_space;
        } ptr;
    } payload;
};

enum NodeTag {
#define NODEDEF(struct_name, short_name) struct_name##_TAG,
NODES()
#undef NODEDEF
};

struct Node {
    const struct Type* type;
    enum NodeTag tag;
    union NodesUnion {
#define NODEDEF(struct_name, short_name) struct struct_name short_name;
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

    NodeRewriteFn rewrite_node;
    TypeRewriteFn rewrite_type;
};

/// Rewrites a node using the rewriter to provide the node and type operands
const struct Node* recreate_node_identity(struct Rewriter*, const struct Node*);
/// Rewrites a type using the rewriter to provide the type operands
const struct Type* recreate_type_identity(struct Rewriter*, const struct Type*);

/// Rewrites a whole program, starting at the root
typedef const struct Node* (RewritePass)(struct IrArena* src_arena, struct IrArena* dst_arena, const struct Node* src_root);

struct Nodes         nodes(struct IrArena*, size_t count, const struct Node*[]);
struct Types         types(struct IrArena*, size_t count, const struct Type*[]);
struct Strings     strings(struct IrArena*, size_t count, const char*[]);

struct Nodes         reserve_nodes(struct IrArena*, size_t count);
struct Types         reserve_types(struct IrArena*, size_t count);
struct Strings     reserve_strings(struct IrArena*, size_t count);

#define NODEDEF(struct_name, short_name) const struct Node* short_name(struct IrArena*, struct struct_name);
NODES()
#undef NODEDEF

const struct Type* void_type(struct IrArena* arena);
const struct Type* int_type(struct IrArena* arena);
const struct Type* float_type(struct IrArena* arena);
const struct Type* record_type(struct IrArena* arena, const char* name, struct Types members);
const struct Type* cont_type(struct IrArena* arena, struct Types params);
const struct Type* fn_type(struct IrArena* arena, struct Types params, const struct Type* return_type);
const struct Type* ptr_type(struct IrArena* arena, const struct Type* pointed_type, enum AddressSpace);
const struct Type* qualified_type(struct IrArena* arena, bool is_uniform, const struct Type*);

String string_sized(struct IrArena* arena, size_t size, const char* start);
String string(struct IrArena* arena, const char* start);

void print_node(const struct Node* node);
void print_type(const struct Type* type);

#define SHADY_IR_H

#endif
