#ifndef SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>

struct IrArena;
struct Node;
struct Type;

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

#define NODES() \
  INSTRUCTIONS() \
  NODEDEF(Variable, var) \
  NODEDEF(UntypedNumber, untyped_number) \
  NODEDEF(Function, fn) \
  NODEDEF(PrimOp, primop) \

struct Variables {
    size_t count;
    const struct Variable** variables;
};

struct Nodes {
    size_t count;
    const struct Node** nodes;
};

struct Strings {
    size_t count;
    const char** strings;
};

struct Variable {
    const struct Type* type;
    const char* name;
};

struct UntypedNumber {
    const char* plaintext;
};

/// Function with _structured_ control flow
struct Function {
    const char* name;
    struct Variables params;
    const struct Type* return_type;
    struct Nodes instructions;
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
    struct Variables params;
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
    struct Strings names;
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
            const char* name;
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

struct IrArena* new_arena();
void destroy_arena(struct IrArena*);
struct IrArena* rebuild_arena(struct IrArena*);

struct Nodes         nodes(struct IrArena*, size_t count, const struct Node*[]);
struct Variables variables(struct IrArena*, size_t count, const struct Variable*[]);
struct Types         types(struct IrArena*, size_t count, const struct Type*[]);
struct Strings     strings(struct IrArena*, size_t count, const char*[]);

struct Nodes         reserve_nodes(struct IrArena*, size_t count);
struct Variables reserve_variables(struct IrArena*, size_t count);
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

const char* string(struct IrArena* arena, size_t size, const char* start);

struct Program {
    struct Nodes declarations_and_definitions;
};

void print_program(const struct Program* program);
void print_node(const struct Node* node, bool);
void print_type(const struct Type* type);

#define SHADY_IR_H

#endif
