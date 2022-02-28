#ifndef SHADY_IR_H

#include <stdbool.h>
#include <stddef.h>

struct IrArena;
struct Node;
struct Type;

#define INSTRUCTIONS() \
  NODEDEF(VariableDecl, var_decl) \
  NODEDEF(ExpressionEval, expr_eval) \
  NODEDEF(Call, call) \

#define NODES() \
  INSTRUCTIONS() \
  NODEDEF(Variable, var) \
  NODEDEF(Function, fn) \

struct Variable {
    const struct Type* type;
    const char* name;
};

struct Variables {
    size_t count;
    const struct Variable** variables;
};

struct Nodes {
    size_t count;
    const struct Node** nodes;
};

/// Function with _structured_ control flow
struct Function {
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
    const struct Node* variable;
    const struct Node* init;
};

struct ExpressionEval {
    const struct Node* expr;
};

struct Call {
    const struct Node* callee;
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

//struct Return {};
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
    FnType
};

struct Types {
    size_t count;
    const struct Type** types;
};

struct Type {
    bool uniform;
    enum TypeTag tag;
    union TypesUnion {
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

struct Nodes         nodes(struct IrArena*, size_t count, struct Node*[]);
struct Variables variables(struct IrArena*, size_t count, struct Variable*[]);
struct Types         types(struct IrArena*, size_t count, struct Type*[]);

#define NODEDEF(struct_name, short_name) const struct Node* short_name(struct IrArena*, struct struct_name);
NODES()
#undef NODEDEF

struct Type* void_type(struct IrArena* arena);
struct Type* int_type(struct IrArena* arena, bool uniform);
struct Type* float_type(struct IrArena* arena, bool uniform);
struct Type* record_type(struct IrArena* arena, char* name, struct Types members);
struct Type* cont_type(struct IrArena* arena, bool uniform, struct Types params);
struct Type* fn_type(struct IrArena* arena, bool uniform, struct Types params, struct Type* return_type);

const char* string(struct IrArena* arena, size_t size, char* start);

#define SHADY_IR_H

#endif
