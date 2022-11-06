#ifndef SHADY_IR_PRIVATE_H
#define SHADY_IR_PRIVATE_H

#include "shady/ir.h"

#include "arena.h"

#include "stdlib.h"
#include "stdio.h"

typedef struct IrArena_ {
    Arena* arena;
    ArenaConfig config;

    VarId next_free_id;
    struct List* modules;

    struct Dict* node_set;
    struct Dict* string_set;

    struct Dict* nodes_set;
    struct Dict* strings_set;
} IrArena_;

struct Module_ {
    IrArena* arena;
    String name;
    struct List* decls;
};

void register_decl_module(Module*, Node*);
void destroy_module(Module* m);

struct BodyBuilder_ {
    Module* module;
    IrArena* arena;
    struct List* stack;
};

VarId fresh_id(IrArena*);

struct List;
Nodes list_to_nodes(IrArena*, struct List*);

typedef enum {
    NotAnEntryPoint,
    Compute,
    Fragment,
    Vertex
} ExecutionModel;

ExecutionModel execution_model_from_string(const char*);

#endif
