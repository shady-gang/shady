#ifndef SHADY_IR_PRIVATE_H
#define SHADY_IR_PRIVATE_H

#include "shady/ir.h"
#include "shady/config.h"

#include "arena.h"
#include "growy.h"

#include "stdlib.h"
#include "stdio.h"

typedef struct IrArena_ {
    Arena* arena;
    ArenaConfig config;

    Growy* ids;
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
    bool sealed;
};

void register_decl_module(Module*, Node*);
void destroy_module(Module* m);

struct BodyBuilder_ {
    IrArena* arena;
    struct List* stack;
    const Node* block_entry_mem;
    const Node* mem;
    Node* bb;
};

NodeId allocate_node_id(IrArena*, const Node* n);

struct List;
Nodes list_to_nodes(IrArena*, struct List*);

#endif
