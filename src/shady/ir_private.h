#ifndef SHADY_IR_PRIVATE_H
#define SHADY_IR_PRIVATE_H

#include "shady/ir.h"
#include "shady/config.h"

#include "arena.h"
#include "growy.h"

#include "stdlib.h"
#include "stdio.h"

struct IrArena_ {
    Arena* arena;
    ArenaConfig config;

    Growy* ids;
    struct List* modules;

    struct Dict* node_set;
    struct Dict* string_set;

    struct Dict* nodes_set;
    struct Dict* strings_set;
};

struct Module_ {
    IrArena* arena;
    String name;
    struct List* decls;
    bool sealed;
};

void _shd_module_add_decl(Module* m, Node* node);
void shd_destroy_module(Module* m);

NodeId _shd_allocate_node_id(IrArena* arena, const Node* n);

IrArena* _shd_get_bb_arena(BodyBuilder* bb);
const Node* _shd_bb_insert_mem(BodyBuilder* bb);
const Node* _shd_bb_insert_block(BodyBuilder* bb);
const Node* _shd_bld_finish_pseudo_instr(BodyBuilder* bb, const Node* terminator);

struct List;
Nodes shd_list_to_nodes(IrArena* arena, struct List* list);

#endif
