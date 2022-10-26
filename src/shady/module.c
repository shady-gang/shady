#include "ir_private.h"

#include "list.h"
#include "portability.h"

Module* new_module(IrArena* arena, String name) {
    Module* m = arena_alloc(arena->arena, sizeof(Module));
    *m = (Module) {
        .arena = arena,
        .name = string(arena, name),
        .decls = new_list(Node*),
    };
    return m;
}

IrArena* get_module_arena(const Module* m) {
    return m->arena;
}

String get_module_name(const Module* m) {
    return m->name;
}

Nodes get_module_declarations(const Module* m) {
    size_t count = entries_count_list(m->decls);
    const Node** start = read_list(const Node*, m->decls);
    return nodes(get_module_arena(m), count, start);
}

void register_decl_module(Module* mod, Node* node) {
    assert(is_declaration(node));
    append_list(Node*, mod->decls, node);
}
