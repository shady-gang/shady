#include "ir_private.h"

#include "list.h"
#include "portability.h"

#include <string.h>

Module* new_module(IrArena* arena, String name) {
    Module* m = arena_alloc(arena->arena, sizeof(Module));
    *m = (Module) {
        .arena = arena,
        .name = string(arena, name),
        .decls = new_list(Node*),
    };
    append_list(Module*, arena->modules, m);
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
    Nodes existing_decls = get_module_declarations(mod);
    for (size_t i = 0; i < existing_decls.count; i++) {
        if (strcmp(get_decl_name(existing_decls.nodes[i]), get_decl_name(node)) == 0)
            assert(false);
    }
    append_list(Node*, mod->decls, node);
}

void destroy_module(Module* m) {
    destroy_list(m->decls);
}
