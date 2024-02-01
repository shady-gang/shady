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

void register_decl_module(Module* m, Node* node) {
    assert(is_declaration(node));
    assert(!get_declaration(m, get_declaration_name(node)) && "duplicate declaration");
    append_list(Node*, m->decls, node);
}

const Node* get_declaration(const Module* m, String name) {
    Nodes existing_decls = get_module_declarations(m);
    for (size_t i = 0; i < existing_decls.count; i++) {
        if (strcmp(get_declaration_name(existing_decls.nodes[i]), name) == 0)
            return existing_decls.nodes[i];
    }
    return NULL;
}

void destroy_module(Module* m) {
    destroy_list(m->decls);
}
