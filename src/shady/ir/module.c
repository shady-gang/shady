#include "../ir_private.h"

#include "dict.h"
#include "list.h"
#include "portability.h"

#include <string.h>

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

Module* shd_new_module(IrArena* arena, String name) {
    Module* m = shd_arena_alloc(arena->arena, sizeof(Module));
    *m = (Module) {
        .arena = arena,
        .name = shd_string(arena, name),
        .decls = shd_new_dict(String, const Node*, (HashFn) shd_hash_string, (CmpFn) shd_compare_string),
    };
    shd_list_append(Module*, arena->modules, m);
    return m;
}

IrArena* shd_module_get_arena(const Module* m) {
    return m->arena;
}

String shd_module_get_name(const Module* m) {
    return m->name;
}

Nodes shd_module_get_declarations(const Module* m) {
    size_t count = shd_dict_count(m->decls);
    const Node** alloc = malloc(sizeof(const Node*) * count);

    const Node* def;
    size_t i = 0, j = 0;
    while (shd_dict_iter(m->decls, &i, NULL, &def)) {
        alloc[j++] = def;
    }

    Nodes n = shd_nodes(m->arena, count, alloc);
    free(alloc);
    return n;
}

String shd_get_exported_name(const Node* node) {
    IrArena* a = node->arena;
    const Node* ea = shd_lookup_annotation(node, "Export");
    if (ea) {
        assert(ea->tag == AnnotationValue_TAG);
        AnnotationValue payload = ea->payload.annotation_value;
        return shd_get_string_literal(a, payload.value);
    }
    return NULL;
}

void shd_module_add_export(Module* m, String name, const Node* node) {
    assert(!m->sealed);
    assert(name);
    assert(is_declaration(node));
    const Node* conflict = shd_module_get_declaration(m, name);
    assert((!conflict || conflict == node) && "duplicate export");

    IrArena* a = m->arena;
    String already_exported_name = shd_get_exported_name(node);
    if (already_exported_name) {
        assert((strcmp(already_exported_name, name) == 0) && "exporting a def that is already annotated with a different export name!");
    } else {
        shd_add_annotation(node, annotation_value_helper(a, "Exported", string_lit_helper(a, name)));
    }

    bool def_inserted_ok = shd_dict_insert(String, const Node*, m->decls, name, node);
    assert(def_inserted_ok);
}

const Node* shd_module_get_declaration(const Module* m, String name) {
    const Node** found = shd_dict_find_value(String, const Node*, m->decls, name);
    if (found) return *found;
    return NULL;
}

static Node* make_init_fini_fn(Module* m, String name) {
    IrArena* a = shd_module_get_arena(m);
    Node* fn = function_helper(m, shd_nodes(a, 0, NULL), name, shd_nodes(a, 0, NULL));
    shd_add_annotation_named(fn, "Generated");
    shd_add_annotation_named(fn, "Internal");
    shd_add_annotation_named(fn, "Leaf");
    shd_set_abstraction_body(fn, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_get_abstraction_mem(fn) }));
    return fn;
}

Node* shd_module_get_init_fn(Module* m) {
    const Node* found = shd_module_get_declaration(m, "generated_init");
    if (found)
        return (Node*) found;
    return make_init_fini_fn(m, "generated_init");
}

Node* shd_module_get_fini_fn(Module* m) {
    const Node* found = shd_module_get_declaration(m, "generated_fini");
    if (found)
        return (Node*) found;
    return make_init_fini_fn(m, "generated_fini");
}

void shd_destroy_module(Module* m) {
    shd_destroy_dict(m->decls);
}
