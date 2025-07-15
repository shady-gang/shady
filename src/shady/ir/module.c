#include "../ir_private.h"

#include "shady/visit.h"

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

static Nodes dict2nodes(IrArena* a, struct Dict* d, bool values) {
    size_t count = shd_dict_count(d);
    const Node** alloc = malloc(sizeof(const Node*) * count);

    const Node* def;
    size_t i = 0, j = 0;
    while (values ? shd_dict_iter(d, &i, NULL, &def) : shd_dict_iter(d, &i, &def, NULL)) {
        alloc[j++] = def;
    }

    Nodes n = shd_nodes(a, count, alloc);
    free(alloc);
    return n;
}

Nodes shd_module_get_all_exported(const Module* m) {
    return dict2nodes(m->arena, m->decls, true);
}

KeyHash shd_hash_node(const Node**);
bool shd_compare_node(const Node**, const Node**);

typedef struct {
    Visitor v;
    NodeTag tag;
    struct Dict* set;
    struct Dict* seen;
} VisitorCtx;

static void visit_node(VisitorCtx* ctx, const Node* n) {
    if (shd_dict_find_key(const Node*, ctx->seen, n))
        return;
    shd_set_insert(const Node*, ctx->seen, n);
    if (n->tag == ctx->tag)
        shd_set_insert(const Node*, ctx->set, n);
    shd_visit_node_operands(&ctx->v, 0, n);
}

static Nodes collect_nodes_by_tag(const Module* m, NodeTag tag) {
    VisitorCtx ctx = {
        .v = {
            .visit_node_fn = (VisitNodeFn) visit_node
        },
        .tag = tag,
    };
    ctx.set = shd_new_set(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);
    ctx.seen = shd_new_set(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);

    shd_visit_module(&ctx.v, m);

    Nodes n = dict2nodes(m->arena, ctx.set, false);

    shd_destroy_dict(ctx.set);
    shd_destroy_dict(ctx.seen);
    return n;
}

Nodes shd_module_collect_reachable_globals(const Module* m) {
    return collect_nodes_by_tag(m, GlobalVariable_TAG);
}

Nodes shd_module_collect_reachable_functions(const Module* m) {
    return collect_nodes_by_tag(m, Function_TAG);
}

String shd_get_exported_name(const Node* node) {
    IrArena* a = node->arena;
    const Node* ea = shd_lookup_annotation(node, "Exported");
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
    const Node* conflict = shd_module_get_exported(m, name);
    assert((!conflict || conflict == node) && "duplicate export");
    if (conflict == node)
        return;

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

const Node* shd_module_get_exported(const Module* m, String name) {
    const Node** found = shd_dict_find_value(String, const Node*, m->decls, name);
    if (found) return *found;
    return NULL;
}

static Node* make_init_fini_fn(Module* m, String name) {
    IrArena* a = shd_module_get_arena(m);
    Node* fn = function_helper(m, shd_nodes(a, 0, NULL), shd_nodes(a, 0, NULL));
    shd_set_debug_name(fn, name);
    shd_add_annotation_named(fn, "Generated");
    shd_add_annotation_named(fn, "Leaf");
    shd_set_abstraction_body(fn, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_get_abstraction_mem(fn) }));
    shd_module_add_export(m, name, fn);
    return fn;
}

Node* shd_module_get_init_fn(Module* m) {
    const Node* found = shd_module_get_exported(m, "generated_init");
    if (found)
        return (Node*) found;
    return make_init_fini_fn(m, "generated_init");
}

Node* shd_module_get_fini_fn(Module* m) {
    const Node* found = shd_module_get_exported(m, "generated_fini");
    if (found)
        return (Node*) found;
    return make_init_fini_fn(m, "generated_fini");
}

void shd_destroy_module(Module* m) {
    shd_destroy_dict(m->decls);
}
