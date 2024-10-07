#include "uses.h"

#include "log.h"

#include "shady/visit.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

struct UsesMap_ {
    struct Dict* map;
    Arena* a;
};

typedef struct {
    Visitor v;
    UsesMap* map;
    NodeClass exclude;
    struct Dict* seen;
    const Node* user;
} UsesMapVisitor;

static Use* get_last_use(UsesMap* map, const Node* n) {
    Use* use = (Use*) get_first_use(map, n);
    if (!use)
        return NULL;
    while (use->next_use)
        use = (Use*) use->next_use;
    assert(use);
    return use;
}

static void uses_visit_node(UsesMapVisitor* v, const Node* n) {
    if (!shd_dict_find_key(const Node*, v->seen, n)) {
        shd_set_insert_get_result(const Node*, v->seen, n);
        UsesMapVisitor nv = *v;
        nv.user = n;
        shd_visit_node_operands(&nv.v, v->exclude, n);
    }
}

static void uses_visit_op(UsesMapVisitor* v, NodeClass class, String op_name, const Node* op, size_t i) {
    Use* use = shd_arena_alloc(v->map->a, sizeof(Use));
    memset(use, 0, sizeof(Use));
    *use = (Use) {
        .user = v->user,
        .operand_class = class,
        .operand_name = op_name,
        .operand_index = i,
        .next_use = NULL,
    };

    Use* last_use = get_last_use(v->map, op);
    if (last_use)
        last_use->next_use = use;
    else
        shd_dict_insert(const Node*, const Use*, v->map->map, op, use);

    uses_visit_node(v, op);
}

static const UsesMap* create_uses_map_(const Node* root, const Module* m, NodeClass exclude) {
    UsesMap* uses = calloc(sizeof(UsesMap), 1);
    *uses = (UsesMap) {
        .map = shd_new_dict(const Node*, Use*, (HashFn) hash_node, (CmpFn) compare_node),
        .a = shd_new_arena(),
    };

    UsesMapVisitor v = {
        .v = { .visit_op_fn = (VisitOpFn) uses_visit_op },
        .map = uses,
        .exclude = exclude,
        .seen = shd_new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
    };
    if (root)
        uses_visit_node(&v, root);
    if (m) {
        Nodes nodes = get_module_declarations(m);
        for (size_t i = 0; i < nodes.count; i++)
            uses_visit_node(&v, nodes.nodes[i]);
    }
    shd_destroy_dict(v.seen);
    return uses;
}

const UsesMap* create_fn_uses_map(const Node* root, NodeClass exclude) {
    return create_uses_map_(root, NULL, exclude);
}

const UsesMap* create_module_uses_map(const Module* m, NodeClass exclude) {
    return create_uses_map_(NULL, m, exclude);
}

void destroy_uses_map(const UsesMap* map) {
    shd_destroy_arena(map->a);
    shd_destroy_dict(map->map);
    free((void*) map);
}

const Use* get_first_use(const UsesMap* map, const Node* n) {
    const Use** found = shd_dict_find_value(const Node*, const Use*, map->map, n);
    if (found)
        return *found;
    return NULL;
}