#include "shady/analysis/uses.h"

#include "shady/visit.h"

#include "arena.h"
#include "dict.h"
#include "log.h"

#include "shady/visit.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

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
    Use* use = (Use*) shd_get_first_use(map, n);
    if (!use)
        return NULL;
    while (use->next_use)
        use = (Use*) use->next_use;
    assert(use);
    return use;
}

static void uses_visit_node(UsesMapVisitor* v, const Node* n) {
    if (!shd_dict_find_key(const Node*, v->seen, n)) {
        shd_set_insert(const Node*, v->seen, n);
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
    UsesMap* uses = calloc(1, sizeof(UsesMap));
    *uses = (UsesMap) {
        .map = shd_new_dict(const Node*, Use*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .a = shd_new_arena(),
    };

    UsesMapVisitor v = {
        .v = { .visit_op_fn = (VisitOpFn) uses_visit_op },
        .map = uses,
        .exclude = exclude,
        .seen = shd_new_set(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };
    if (root)
        uses_visit_node(&v, root);
    if (m) {
        Nodes nodes = shd_module_get_all_exported(m);
        for (size_t i = 0; i < nodes.count; i++)
            uses_visit_node(&v, nodes.nodes[i]);
    }
    shd_destroy_dict(v.seen);
    return uses;
}

const UsesMap* shd_new_uses_map_fn(const Node* root, NodeClass exclude) {
    return create_uses_map_(root, NULL, exclude);
}

const UsesMap* shd_new_uses_map_module(const Module* m, NodeClass exclude) {
    return create_uses_map_(NULL, m, exclude);
}

void shd_destroy_uses_map(const UsesMap* map) {
    shd_destroy_arena(map->a);
    shd_destroy_dict(map->map);
    free((void*) map);
}

const Use* shd_get_first_use(const UsesMap* map, const Node* n) {
    const Use** found = shd_dict_find_value(const Node*, const Use*, map->map, n);
    if (found)
        return *found;
    return NULL;
}