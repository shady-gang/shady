#include "uses.h"

#include "log.h"

#include "../visit.h"

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

static void uses_visit_op(UsesMapVisitor* v, NodeClass class, String op_name, const Node* op) {
    Use* use = arena_alloc(v->map->a, sizeof(Use));
    memset(use, 0, sizeof(Use));
    *use = (Use) {
        .user = v->user,
        .operand_class = class,
        .operand_name = op_name,
        .next_use = NULL
    };

    Use* last_use = get_last_use(v->map, op);
    if (last_use)
        last_use->next_use = use;
    else
        insert_dict(const Node*, const Use*, v->map->map, op, use);

    if (!find_key_dict(const Node*, v->seen, op)) {
        insert_set_get_result(const Node*, v->seen, op);
        UsesMapVisitor nv = *v;
        nv.user = op;
        visit_node_operands(&nv.v, v->exclude, op);
    }
}

const UsesMap* create_uses_map(const Node* root, NodeClass exclude) {
    UsesMap* uses = calloc(sizeof(UsesMap), 1);
    *uses = (UsesMap) {
        .map = new_dict(const Node*, Use*, (HashFn) hash_node, (CmpFn) compare_node),
        .a = new_arena(),
    };

    UsesMapVisitor v = {
        .v = { .visit_op_fn = (VisitOpFn) uses_visit_op },
        .map = uses,
        .exclude = exclude,
        .seen = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .user = root,
    };
    insert_set_get_result(const Node*, v.seen, root);
    visit_node_operands(&v.v, exclude, root);
    destroy_dict(v.seen);
    return uses;
}

void destroy_uses_map(const UsesMap* map) {
    destroy_arena(map->a);
    destroy_dict(map->map);
    free((void*) map);
}

const Use* get_first_use(const UsesMap* map, const Node* n) {
    const Use** found = find_value_dict(const Node*, const Use*, map->map, n);
    if (found)
        return *found;
    return NULL;
}