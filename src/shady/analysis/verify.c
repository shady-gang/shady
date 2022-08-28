#include "verify.h"

#include "../visit.h"

#include "dict.h"

#include <assert.h>

typedef struct {
    Visitor visitor;
    const IrArena* arena;
    struct Dict* once;
} ArenaVerifyVisitor;

static void visit_verify_same_arena(ArenaVerifyVisitor* visitor, const Node* node) {
    assert(visitor->arena == node->arena);
    if (find_key_dict(const Node*, visitor->once, node))
        return;
    if (node->tag == Function_TAG)
        insert_set_get_result(const Node*, visitor->once, node);
    visit_children(&visitor->visitor, node);
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

static void verify_same_arena(const Node* program) {
    const IrArena* arena = program->arena;
    ArenaVerifyVisitor visitor = {
        .visitor = {
            .visit_fn = (VisitFn) visit_verify_same_arena,
            .visit_fn_scope_rpo = true,
        },
        .arena = arena,
        .once = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
    visit_verify_same_arena(&visitor, program);
    destroy_dict(visitor.once);
}

void verify_program(const Node* program) {
    verify_same_arena(program);
}
