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
    if (node->tag == Lambda_TAG)
        insert_set_get_result(const Node*, visitor->once, node);
    visit_children(&visitor->visitor, node);
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

static void verify_same_arena(Module* mod) {
    const IrArena* arena = get_module_arena(mod);
    ArenaVerifyVisitor visitor = {
        .visitor = {
            .visit_fn = (VisitFn) visit_verify_same_arena,
            .visit_fn_scope_rpo = true,

            // we also need to visit these potentially recursive things
            // not because we might miss nodes in a well-formed program (we shouldn't)
            // but rather because we might in a program that isn't.
            .visit_fn_addr = true,
            .visit_referenced_decls = true,
            .visit_continuations = true,
            .visit_return_fn_annotation = true,
        },
        .arena = arena,
        .once = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
    visit_module(&visitor, mod);
    destroy_dict(visitor.once);
}

void verify_module(Module* mod) {
    verify_same_arena(mod);
}
