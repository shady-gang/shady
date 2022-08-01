#include "verify.h"

#include "../visit.h"

#include <assert.h>

typedef struct {
    Visitor visitor;
    const IrArena* arena;
} ArenaVerifyVisitor;

static void visit_verify_same_arena(ArenaVerifyVisitor* visitor, const Node* node) {
    assert(visitor->arena == node->arena);
    visit_children(&visitor->visitor, node);
}

static void verify_same_arena(const Node* program) {
    const IrArena* arena = program->arena;
    ArenaVerifyVisitor visitor = {
        .visitor = {
            .visit_fn = (VisitFn) visit_verify_same_arena,
            .visit_fn_scope_rpo = true,
            .visit_cf_targets = false,
            .visit_return_fn_annotation = false,
        },
        .arena = arena
    };
    visit_verify_same_arena(&visitor, program);
}

void verify_program(const Node* program) {
    verify_same_arena(program);
}
