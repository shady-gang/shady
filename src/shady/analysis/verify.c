#include "verify.h"
#include "free_variables.h"
#include "scope.h"
#include "log.h"

#include "../visit.h"
#include "../ir_private.h"

#include "dict.h"
#include "list.h"

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
            .visit_referenced_decls = true,
            .visit_continuations = true,
        },
        .arena = arena,
        .once = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
    visit_module(&visitor.visitor, mod);
    destroy_dict(visitor.once);
}

static void verify_scoping(Module* mod) {
    struct List* scopes = build_scopes(mod);
    for (size_t i = 0; i < entries_count_list(scopes); i++) {
        Scope* scope = read_list(Scope*, scopes)[i];
        struct List* leaking = compute_free_variables(scope);
        for (size_t j = 0; j < entries_count_list(leaking); j++) {
            log_node(ERROR, read_list(const Node*, leaking)[j]);
            error_print("\n");
        }
        assert(entries_count_list(leaking) == 0);
        destroy_list(leaking);
        destroy_scope(scope);
    }
    destroy_list(scopes);
}

void verify_module(Module* mod) {
    verify_same_arena(mod);
    // before we normalize the IR, scopes are broken because decls appear where they should not
    // TODO add a normalized flag to the IR and check grammar is adhered to strictly
    if (get_module_arena(mod)->config.check_types)
       verify_scoping(mod);
}
