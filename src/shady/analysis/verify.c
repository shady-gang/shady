#include "verify.h"
#include "free_variables.h"
#include "scope.h"
#include "log.h"

#include "../visit.h"
#include "../ir_private.h"
#include "../type.h"

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
            .visit_cf_targets = true,
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

static void verify_nominal_node(const Node* fn, const Node* n) {
    switch (n->tag) {
        case Function_TAG: {
            assert(!fn && "functions cannot be part of a scope, except as the entry");
            break;
        }
        case BasicBlock_TAG: {
            assert(is_subtype(noret_type(n->arena), n->payload.basic_block.body->type));
            break;
        }
        case NominalType_TAG: {
            assert(is_type(n->payload.nom_type.body));
            break;
        }
        case Constant_TAG: {
            const Type* t = n->payload.constant.value->type;
            bool u = deconstruct_qualified_type(&t);
            assert(u);
            assert(is_subtype(n->payload.constant.type_hint, t));
            break;
        }
        case GlobalVariable_TAG: {
            if (n->payload.global_variable.init) {
                const Type* t = n->payload.global_variable.init->type;
                bool u = deconstruct_qualified_type(&t);
                assert(u);
                assert(is_subtype(n->payload.global_variable.type, t));
            }
            break;
        }
        default: break;
    }
}

static void verify_bodies(Module* mod) {
    struct List* scopes = build_scopes(mod);
    for (size_t i = 0; i < entries_count_list(scopes); i++) {
        Scope* scope = read_list(Scope*, scopes)[i];

        for (size_t j = 0; j < scope->size; j++) {
            CFNode* n = scope->rpo[j];
            if (n->node->tag == BasicBlock_TAG) {
                verify_nominal_node(scope->entry->node, n->node);
            }
        }

        destroy_scope(scope);
    }
    destroy_list(scopes);

    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        verify_nominal_node(NULL, decl);
    }
}

void verify_module(Module* mod) {
    verify_same_arena(mod);
    // before we normalize the IR, scopes are broken because decls appear where they should not
    // TODO add a normalized flag to the IR and check grammar is adhered to strictly
    if (get_module_arena(mod)->config.check_types) {
        verify_scoping(mod);
        verify_bodies(mod);
    }
}
