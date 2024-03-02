#include "leak.h"

#include <assert.h>
#include <string.h>

#include "../visit.h"

void visit_enclosing_abstractions(UsesMap* map, const Node* n, void* uptr, VisitEnclosingAbsCallback fn) {
    const Use* use = get_first_use(map, n);
    for (;use; use = use->next_use) {
        if (is_abstraction(use->user)) {
            fn(uptr, use);
            continue;
        }

        if (is_declaration(use->user))
            continue;

        visit_enclosing_abstractions(map, n, uptr, fn);
    }
}

const Node* get_var_binding_abstraction(const UsesMap* map, const Node* var) {
    assert(var->tag == Variable_TAG);
    const Use* use = get_first_use(map, var);
    assert(use);
    const Use* binding_use = NULL;
    for (;use; use = use->next_use) {
        if (is_abstraction(use->user) && use->operand_class == NcVariable) {
            assert(!binding_use);
            binding_use = use;
        }
    }
    assert(binding_use && "Failed to find the binding abstraction in the uses map");
    return binding_use->user;
}

const Node* get_case_user(const UsesMap* map, const Node* cas) {
    const Use* use = get_first_use(map, cas);
    if (!use)
        return NULL;
    assert(!use->next_use);
    assert(use->operand_class == NcCase);
    return use->user;
}

bool is_control_static(const UsesMap* map, const Node* control) {
    assert(control->tag == Control_TAG);
    const Node* inside = control->payload.control.inside;
    assert(is_case(inside));
    const Node* jp = first(get_abstraction_params(inside));

    bool found_binding_abs = false;
    const Use* use = get_first_use(map, jp);
    assert(use && "we expected at least one use ... ");
    for (;use; use = use->next_use) {
        if (use->user == control->payload.control.inside) {
            found_binding_abs = true;
            continue;
        }

        if (use->user->tag == Join_TAG && strcmp(use->operand_name, "join_point") == 0)
            continue;
        return false;
    }
    assert(found_binding_abs);
    return true;
}
