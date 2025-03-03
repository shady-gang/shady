#include "leak.h"

#include <assert.h>
#include <string.h>

#include "shady/visit.h"

void shd_visit_enclosing_abstractions(UsesMap* map, const Node* n, void* uptr, VisitEnclosingAbsCallback fn) {
    const Use* use = shd_get_first_use(map, n);
    for (;use; use = use->next_use) {
        if (is_abstraction(use->user)) {
            fn(uptr, use);
            continue;
        }

        if (is_declaration(use->user))
            continue;

        shd_visit_enclosing_abstractions(map, n, uptr, fn);
    }
}

bool shd_is_control_static(const UsesMap* map, const Node* control) {
    assert(control->tag == Control_TAG);
    const Node* inside = control->payload.control.inside;
    const Node* jp = shd_first(get_abstraction_params(inside));

    bool found_binding_abs = false;
    const Use* use = shd_get_first_use(map, jp);
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

const Node* shd_get_control_for_jp(const UsesMap* map, const Node* jp) {
    if (!is_param(jp))
        return NULL;
    const Node* abs = jp->payload.param.abs;
    assert(is_abstraction(abs));

    const Use* use = shd_get_first_use(map, abs);
    for (;use; use = use->next_use) {
        if (use->user->tag == Control_TAG && use->operand_class == NcBasic_block && strcmp(use->operand_name, "inside") == 0)
            return use->user;
    }

    return NULL;
}
