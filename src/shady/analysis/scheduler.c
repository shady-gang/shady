#include "scheduler.h"

#include "visit.h"

#include "dict.h"
#include <stdlib.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

struct Scheduler_ {
    Visitor v;
    CFNode* result;
    CFG* cfg;
    struct Dict* scheduled;
};

static void schedule_after(CFNode** scheduled, CFNode* req) {
    if (!req)
        return;
    CFNode* old = *scheduled;
    if (!old)
        *scheduled = req;
    else {
        // TODO: validate that old post-dominates req
        if (req->rpo_index > old->rpo_index)
            *scheduled = req;
    }
}

static void visit_operand(Scheduler* s, NodeClass nc, String opname, const Node* op) {
    switch (nc) {
        // We only care about mem and value dependencies
        case NcMem:
        case NcValue:
            schedule_after(&s->result, schedule_instruction(s, op));
            break;
        default:
            break;
    }
}

Scheduler* new_scheduler(CFG* cfg) {
    Scheduler* s = calloc(sizeof(Scheduler), 1);
    *s = (Scheduler) {
        .v = {
            .visit_op_fn = (VisitOpFn) visit_operand,
        },
        .cfg = cfg,
        .scheduled = new_dict(const Node*, CFNode*, (HashFn) hash_node, (CmpFn) compare_node),
    };
    return s;
}

CFNode* schedule_instruction(Scheduler* s, const Node* n) {
    //assert(n && is_instruction(n));
    CFNode** found = find_value_dict(const Node*, CFNode*, s->scheduled, n);
    if (found)
        return *found;

    Scheduler s2 = *s;
    s2.result = NULL;

    if (n->tag == Param_TAG) {
        schedule_after(&s2.result, cfg_lookup(s->cfg, n->payload.param.abs));
    } else if (n->tag == AbsMem_TAG) {
        schedule_after(&s2.result, cfg_lookup(s->cfg, n->payload.abs_mem.abs));
    }

    visit_node_operands(&s2.v, 0, n);
    insert_dict(const Node*, CFNode*, s->scheduled, n, s2.result);
    return s2.result;
}

void destroy_scheduler(Scheduler* s) {
    destroy_dict(s->scheduled);
    free(s);
}
