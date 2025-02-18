#include "scheduler.h"

#include "shady/visit.h"
#include "shady/be/dump.h"

#include "log.h"
#include "dict.h"

#include <stdlib.h>
#include <setjmp.h>

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

struct Scheduler_ {
    Visitor v;
    CFNode* result;
    CFG* cfg;
    struct Dict* scheduled;
    jmp_buf bail;
};

static void schedule_after(Scheduler* scheduler, const Node* op, CFNode** scheduled, CFNode* req) {
    if (!req)
        return;
    CFNode* old = *scheduled;
    if (!old)
        *scheduled = req;
    else {
        // TODO: validate that old post-dominates req
        if (req->rpo_index > old->rpo_index) {
            CHECK(shd_cfg_is_dominated(req, old), {
                shd_dump_unscheduled(op);
                shd_log_fmt(ERROR, "Scheduling failure: operand ");
                shd_log_node(ERROR, op);
                shd_log_fmt(ERROR, " needs to be scheduled at/after ");
                shd_log_node(ERROR, req->node);
                shd_log_fmt(ERROR, " but that would not be dominated by the previous scheduled location in ");
                shd_log_node(ERROR, old->node);
                shd_log_fmt(ERROR, ".\n");
                longjmp(scheduler->bail, 1);
            });
            *scheduled = req;
        } else {
            CHECK(shd_cfg_is_dominated(old, req), {
                shd_dump_unscheduled(op);
                shd_log_fmt(ERROR, "Scheduling failure: operand ");
                shd_log_node(ERROR, op);
                shd_log_fmt(ERROR, " needs to be scheduled after ");
                shd_log_node(ERROR, req->node);
                shd_log_fmt(ERROR, " but that would not dominate the previous scheduled location in ");
                shd_log_node(ERROR, old->node);
                shd_log_fmt(ERROR, ".\n");
                longjmp(scheduler->bail, 1);
            });
        }
    }
}

static void visit_operand(Scheduler* s, NodeClass nc, String opname, const Node* op, size_t i) {
    if (is_declaration(op))
        return;
    switch (nc) {
        // We only care about mem and value dependencies
        case NcMem:
        case NcValue:
            schedule_after(s, op, &s->result, shd_schedule_instruction(s, op));
            break;
        default:
            break;
    }
}

Scheduler* shd_new_scheduler(CFG* cfg) {
    Scheduler* s = calloc(sizeof(Scheduler), 1);
    *s = (Scheduler) {
        .v = {
            .visit_op_fn = (VisitOpFn) visit_operand,
        },
        .cfg = cfg,
        .scheduled = shd_new_dict(const Node*, CFNode*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };
    return s;
}

CFNode* shd_schedule_instruction(Scheduler* s, const Node* n) {
    //assert(n && is_instruction(n));
    CFNode** found = shd_dict_find_value(const Node*, CFNode*, s->scheduled, n);
    if (found)
        return *found;

    Scheduler s2 = *s;
    s2.result = NULL;

    if (setjmp(s2.bail) != 0)
        abort();

    if (n->tag == Param_TAG) {
        schedule_after(&s2, n, &s2.result, shd_cfg_lookup(s->cfg, n->payload.param.abs));
    } else if (n->tag == BasicBlock_TAG) {
        // assert(false);
        schedule_after(&s2, n, &s2.result, shd_cfg_lookup(s->cfg, n));
    } else if (n->tag == AbsMem_TAG) {
        schedule_after(&s2, n, &s2.result, shd_cfg_lookup(s->cfg, n->payload.abs_mem.abs));
    }

    shd_visit_node_operands(&s2.v, ~(NcValue | NcMem), n);
    shd_dict_insert(const Node*, CFNode*, s->scheduled, n, s2.result);
    return s2.result;
}

void shd_destroy_scheduler(Scheduler* s) {
    shd_destroy_dict(s->scheduled);
    free(s);
}
