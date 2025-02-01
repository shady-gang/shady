#include "free_frontier.h"

#include "shady/visit.h"
#include "shady/dict.h"

typedef struct {
    Visitor v;
    Scheduler* scheduler;
    CFG* cfg;
    CFNode* start;
    NodeSet seen;
    NodeSet frontier;
} FreeFrontierVisitor;

static void visit_free_frontier(FreeFrontierVisitor* v, const Node* node) {
    if (shd_node_set_find(v->seen, node))
        return;
    if (is_declaration(node) && node != v->start->node)
        return;
    shd_node_set_insert(v->seen, node);
    CFNode* where = shd_schedule_instruction(v->scheduler, node);
    if (where) {
        FreeFrontierVisitor vv = *v;
        if (shd_cfg_is_dominated(where, v->start)) {
            shd_visit_node_operands(&vv.v, NcAbstraction | NcFunction | NcType, node);
        } else {
            if (is_abstraction(node)) {
                NodeSet other_ff = shd_free_frontier(v->scheduler, v->cfg, node);
                size_t i = 0;
                const Node* f;
                while (shd_node_set_iter(other_ff, &i, &f)) {
                    shd_node_set_insert(v->frontier, f);
                }
                shd_destroy_node_set(other_ff);
            }
            if (is_value(node)) {
                shd_node_set_insert(v->frontier, node);
            }
        }
    }
}

NodeSet shd_free_frontier(Scheduler* scheduler, CFG* cfg, const Node* abs) {
    FreeFrontierVisitor ffv = {
        .v = {
            .visit_node_fn = (VisitNodeFn) visit_free_frontier,
        },
        .scheduler = scheduler,
        .cfg = cfg,
        .start = shd_cfg_lookup(cfg, abs),
        .frontier = shd_new_node_set(),
        .seen = shd_new_node_set(),
    };
    if (get_abstraction_body(abs))
        visit_free_frontier(&ffv, get_abstraction_body(abs));
    shd_destroy_node_set(ffv.seen);
    return ffv.frontier;
}