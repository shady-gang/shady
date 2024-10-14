#include "free_frontier.h"

#include "shady/visit.h"
#include "dict.h"

typedef struct {
    Visitor v;
    Scheduler* scheduler;
    CFG* cfg;
    CFNode* start;
    struct Dict* seen;
    struct Dict* frontier;
} FreeFrontierVisitor;

static void visit_free_frontier(FreeFrontierVisitor* v, const Node* node) {
    if (shd_dict_find_key(const Node*, v->seen, node))
        return;
    shd_set_insert_get_result(const Node*, v->seen, node);
    CFNode* where = shd_schedule_instruction(v->scheduler, node);
    if (where) {
        FreeFrontierVisitor vv = *v;
        if (shd_cfg_is_dominated(where, v->start)) {
            shd_visit_node_operands(&vv.v, NcAbstraction | NcDeclaration | NcType, node);
        } else {
            if (is_abstraction(node)) {
                struct Dict* other_ff = shd_free_frontier(v->scheduler, v->cfg, node);
                size_t i = 0;
                const Node* f;
                while (shd_dict_iter(other_ff, &i, &f, NULL)) {
                    shd_set_insert_get_result(const Node*, v->frontier, f);
                }
                shd_destroy_dict(other_ff);
            }
            if (is_value(node)) {
                shd_set_insert_get_result(const Node*, v->frontier, node);
            }
        }
    }
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

struct Dict* shd_free_frontier(Scheduler* scheduler, CFG* cfg, const Node* abs) {
    FreeFrontierVisitor ffv = {
        .v = {
            .visit_node_fn = (VisitNodeFn) visit_free_frontier,
        },
        .scheduler = scheduler,
        .cfg = cfg,
        .start = shd_cfg_lookup(cfg, abs),
        .frontier = shd_new_set(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .seen = shd_new_set(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };
    if (get_abstraction_body(abs))
        visit_free_frontier(&ffv, get_abstraction_body(abs));
    shd_destroy_dict(ffv.seen);
    return ffv.frontier;
}