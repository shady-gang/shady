#include "cfg.h"
#include "looptree.h"
#include "log.h"

#include "list.h"
#include "dict.h"
#include "arena.h"
#include "util.h"

#include "../ir_private.h"

#include <stdlib.h>
#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

struct List* build_cfgs(Module* mod, CFGBuildConfig config) {
    struct List* cfgs = shd_new_list(CFG*);

    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != Function_TAG) continue;
        CFG* cfg = build_cfg(decl, decl, config);
        shd_list_append(CFG*, cfgs, cfg);
    }

    return cfgs;
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

typedef struct {
    Arena* arena;
    const Node* function;
    const Node* entry;
    struct Dict* nodes;
    struct List* contents;

    CFGBuildConfig config;

    const Node* selection_construct_tail;
    const Node* loop_construct_head;
    const Node* loop_construct_tail;

    struct Dict* join_point_values;
} CfgBuildContext;

static void process_cf_node(CfgBuildContext* ctx, CFNode* node);

CFNode* cfg_lookup(CFG* cfg, const Node* abs) {
    CFNode** found = shd_dict_find_value(const Node*, CFNode*, cfg->map, abs);
    if (found) {
        CFNode* cfnode = *found;
        assert(cfnode->node);
        assert(cfnode->node == abs);
        return cfnode;
    }
    assert(false);
    return NULL;
}

static CFNode* new_cfnode(Arena* a) {
    CFNode* new = shd_arena_alloc(a, sizeof(CFNode));
    *new = (CFNode) {
        .succ_edges = shd_new_list(CFEdge),
        .pred_edges = shd_new_list(CFEdge),
        .rpo_index = SIZE_MAX,
        .idom = NULL,
        .dominates = NULL,
        .structurally_dominates = shd_new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
    };
    return new;
}

static CFNode* get_or_enqueue(CfgBuildContext* ctx, const Node* abs) {
    assert(is_abstraction(abs));
    assert(!is_function(abs) || abs == ctx->function);
    CFNode** found = shd_dict_find_value(const Node*, CFNode*, ctx->nodes, abs);
    if (found) return *found;

    CFNode* new = new_cfnode(ctx->arena);
    new->node = abs;
    assert(abs && new->node);
    shd_dict_insert(const Node*, CFNode*, ctx->nodes, abs, new);
    process_cf_node(ctx, new);
    shd_list_append(Node*, ctx->contents, new);
    return new;
}

static bool in_loop(LoopTree* lt, const Node* fn, const Node* loopentry, const Node* block) {
    LTNode* lt_node = looptree_lookup(lt, block);
    assert(lt_node);
    LTNode* parent = lt_node->parent;
    assert(parent);

    while (parent) {
        // we're not in a loop like we're expected to
        if (shd_list_count(parent->cf_nodes) == 0 && loopentry == fn)
            return true;

        // we are in the loop we were expected to
        if (shd_list_count(parent->cf_nodes) == 1 && shd_read_list(CFNode*, parent->cf_nodes)[0]->node == loopentry)
            return true;

        parent = parent->parent;
    }

    return false;
}

/// Adds an edge to somewhere inside a basic block
static void add_edge(CfgBuildContext* ctx, const Node* src, const Node* dst, CFEdgeType type, const Node* term) {
    assert(is_abstraction(src) && is_abstraction(dst));
    assert(term && is_terminator(term));
    assert(!is_function(dst));
    if (ctx->config.lt && !in_loop(ctx->config.lt, ctx->function, ctx->entry, dst))
        return;
    if (ctx->config.lt && dst == ctx->entry) {
        return;
    }

    const Node* j = term->tag == Jump_TAG ? term : NULL;

    CFNode* src_node = get_or_enqueue(ctx, src);
    CFNode* dst_node = get_or_enqueue(ctx, dst);
    CFEdge edge = {
        .type = type,
        .src = src_node,
        .dst = dst_node,
        .jump = j,
        .terminator = term,
    };
    shd_list_append(CFEdge, src_node->succ_edges, edge);
    shd_list_append(CFEdge, dst_node->pred_edges, edge);
}

static void add_structural_edge(CfgBuildContext* ctx, CFNode* parent, const Node* dst, CFEdgeType type, const Node* term) {
    add_edge(ctx, parent->node, dst, type, term);
}

static void add_structural_dominance_edge(CfgBuildContext* ctx, CFNode* parent, const Node* dst, CFEdgeType type, const Node* term) {
    add_edge(ctx, parent->node, dst, type, term);
    shd_set_insert_get_result(const Node*, parent->structurally_dominates, dst);
}

static void add_jump_edge(CfgBuildContext* ctx, const Node* src, const Node* j) {
    assert(j->tag == Jump_TAG);
    const Node* target = j->payload.jump.target;
    if (target->tag == BasicBlock_TAG)
        add_edge(ctx, src, target, JumpEdge, j);
}

#pragma GCC diagnostic error "-Wswitch"

static void process_cf_node(CfgBuildContext* ctx, CFNode* node) {
    const Node* const abs = node->node;
    assert(is_abstraction(abs));
    assert(!is_function(abs) || abs == ctx->function);
    const Node* terminator = get_abstraction_body(abs);
    if (!terminator)
        return;
    while (true) {
        switch (is_terminator(terminator)) {
            case Jump_TAG: {
                add_jump_edge(ctx, abs, terminator);
                return;
            }
            case Branch_TAG: {
                add_jump_edge(ctx, abs, terminator->payload.branch.true_jump);
                add_jump_edge(ctx, abs, terminator->payload.branch.false_jump);
                return;
            }
            case Switch_TAG: {
                for (size_t i = 0; i < terminator->payload.br_switch.case_jumps.count; i++)
                    add_jump_edge(ctx, abs, terminator->payload.br_switch.case_jumps.nodes[i]);
                add_jump_edge(ctx, abs, terminator->payload.br_switch.default_jump);
                return;
            }
            case If_TAG: {
                if (ctx->config.include_structured_tails)
                    add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredTailEdge, terminator);
                CfgBuildContext if_ctx = *ctx;
                if_ctx.selection_construct_tail = get_structured_construct_tail(terminator);
                add_structural_edge(&if_ctx, node, terminator->payload.if_instr.if_true, StructuredEnterBodyEdge, terminator);
                if (terminator->payload.if_instr.if_false)
                    add_structural_edge(&if_ctx, node, terminator->payload.if_instr.if_false, StructuredEnterBodyEdge, terminator);
                else
                    add_structural_edge(ctx, node, get_structured_construct_tail(terminator), StructuredLeaveBodyEdge, terminator);

                return;
            } case Match_TAG: {
                if (ctx->config.include_structured_tails)
                    add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredTailEdge, terminator);
                CfgBuildContext match_ctx = *ctx;
                match_ctx.selection_construct_tail = get_structured_construct_tail(terminator);
                for (size_t i = 0; i < terminator->payload.match_instr.cases.count; i++)
                    add_structural_edge(&match_ctx, node, terminator->payload.match_instr.cases.nodes[i], StructuredEnterBodyEdge, terminator);
                add_structural_edge(&match_ctx, node, terminator->payload.match_instr.default_case, StructuredEnterBodyEdge, terminator);
                return;
            } case Loop_TAG: {
                if (ctx->config.include_structured_tails)
                    add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredTailEdge, terminator);
                CfgBuildContext loop_ctx = *ctx;
                loop_ctx.loop_construct_head = terminator->payload.loop_instr.body;
                loop_ctx.loop_construct_tail = get_structured_construct_tail(terminator);
                add_structural_edge(&loop_ctx, node, terminator->payload.loop_instr.body, StructuredEnterBodyEdge, terminator);
                return;
            } case Control_TAG: {
                const Node* param = shd_first(get_abstraction_params(terminator->payload.control.inside));
                //CFNode* let_tail_cfnode = get_or_enqueue(ctx, get_structured_construct_tail(terminator));
                const Node* tail = get_structured_construct_tail(terminator);
                shd_dict_insert(const Node*, const Node*, ctx->join_point_values, param, tail);
                add_structural_dominance_edge(ctx, node, terminator->payload.control.inside, StructuredEnterBodyEdge, terminator);
                if (ctx->config.include_structured_tails)
                    add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredTailEdge, terminator);
                return;
            } case Join_TAG: {
                if (ctx->config.include_structured_exits) {
                    const Node** dst = shd_dict_find_value(const Node*, const Node*, ctx->join_point_values, terminator->payload.join.join_point);
                    if (dst)
                        add_edge(ctx, node->node, *dst, StructuredLeaveBodyEdge, terminator);
                }
                return;
            } case MergeSelection_TAG: {
                assert(ctx->selection_construct_tail);
                if (ctx->config.include_structured_exits)
                    add_structural_edge(ctx, node, ctx->selection_construct_tail, StructuredLeaveBodyEdge, terminator);
                return;
            } case MergeContinue_TAG:{
                assert(ctx->loop_construct_head);
                if (ctx->config.include_structured_exits)
                    add_structural_edge(ctx, node, ctx->loop_construct_head, StructuredLoopContinue, terminator);
                return;
            } case MergeBreak_TAG: {
                assert(ctx->loop_construct_tail);
                if (ctx->config.include_structured_exits)
                    add_structural_edge(ctx, node, ctx->loop_construct_tail, StructuredLeaveBodyEdge, terminator);
                return;
            }
            case TailCall_TAG:
            case Return_TAG:
            case Unreachable_TAG:
                return;
            case NotATerminator:
                shd_error("Grammar problem");
                return;
        }
        SHADY_UNREACHABLE;
    }
}

/**
 * Invert all edges in this cfg. Used to compute a post dominance tree.
 */
static void flip_cfg(CFG* cfg) {
    cfg->entry = NULL;

    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* cur = shd_read_list(CFNode*, cfg->contents)[i];

        struct List* tmp = cur->succ_edges;
        cur->succ_edges = cur->pred_edges;
        cur->pred_edges = tmp;

        for (size_t j = 0; j < shd_list_count(cur->succ_edges); j++) {
            CFEdge* edge = &shd_read_list(CFEdge, cur->succ_edges)[j];

            CFNode* tmp2 = edge->dst;
            edge->dst = edge->src;
            edge->src = tmp2;
        }

        for (size_t j = 0; j < shd_list_count(cur->pred_edges); j++) {
            CFEdge* edge = &shd_read_list(CFEdge, cur->pred_edges)[j];

            CFNode* tmp2 = edge->dst;
            edge->dst = edge->src;
            edge->src = tmp2;
        }

        if (shd_list_count(cur->pred_edges) == 0) {
            if (cfg->entry != NULL) {
                if (cfg->entry->node) {
                    CFNode* new_entry = new_cfnode(cfg->arena);
                    CFEdge prev_entry_edge = {
                        .type = JumpEdge,
                        .src = new_entry,
                        .dst = cfg->entry
                    };
                    shd_list_append(CFEdge, new_entry->succ_edges, prev_entry_edge);
                    shd_list_append(CFEdge, cfg->entry->pred_edges, prev_entry_edge);
                    cfg->entry = new_entry;
                }

                CFEdge new_edge = {
                    .type = JumpEdge,
                    .src = cfg->entry,
                    .dst = cur
                };
                shd_list_append(CFEdge, cfg->entry->succ_edges, new_edge);
                shd_list_append(CFEdge, cur->pred_edges, new_edge);
            } else {
                cfg->entry = cur;
            }
        }
    }

    assert(cfg->entry);
    if (!cfg->entry->node) {
        cfg->size += 1;
        shd_list_append(Node*, cfg->contents, cfg->entry);
    }
}

static void validate_cfg(CFG* cfg) {
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* node = shd_read_list(CFNode*, cfg->contents)[i];
        size_t structured_body_uses = 0;
        size_t num_jumps = 0;
        size_t num_exits = 0;
        bool is_tail = false;
        for (size_t j = 0; j < shd_list_count(node->pred_edges); j++) {
            CFEdge edge = shd_read_list(CFEdge, node->pred_edges)[j];
            switch (edge.type) {
                case JumpEdge:
                    num_jumps++;
                    break;
                case StructuredLoopContinue:
                    break;
                case StructuredEnterBodyEdge:
                    structured_body_uses += 1;
                    break;
                case StructuredTailEdge:
                    structured_body_uses += 1;
                    is_tail = true;
                    break;
                case StructuredLeaveBodyEdge:
                    num_exits += 1;
                    break;
            }
        }
        if (node != cfg->entry /* this exception exists since we might build CFGs rooted in cases */) {
            if (structured_body_uses > 0) {
                if (structured_body_uses > 1) {
                    shd_error_print("Basic block %s is used as a structural target more than once (structured_body_uses: %zu)", get_abstraction_name_safe(node->node), structured_body_uses);
                    shd_error_die();
                }
                if (num_jumps > 0) {
                    shd_error_print("Basic block %s is used as structural target, but is also jumped into (num_jumps: %zu)", get_abstraction_name_safe(node->node), num_jumps);
                    shd_error_die();
                }
                if (!is_tail && num_exits > 0) {
                    shd_error_print("Basic block %s is not a merge target yet is used as once (num_exits: %zu)", get_abstraction_name_safe(node->node), num_exits);
                    shd_error_die();
                }
            }
        }
    }
}

static void mark_reachable(CFNode* n) {
    if (!n->reachable) {
        n->reachable = true;
        for (size_t i = 0; i < shd_list_count(n->succ_edges); i++) {
            CFEdge e = shd_read_list(CFEdge, n->succ_edges)[i];
            if (e.type == StructuredTailEdge)
                continue;
            mark_reachable(e.dst);
        }
    }
}

CFG* build_cfg(const Node* function, const Node* entry, CFGBuildConfig config) {
    assert(function && function->tag == Function_TAG);
    assert(is_abstraction(entry));
    Arena* arena = shd_new_arena();

    CfgBuildContext context = {
        .arena = arena,
        .function = function,
        .entry = entry,
        .nodes = shd_new_dict(const Node*, CFNode*, (HashFn) hash_node, (CmpFn) compare_node),
        .join_point_values = shd_new_dict(const Node*, const Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .contents = shd_new_list(CFNode*),
        .config = config,
    };

    CFNode* entry_node = get_or_enqueue(&context, entry);
    mark_reachable(entry_node);
    //process_cf_node(&context, entry_node);

    //while (entries_count_list(context.queue) > 0) {
    //    CFNode* this = pop_last_list(CFNode*, context.queue);
    //    process_cf_node(&context, this);
    //}

    shd_destroy_dict(context.join_point_values);

    CFG* cfg = calloc(sizeof(CFG), 1);
    *cfg = (CFG) {
        .arena = arena,
        .config = config,
        .entry = entry_node,
        .size = shd_list_count(context.contents),
        .flipped = config.flipped,
        .contents = context.contents,
        .map = context.nodes,
        .rpo = NULL
    };

    validate_cfg(cfg);

    if (config.flipped)
        flip_cfg(cfg);

    compute_rpo(cfg);
    compute_domtree(cfg);

    return cfg;
}

void destroy_cfg(CFG* cfg) {
    bool entry_destroyed = false;
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* node = shd_read_list(CFNode*, cfg->contents)[i];
        entry_destroyed |= node == cfg->entry;
        shd_destroy_list(node->pred_edges);
        shd_destroy_list(node->succ_edges);
        if (node->dominates)
            shd_destroy_list(node->dominates);
        if (node->structurally_dominates)
            shd_destroy_dict(node->structurally_dominates);
    }
    if (!entry_destroyed) {
        shd_destroy_list(cfg->entry->pred_edges);
        shd_destroy_list(cfg->entry->succ_edges);
        if (cfg->entry->dominates)
            shd_destroy_list(cfg->entry->dominates);
    }
    shd_destroy_dict(cfg->map);
    shd_destroy_arena(cfg->arena);
    free(cfg->rpo);
    shd_destroy_list(cfg->contents);
    free(cfg);
}

static size_t post_order_visit(CFG* cfg, CFNode* n, size_t i) {
    n->rpo_index = -2;

    for (int phase = 0; phase < 2; phase++) {
        for (size_t j = 0; j < shd_list_count(n->succ_edges); j++) {
            CFEdge edge = shd_read_list(CFEdge, n->succ_edges)[j];
            // always visit structured tail edges last
            if ((edge.type == StructuredTailEdge) == (phase == 0))
                continue;
            if (edge.dst->rpo_index == SIZE_MAX)
                i = post_order_visit(cfg, edge.dst, i);
        }
    }

    n->rpo_index = i - 1;
    cfg->rpo[n->rpo_index] = n;
    return n->rpo_index;
}

void compute_rpo(CFG* cfg) {
    /*cfg->reachable_size = 0;
    for (size_t i = 0; i < entries_count_list(cfg->contents); i++) {
        CFNode* n = read_list(CFNode*, cfg->contents)[i];
        if (n->reachable)
            cfg->reachable_size++;
    }*/
    cfg->reachable_size = cfg->size;

    cfg->rpo = malloc(sizeof(const CFNode*) * cfg->size);
    size_t index = post_order_visit(cfg, cfg->entry, cfg->reachable_size);
    assert(index == 0);

    // debug_print("RPO: ");
    // for (size_t i = 0; i < cfg->size; i++) {
    //     debug_print("%s, ", cfg->rpo[i]->node->payload.lam.name);
    // }
    // debug_print("\n");
}

bool is_cfnode_structural_target(CFNode* cfn) {
    for (size_t i = 0; i < shd_list_count(cfn->pred_edges); i++) {
        if (shd_read_list(CFEdge, cfn->pred_edges)[i].type != JumpEdge)
            return true;
    }
    return false;
}

CFNode* least_common_ancestor(CFNode* i, CFNode* j) {
    assert(i && j);
    while (i->rpo_index != j->rpo_index) {
        while (i->rpo_index < j->rpo_index) j = j->idom;
        while (i->rpo_index > j->rpo_index) i = i->idom;
    }
    return i;
}

bool cfg_is_dominated(CFNode* a, CFNode* b) {
    while (a) {
        if (a == b)
            return true;
        if (a->idom)
            a = a->idom;
        else if (a->structured_idom)
            a = a->structured_idom;
        else
            break;
    }
    return false;
}

void compute_domtree(CFG* cfg) {
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = shd_read_list(CFNode*, cfg->contents)[i];
        if (n == cfg->entry/* || !n->reachable*/)
            continue;
        CFNode* structured_idom = NULL;
        for (size_t j = 0; j < shd_list_count(n->pred_edges); j++) {
            CFEdge e = shd_read_list(CFEdge, n->pred_edges)[j];
            if (e.type == StructuredTailEdge) {
                structured_idom = n->structured_idom = e.src;
                n->structured_idom_edge = e;
                continue;
            }
        }
        for (size_t j = 0; j < shd_list_count(n->pred_edges); j++) {
            CFEdge e = shd_read_list(CFEdge, n->pred_edges)[j];
            if (e.src->rpo_index < n->rpo_index) {
                n->idom = e.src;
                goto outer_loop;
            }
        }
        if (structured_idom) {
            continue;
        }
        shd_error("no idom found");
        outer_loop:;
    }

    bool todo = true;
    while (todo) {
        todo = false;
        for (size_t i = 0; i < cfg->size; i++) {
            CFNode* n = shd_read_list(CFNode*, cfg->contents)[i];
            if (n == cfg->entry || n->structured_idom)
                continue;
            CFNode* new_idom = NULL;
            for (size_t j = 0; j < shd_list_count(n->pred_edges); j++) {
                CFEdge e = shd_read_list(CFEdge, n->pred_edges)[j];
                 if (e.type == StructuredTailEdge)
                     continue;
                CFNode* p = e.src;
                new_idom = new_idom ? least_common_ancestor(new_idom, p) : p;
            }
            assert(new_idom);
            if (n->idom != new_idom) {
                n->idom = new_idom;
                todo = true;
            }
        }
    }

    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = cfg->rpo[i];
        n->dominates = shd_new_list(CFNode*);
    }
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = cfg->rpo[i];
        if (!n->idom)
            continue;
        shd_list_append(CFNode*, n->idom->dominates, n);
    }
}
