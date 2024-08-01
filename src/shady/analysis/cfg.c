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

struct List* build_cfgs(Module* mod) {
    struct List* cfgs = new_list(CFG*);

    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != Function_TAG) continue;
        CFG* cfg = build_fn_cfg(decl);
        append_list(CFG*, cfgs, cfg);
    }

    return cfgs;
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

typedef struct {
    Arena* arena;
    const Node* function;
    const Node* entry;
    LoopTree* lt;
    struct Dict* nodes;
    struct List* queue;
    struct List* contents;

    struct Dict* join_point_values;
} CfgBuildContext;

CFNode* cfg_lookup(CFG* cfg, const Node* abs) {
    CFNode** found = find_value_dict(const Node*, CFNode*, cfg->map, abs);
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
    CFNode* new = arena_alloc(a, sizeof(CFNode));
    *new = (CFNode) {
        .succ_edges = new_list(CFEdge),
        .pred_edges = new_list(CFEdge),
        .rpo_index = SIZE_MAX,
        .idom = NULL,
        .dominates = NULL,
        .structurally_dominates = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
    };
    return new;
}

static CFNode* get_or_enqueue(CfgBuildContext* ctx, const Node* abs) {
    assert(is_abstraction(abs));
    assert(!is_function(abs) || abs == ctx->function);
    CFNode** found = find_value_dict(const Node*, CFNode*, ctx->nodes, abs);
    if (found) return *found;

    CFNode* new = new_cfnode(ctx->arena);
    new->node = abs;
    assert(abs && new->node);
    insert_dict(const Node*, CFNode*, ctx->nodes, abs, new);
    append_list(Node*, ctx->queue, new);
    append_list(Node*, ctx->contents, new);
    return new;
}

static bool in_loop(LoopTree* lt, const Node* entry, const Node* block) {
    LTNode* lt_node = looptree_lookup(lt, block);
    assert(lt_node);
    LTNode* parent = lt_node->parent;
    assert(parent);

    while (parent) {
        if (entries_count_list(parent->cf_nodes) != 1)
            return false;

        if (read_list(CFNode*, parent->cf_nodes)[0]->node == entry)
            return true;

        parent = parent->parent;
    }

    return false;
}

static bool is_structural_edge(CFEdgeType edge_type) { return edge_type != JumpEdge; }

/// Adds an edge to somewhere inside a basic block
static void add_edge(CfgBuildContext* ctx, const Node* src, const Node* dst, CFEdgeType type) {
    assert(is_abstraction(src) && is_abstraction(dst));
    assert(!is_function(dst));
    assert(is_structural_edge(type) == (bool) is_case(dst));
    if (ctx->lt && !in_loop(ctx->lt, ctx->entry, dst))
        return;
    if (ctx->lt && dst == ctx->entry) {
        return;
    }

    CFNode* src_node = get_or_enqueue(ctx, src);
    CFNode* dst_node = get_or_enqueue(ctx, dst);
    CFEdge edge = {
        .type = type,
        .src = src_node,
        .dst = dst_node,
    };
    append_list(CFEdge, src_node->succ_edges, edge);
    append_list(CFEdge, dst_node->pred_edges, edge);
}

static void add_structural_dominance_edge(CfgBuildContext* ctx, CFNode* parent, const Node* dst, CFEdgeType type) {
    add_edge(ctx, parent->node, dst, type);
    insert_set_get_result(const Node*, parent->structurally_dominates, dst);
}

static void add_jump_edge(CfgBuildContext* ctx, const Node* src, const Node* j) {
    assert(j->tag == Jump_TAG);
    const Node* target = j->payload.jump.target;
    add_edge(ctx, src, target, JumpEdge);
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
            case Let_TAG: {
                terminator = terminator->payload.let.in;
                continue;
                // const Node* target = get_let_tail(terminator);
                // add_structural_dominance_edge(ctx, node, target, LetTailEdge);
                // break;
            }
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
            case Join_TAG: {
                CFNode** dst = find_value_dict(const Node*, CFNode*, ctx->join_point_values, terminator->payload.join.join_point);
                if (dst)
                    add_edge(ctx, node->node, (*dst)->node, StructuredLeaveBodyEdge);
                return;
            }
            case If_TAG:
                add_structural_dominance_edge(ctx, node, terminator->payload.if_instr.if_true, StructuredEnterBodyEdge);
                if (terminator->payload.if_instr.if_false)
                    add_structural_dominance_edge(ctx, node, terminator->payload.if_instr.if_false, StructuredEnterBodyEdge);
                add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredPseudoExitEdge);
                return;
            case Match_TAG:
                for (size_t i = 0; i < terminator->payload.match_instr.cases.count; i++)
                    add_structural_dominance_edge(ctx, node, terminator->payload.match_instr.cases.nodes[i], StructuredEnterBodyEdge);
                add_structural_dominance_edge(ctx, node, terminator->payload.match_instr.default_case, StructuredEnterBodyEdge);
                add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredPseudoExitEdge);
                return;
            case Loop_TAG:
                add_structural_dominance_edge(ctx, node, terminator->payload.loop_instr.body, StructuredEnterBodyEdge);
                add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredPseudoExitEdge);
                return;
            case Control_TAG:
                add_structural_dominance_edge(ctx, node, terminator->payload.control.inside, StructuredEnterBodyEdge);
                const Node* param = first(get_abstraction_params(terminator->payload.control.inside));
                CFNode* let_tail_cfnode = get_or_enqueue(ctx, get_structured_construct_tail(terminator));
                insert_dict(const Node*, CFNode*, ctx->join_point_values, param, let_tail_cfnode);
                add_structural_dominance_edge(ctx, node, get_structured_construct_tail(terminator), StructuredPseudoExitEdge);
                return;
            case MergeSelection_TAG:
            case MergeContinue_TAG:
            case MergeBreak_TAG: {
                return; // TODO i guess
            }
            case Terminator_BlockYield_TAG: {
                return;
            }
            case TailCall_TAG:
            case Return_TAG:
            case Unreachable_TAG:
                return;
            case NotATerminator:
                if (terminator->arena->config.check_types) {error("Grammar problem"); }
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
        CFNode* cur = read_list(CFNode*, cfg->contents)[i];

        struct List* tmp = cur->succ_edges;
        cur->succ_edges = cur->pred_edges;
        cur->pred_edges = tmp;

        for (size_t j = 0; j < entries_count_list(cur->succ_edges); j++) {
            CFEdge* edge = &read_list(CFEdge, cur->succ_edges)[j];

            CFNode* tmp2 = edge->dst;
            edge->dst = edge->src;
            edge->src = tmp2;
        }

        for (size_t j = 0; j < entries_count_list(cur->pred_edges); j++) {
            CFEdge* edge = &read_list(CFEdge, cur->pred_edges)[j];

            CFNode* tmp2 = edge->dst;
            edge->dst = edge->src;
            edge->src = tmp2;
        }

        if (entries_count_list(cur->pred_edges) == 0) {
            if (cfg->entry != NULL) {
                if (cfg->entry->node) {
                    CFNode* new_entry = new_cfnode(cfg->arena);
                    CFEdge prev_entry_edge = {
                        .type = JumpEdge,
                        .src = new_entry,
                        .dst = cfg->entry
                    };
                    append_list(CFEdge, new_entry->succ_edges, prev_entry_edge);
                    append_list(CFEdge, cfg->entry->pred_edges, prev_entry_edge);
                    cfg->entry = new_entry;
                }

                CFEdge new_edge = {
                    .type = JumpEdge,
                    .src = cfg->entry,
                    .dst = cur
                };
                append_list(CFEdge, cfg->entry->succ_edges, new_edge);
                append_list(CFEdge, cur->pred_edges, new_edge);
            } else {
                cfg->entry = cur;
            }
        }
    }

    assert(cfg->entry);
    if (!cfg->entry->node) {
        cfg->size += 1;
        append_list(Node*, cfg->contents, cfg->entry);
    }
}

static void validate_cfg(CFG* cfg) {
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* node = read_list(CFNode*, cfg->contents)[i];
        if (is_case(node->node)) {
            size_t structured_body_uses = 0;
            for (size_t j = 0; j < entries_count_list(node->pred_edges); j++) {
                CFEdge edge = read_list(CFEdge, node->pred_edges)[j];
                switch (edge.type) {
                    case JumpEdge:
                        error_print("Error: cases cannot be jumped to directly.");
                        error_die();
                    case LetTailEdge:
                        structured_body_uses += 1;
                        break;
                    case StructuredEnterBodyEdge:
                        structured_body_uses += 1;
                        break;
                    case StructuredPseudoExitEdge:
                        structured_body_uses += 1;
                    case StructuredLeaveBodyEdge:
                        break;
                }
            }
            if (structured_body_uses != 1 && node != cfg->entry /* this exception exists since we might build CFGs rooted in cases */) {
                error_print("reachable cases must be used be as bodies exactly once (actual uses: %zu)", structured_body_uses);
                error_die();
            }
        }
    }
}

CFG* build_cfg(const Node* function, const Node* entry, LoopTree* lt, bool flipped) {
    assert(function && function->tag == Function_TAG);
    assert(is_abstraction(entry));
    Arena* arena = new_arena();

    CfgBuildContext context = {
        .arena = arena,
        .function = function,
        .entry = entry,
        .lt = lt,
        .nodes = new_dict(const Node*, CFNode*, (HashFn) hash_node, (CmpFn) compare_node),
        .join_point_values = new_dict(const Node*, CFNode*, (HashFn) hash_node, (CmpFn) compare_node),
        .queue = new_list(CFNode*),
        .contents = new_list(CFNode*),
    };

    CFNode* entry_node = get_or_enqueue(&context, entry);

    while (entries_count_list(context.queue) > 0) {
        CFNode* this = pop_last_list(CFNode*, context.queue);
        process_cf_node(&context, this);
    }

    destroy_list(context.queue);
    destroy_dict(context.join_point_values);

    CFG* cfg = calloc(sizeof(CFG), 1);
    *cfg = (CFG) {
        .arena = arena,
        .entry = entry_node,
        .size = entries_count_list(context.contents),
        .flipped = flipped,
        .contents = context.contents,
        .map = context.nodes,
        .rpo = NULL
    };

    validate_cfg(cfg);

    if (flipped)
        flip_cfg(cfg);

    compute_rpo(cfg);
    compute_domtree(cfg);

    return cfg;
}

void destroy_cfg(CFG* cfg) {
    bool entry_destroyed = false;
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* node = read_list(CFNode*, cfg->contents)[i];
        entry_destroyed |= node == cfg->entry;
        destroy_list(node->pred_edges);
        destroy_list(node->succ_edges);
        if (node->dominates)
            destroy_list(node->dominates);
        if (node->structurally_dominates)
            destroy_dict(node->structurally_dominates);
    }
    if (!entry_destroyed) {
        destroy_list(cfg->entry->pred_edges);
        destroy_list(cfg->entry->succ_edges);
        if (cfg->entry->dominates)
            destroy_list(cfg->entry->dominates);
    }
    destroy_dict(cfg->map);
    destroy_arena(cfg->arena);
    free(cfg->rpo);
    destroy_list(cfg->contents);
    free(cfg);
}

static size_t post_order_visit(CFG* cfg, CFNode* n, size_t i) {
    n->rpo_index = -2;

    for (size_t j = 0; j < entries_count_list(n->succ_edges); j++) {
        CFEdge edge = read_list(CFEdge, n->succ_edges)[j];
        if (edge.dst->rpo_index == SIZE_MAX)
            i = post_order_visit(cfg, edge.dst, i);
    }

    n->rpo_index = i - 1;
    cfg->rpo[n->rpo_index] = n;
    return n->rpo_index;
}

void compute_rpo(CFG* cfg) {
    cfg->rpo = malloc(sizeof(const CFNode*) * cfg->size);
    size_t index = post_order_visit(cfg, cfg->entry, cfg->size);
    assert(index == 0);

    // debug_print("RPO: ");
    // for (size_t i = 0; i < cfg->size; i++) {
    //     debug_print("%s, ", cfg->rpo[i]->node->payload.lam.name);
    // }
    // debug_print("\n");
}

CFNode* least_common_ancestor(CFNode* i, CFNode* j) {
    assert(i && j);
    while (i->rpo_index != j->rpo_index) {
        while (i->rpo_index < j->rpo_index) j = j->idom;
        while (i->rpo_index > j->rpo_index) i = i->idom;
    }
    return i;
}

void compute_domtree(CFG* cfg) {
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = read_list(CFNode*, cfg->contents)[i];
        if (n == cfg->entry)
            continue;
        for (size_t j = 0; j < entries_count_list(n->pred_edges); j++) {
            CFEdge e = read_list(CFEdge, n->pred_edges)[j];
            CFNode* pred = e.src;
            if (pred->rpo_index < n->rpo_index) {
                n->idom = pred;
                goto outer_loop;
            }
        }
        error("no idom found");
        outer_loop:;
    }

    bool todo = true;
    while (todo) {
        todo = false;
        for (size_t i = 0; i < cfg->size; i++) {
            CFNode* n = read_list(CFNode*, cfg->contents)[i];
            if (n == cfg->entry)
                continue;
            CFNode* new_idom = NULL;
            for (size_t j = 0; j < entries_count_list(n->pred_edges); j++) {
                CFEdge e = read_list(CFEdge, n->pred_edges)[j];
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
        CFNode* n = read_list(CFNode*, cfg->contents)[i];
        n->dominates = new_list(CFNode*);
    }
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = read_list(CFNode*, cfg->contents)[i];
        if (n == cfg->entry)
            continue;
        append_list(CFNode*, n->idom->dominates, n);
    }
}
