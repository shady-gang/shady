#include "looptree.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "log.h"

#include <stdlib.h>
#include <stdio.h>

typedef struct {
    bool in_scc, on_stack, is_head;
    bool in_set;
    size_t dfs, low_link;
} State;

typedef struct {
    CFG* s;
    State* states;

    /**
     * @ref List of @ref CFNode*
     */
    struct List* stack;
} LoopTreeBuilder;

KeyHash shd_hash_node(const Node**);
bool shd_compare_node(const Node**, const Node**);

LTNode* new_lf_node(int type, LTNode* parent, int depth, struct List* cf_nodes) {
    LTNode* n = calloc(sizeof(LTNode), 1);
    n->parent = parent;
    n->type = type;
    n->lf_children = shd_new_list(LTNode*);
    n->cf_nodes = cf_nodes;
    n->depth = depth;
    if (parent) {
        shd_list_append(LTNode*, parent->lf_children, n);
    }
    return n;
}

#define shady_min(a, b) (((a) < (b)) ? (a) : (b))

#define is_head(ltb, n) ltb->states[n->rpo_index].is_head
#define in_set(ltb, n) ltb->states[n->rpo_index].in_set
#define in_scc(ltb, n) ltb->states[n->rpo_index].in_scc
#define on_stack(ltb, n) ltb->states[n->rpo_index].on_stack
#define lowlink(ltb, n) ltb->states[n->rpo_index].low_link
#define dfs(ltb, n) ltb->states[n->rpo_index].dfs

static bool is_leaf(LoopTreeBuilder* ltb, const CFNode* n, size_t num) {
    if (num == 1) {
        struct List* succ_edges = n->succ_edges;
        for (size_t i = 0; i < shd_list_count(succ_edges); i++) {
            CFEdge e = shd_read_list(CFEdge, succ_edges)[i];
            CFNode* succ = e.dst;
            if (!is_head(ltb, succ) && n == succ)
                return false;
        }
        return true;
    }
    return false;
}

static int visit(LoopTreeBuilder* ltb, const CFNode* n, int counter) {
    // debug_print("visiting %s \n", get_abstraction_name(n->node));
    // add n to the 'set'
    in_set(ltb, n) = true;
    // set the numbers
    ltb->states[n->rpo_index].dfs = counter;
    ltb->states[n->rpo_index].low_link = counter;
    // push it
    shd_list_append(const CFNode*, ltb->stack, n);
    on_stack(ltb, n) = true;
    return counter + 1;
}

static int walk_scc(LoopTreeBuilder* ltb, const CFNode* cur, LTNode* parent, int depth, int scc_counter) {
    scc_counter = visit(ltb, cur, scc_counter);

    for (size_t succi = 0; succi < shd_list_count(cur->succ_edges); succi++) {
        CFEdge succe = shd_read_list(CFEdge, cur->succ_edges)[succi];
        CFNode* succ = succe.dst;
        if (is_head(ltb, succ))
            continue; // this is a backedge
        if (!in_set(ltb, succ)) {
            scc_counter = walk_scc(ltb, succ, parent, depth, scc_counter);
            lowlink(ltb, cur) = shady_min(lowlink(ltb, cur), lowlink(ltb, succ));
        } else if (on_stack(ltb, succ)) {
            lowlink(ltb, cur) = shady_min(lowlink(ltb, cur), lowlink(ltb, succ));
        }
    }

    // root of SCC
    if (lowlink(ltb, cur) == dfs(ltb, cur)) {
        struct List* heads = shd_new_list(const CFNode*);

        // mark all cf_nodes in current SCC (all cf_nodes from back to cur on the stack) as 'in_scc'
        size_t num = 0, e = shd_list_count(ltb->stack);
        size_t b = e - 1;
        do {
            in_scc(ltb, shd_read_list(const CFNode*, ltb->stack)[b]) = true;
            ++num;
        } while (shd_read_list(const CFNode*, ltb->stack)[b--] != cur);

        // for all cf_nodes in current SCC
        for (size_t i = ++b; i != e; i++) {
            const CFNode* n = shd_read_list(const CFNode*, ltb->stack)[i];

            if (ltb->s->entry == n) {
                shd_list_append(const CFNode*, heads, n); // entries are axiomatically heads
            } else {
                for (size_t j = 0; j < shd_list_count(n->pred_edges); j++) {
                    assert(n == shd_read_list(CFEdge, n->pred_edges)[j].dst);
                    const CFNode* pred = shd_read_list(CFEdge, n->pred_edges)[j].src;
                    // all backedges are also inducing heads
                    // but do not yet mark them globally as head -- we are still running through the SCC
                    if (!in_scc(ltb, pred)) {
                        shd_list_append(const CFNode*, heads, n);
                        break;
                    }
                }
            }
        }

        if (is_leaf(ltb, cur, num)) {
            assert(shd_list_count(heads) == 1);
            new_lf_node(LF_LEAF, parent, depth, heads);
        } else {
            new_lf_node(LF_HEAD, parent, depth, heads);
        }

        // reset in_scc and on_stack flags
        for (size_t i = b; i != e; ++i) {
            in_scc(ltb, shd_read_list(const CFNode*, ltb->stack)[i]) = false;
            on_stack(ltb, shd_read_list(const CFNode*, ltb->stack)[i]) = false;
        }

        // pop whole SCC
        while (shd_list_count(ltb->stack) != b) {
            shd_list_pop(const CFNode*, ltb->stack);
        }
    }

    return scc_counter;
}

static void clear_set(LoopTreeBuilder* ltb) {
    for (size_t i = 0; i < ltb->s->size; i++)
        ltb->states[i].in_set = false;
}

static void recurse(LoopTreeBuilder* ltb, LTNode* parent, struct List* heads, int depth) {
    assert(parent->type == LF_HEAD);
    size_t cur_new_child = 0;
    for (size_t i = 0; i < shd_list_count(heads); i++) {
        const CFNode* head = shd_read_list(const CFNode*, heads)[i];
        clear_set(ltb);
        walk_scc(ltb, head, parent, depth, 0);

        for (size_t e = shd_list_count(parent->lf_children); cur_new_child != e; ++cur_new_child) {
            struct List* new_child_nodes = shd_read_list(LTNode*, parent->lf_children)[cur_new_child]->cf_nodes;
            for (size_t j = 0; j < shd_list_count(new_child_nodes); j++) {
                CFNode* head2 = shd_read_list(CFNode*, new_child_nodes)[j];
                is_head(ltb, head2) = true;
            }
        }
    }

    for (size_t i = 0; i < shd_list_count(parent->lf_children); i++) {
        LTNode* node = shd_read_list(LTNode*, parent->lf_children)[i];
        if (node->type == LF_HEAD)
            recurse(ltb, node, node->cf_nodes, depth + 1);
    }
}

static void build_map_recursive(struct Dict* map, LTNode* n) {
    if (n->type == LF_LEAF) {
        assert(shd_list_count(n->cf_nodes) == 1);
        const Node* node = shd_read_list(CFNode*, n->cf_nodes)[0]->node;
        shd_dict_insert(const Node*, LTNode*, map, node, n);
    } else {
        for (size_t i = 0; i < shd_list_count(n->lf_children); i++) {
            LTNode* child = shd_read_list(LTNode*, n->lf_children)[i];
            build_map_recursive(map, child);
        }
    }
}

LTNode* looptree_lookup(LoopTree* lt, const Node* block) {
    LTNode** found = shd_dict_find_value(const Node*, LTNode*, lt->map, block);
    if (found) return *found;
    assert(false);
}

LoopTree* build_loop_tree(CFG* s) {
    LARRAY(State, states, s->size);
    for (size_t i = 0; i < s->size; i++) {
        states[i] = (State) {
            .in_scc = false,
            .is_head = false,
            .on_stack = false,
            .in_set = false,
            .low_link = -1,
            .dfs = -1,
        };
    }
    LoopTreeBuilder ltb = {
        .states = states,
        .s = s,
        .stack = shd_new_list(const CFNode*),
    };

    LoopTree* lt = calloc(sizeof(LoopTree), 1);
    struct List* empty_list = shd_new_list(CFNode*);
    lt->root = new_lf_node(LF_HEAD, NULL, 0, empty_list);
    const CFNode* entry = s->entry;
    struct List* global_heads = shd_new_list(const CFNode*);
    shd_list_append(const CFNode*, global_heads, entry);
    recurse(&ltb, lt->root, global_heads, 1);
    shd_destroy_list(global_heads);
    shd_destroy_list(ltb.stack);

    lt->map = shd_new_dict(const Node*, LTNode*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);
    build_map_recursive(lt->map, lt->root);

    return lt;
}

static void destroy_lt_node(LTNode* n) {
    for (size_t i = 0; i < shd_list_count(n->lf_children); i++) {
        destroy_lt_node(shd_read_list(LTNode*, n->lf_children)[i]);
    }
    shd_destroy_list(n->lf_children);
    shd_destroy_list(n->cf_nodes);
    free(n);
}

void destroy_loop_tree(LoopTree* lt) {
    destroy_lt_node(lt->root);
    shd_destroy_dict(lt->map);
    free(lt);
}

static int extra_uniqueness = 0;

static void dump_lt_node(FILE* f, const LTNode* n) {
    if (n->type == LF_HEAD) {
        fprintf(f, "subgraph cluster_%d {\n", extra_uniqueness++);
        if (shd_list_count(n->cf_nodes) == 0) {
            fprintf(f, "label = \"%s\";\n", "Entry");
        } else {
            fprintf(f, "label = \"%s\";\n", "LoopHead");
        }
    } else {
        fprintf(f, "subgraph cluster_%d {\n", extra_uniqueness++);
        fprintf(f, "label = \"%s\";\n", "Leaf");
    }

    for (size_t i = 0; i < shd_list_count(n->cf_nodes); i++) {
        const Node* bb = shd_read_list(const CFNode*, n->cf_nodes)[i]->node;
        fprintf(f, "bb_%d[label=\"%s\"];\n", extra_uniqueness++, get_abstraction_name_safe(bb));
    }

    for (size_t i = 0; i < shd_list_count(n->lf_children); i++) {
        const LTNode* child = shd_read_list(const LTNode*, n->lf_children)[i];
        dump_lt_node(f, child);
    }

    //if (n->type == LF_HEAD)
        fprintf(f, "}\n");
}

void dump_loop_tree(FILE* f, LoopTree* lt) {
    //fprintf(f, "digraph G {\n");
    fprintf(f, "subgraph cluster_%d {\n", extra_uniqueness++);
    dump_lt_node(f, lt->root);
    fprintf(f, "}\n");
    //fprintf(f, "}\n");
}

void dump_loop_trees(FILE* output, Module* mod) {
    if (output == NULL)
        output = stderr;

    fprintf(output, "digraph G {\n");
    struct List* cfgs = build_cfgs(mod, default_forward_cfg_build());
    for (size_t i = 0; i < shd_list_count(cfgs); i++) {
        CFG* cfg = shd_read_list(CFG*, cfgs)[i];
        LoopTree* lt = build_loop_tree(cfg);
        dump_loop_tree(output, lt);
        destroy_loop_tree(lt);
        destroy_cfg(cfg);
    }
    shd_destroy_list(cfgs);
    fprintf(output, "}\n");
}


void dump_loop_trees_auto(Module* mod) {
    FILE* f = fopen("loop_trees.dot", "wb");
    dump_loop_trees(f, mod);
    fclose(f);
}
