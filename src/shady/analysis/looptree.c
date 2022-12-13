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
    Scope* s;
    State* states;
    struct List* stack;
} LoopTreeBuilder;

LTNode* new_lf_node(int type, LTNode* parent, int depth, struct List* cf_nodes) {
    LTNode* n = calloc(sizeof(LTNode), 1);
    n->parent = parent;
    n->type = type;
    n->lf_children = new_list(LTNode*);
    n->cf_nodes = cf_nodes;
    n->depth = depth;
    if (parent) {
        append_list(LTNode*, parent->lf_children, n);
    }
    return n;
}

static int visit(LoopTreeBuilder* ltb, const CFNode* n, int counter) {
    // debug_print("visiting %s \n", get_abstraction_name(n->node));
    // add n to the 'set'
    ltb->states[n->rpo_index].in_set = true;
    // set the numbers
    ltb->states[n->rpo_index].dfs = counter;
    ltb->states[n->rpo_index].low_link = counter;
    // push it
    append_list(const CFNode*, ltb->stack, n);
    return counter + 1;
}

static bool is_head(LoopTreeBuilder* ltb, const CFNode* n) { return ltb->states[n->rpo_index].is_head; }
static bool in_set(LoopTreeBuilder* ltb, const CFNode* n) { return ltb->states[n->rpo_index].in_set; }
static bool in_scc(LoopTreeBuilder* ltb, const CFNode* n) { return ltb->states[n->rpo_index].in_scc; }
static bool on_stack(LoopTreeBuilder* ltb, const CFNode* n) { return ltb->states[n->rpo_index].on_stack; }

static size_t* lowlink(LoopTreeBuilder* ltb, const CFNode* n) { return &ltb->states[n->rpo_index].low_link; }
static size_t* dfs(LoopTreeBuilder* ltb, const CFNode* n) { return &ltb->states[n->rpo_index].dfs; }

static bool is_leaf(LoopTreeBuilder* ltb, const CFNode* n, size_t num) {
    if (num == 1) {
        struct List* succ_edges = n->succ_edges;
        for (size_t i = 0; i < entries_count_list(succ_edges); i++) {
            CFEdge e = read_list(CFEdge, succ_edges)[i];
            CFNode* succ = e.dst;
            if (!is_head(ltb, succ) && n == succ)
                return false;
        }
        return true;
    }
    return false;
}

static size_t shady_min(size_t a, size_t b) { if (a < b) return a; return b; }

static int walk_scc(LoopTreeBuilder* ltb, const CFNode* cur, LTNode* parent, int depth, int scc_counter) {
    scc_counter = visit(ltb, cur, scc_counter);

    for (size_t succi = 0; succi < entries_count_list(cur->succ_edges); succi++) {
        CFEdge succe = read_list(CFEdge, cur->succ_edges)[succi];
        CFNode* succ = succe.dst;
        if (is_head(ltb, succ))
            continue; // this is a backedge
        else if (!in_set(ltb, succ)) {
            scc_counter = walk_scc(ltb, succ, parent, depth, scc_counter);
            *lowlink(ltb, cur) = shady_min(*lowlink(ltb, cur), *lowlink(ltb, succ));
        } else if (on_stack(ltb, succ))
            *lowlink(ltb, cur) = shady_min(*lowlink(ltb, cur), *lowlink(ltb, succ));
    }

    // root of SCC
    if (*lowlink(ltb, cur) == *dfs(ltb, cur)) {
        struct List* heads = new_list(const CFNode*);

        // mark all cf_nodes in current SCC (all cf_nodes from back to cur on the stack) as 'in_scc'
        size_t num = 0, e = entries_count_list(ltb->stack);
        size_t b = e - 1;
        do {
            ltb->states[read_list(const CFNode*, ltb->stack)[b]->rpo_index].in_scc = true;
            ++num;
        } while (read_list(const CFNode*, ltb->stack)[b--] != cur);

        // for all cf_nodes in current SCC
        for (size_t i = ++b; i != e; i++) {
            const CFNode* n = read_list(const CFNode*, ltb->stack)[i];

            if (ltb->s->entry == n) {
                append_list(const CFNode*, heads, n); // entries are axiomatically heads
            } else {
                for (size_t j = 0; j < entries_count_list(n->pred_edges); j++) {
                    assert(n == read_list(CFEdge, n->pred_edges)[j].dst);
                    const CFNode* pred = read_list(CFEdge, n->pred_edges)[j].src;
                    // all backedges are also inducing heads
                    // but do not yet mark them globally as head -- we are still running through the SCC
                    if (!in_scc(ltb, pred)) {
                        append_list(const CFNode*, heads, n);
                    }
                }
            }
        }

        if (is_leaf(ltb, cur, num)) {
            assert(entries_count_list(heads) == 1);
            new_lf_node(LF_LEAF, parent, depth, heads);
        } else {
            new_lf_node(LF_HEAD, parent, depth, heads);
        }

        // reset in_scc and on_stack flags
        for (size_t i = b; i != e; ++i) {
            ltb->states[read_list(const CFNode*, ltb->stack)[i]->rpo_index].in_scc = false;
            ltb->states[read_list(const CFNode*, ltb->stack)[i]->rpo_index].on_stack = false;
        }

        // pop whole SCC
        while (entries_count_list(ltb->stack) != b) {
            pop_last_list(const CFNode*, ltb->stack);
        }
    }

    return scc_counter;
}

static void clear_set(LoopTreeBuilder* ltb) {
    for (size_t i = 0; i < ltb->s->size; i++)
        ltb->states[i].in_set = false;
}

static void recurse(LoopTreeBuilder* ltb, LTNode* parent, const CFNode** heads, size_t heads_count, int depth) {
    assert(parent->type == LF_HEAD);
    size_t cur_new_child = 0;
    for (size_t i = 0; i < heads_count; i++) {
        const CFNode* head = heads[i];
        clear_set(ltb);
        // clear_dict(ltb->set);
        walk_scc(ltb, head, parent, depth, 0);

        for (size_t e = entries_count_list(parent->lf_children); cur_new_child != e; ++cur_new_child) {
            struct List* new_child_nodes = read_list(LTNode*, parent->lf_children)[cur_new_child]->cf_nodes;
            for (size_t j = 0; j < entries_count_list(new_child_nodes); j++) {
                CFNode* head2 = read_list(CFNode*, new_child_nodes)[j];
                ltb->states[head2->rpo_index].is_head = true;
            }
        }
    }

    for (size_t i = 0; i < entries_count_list(parent->lf_children); i++) {
        LTNode* node = read_list(LTNode*, parent->lf_children)[i];
        if (node->type == LF_HEAD)
            recurse(ltb, node, read_list(const CFNode*, node->cf_nodes), entries_count_list(node->cf_nodes), depth + 1);
    }
}

LoopTree* build_loop_tree(Scope* s) {
    LARRAY(State, states, s->size);
    for (size_t i = 0; i < s->size; i++) {
        states[i] = (State) {
            .in_scc = false,
            .on_stack = false,
            .in_set = false,
            .low_link = -1,
            .dfs = -1,
        };
    }
    LoopTreeBuilder ltb = {
        .states = states,
        .s = s,
        .stack = new_list(const CFNode*),
    };

    LoopTree* lt = calloc(sizeof(LoopTree), 1);
    struct List* empty_list = new_list(CFNode*);
    lt->root = new_lf_node(LF_HEAD, NULL, 0, empty_list);
    const CFNode* entry = s->entry;
    recurse(&ltb, lt->root, &entry, 1, 1);
    destroy_list(ltb.stack);
    return lt;
}

static void destroy_lt_node(LTNode* n) {
    for (size_t i = 0; i < entries_count_list(n->lf_children); i++) {
        destroy_lt_node(read_list(LTNode*, n->lf_children)[i]);
    }
    destroy_list(n->lf_children);
    destroy_list(n->cf_nodes);
    free(n);
}

void destroy_loop_tree(LoopTree* lt) {
    destroy_lt_node(lt->root);
    free(lt);
}

static int extra_uniqueness = 0;

static void dump_lt_node(FILE* f, const LTNode* n) {
    if (n->type == LF_HEAD) {
        fprintf(f, "subgraph cluster_%d {\n", extra_uniqueness++);
        if (entries_count_list(n->cf_nodes) == 0) {
            fprintf(f, "label = \"%s\";\n", "Entry");
        } else {
            fprintf(f, "label = \"%s\";\n", "LoopHead");
        }
    } else {
        //fprintf(f, "label = \"%s\";\n", "Leaf");
    }

    for (size_t i = 0; i < entries_count_list(n->cf_nodes); i++) {
        const Node* bb = read_list(const CFNode*, n->cf_nodes)[i]->node;
        fprintf(f, "%s_%d;\n", get_abstraction_name(bb), extra_uniqueness++);
    }

    for (size_t i = 0; i < entries_count_list(n->lf_children); i++) {
        const LTNode* child = read_list(const LTNode*, n->lf_children)[i];
        dump_lt_node(f, child);
    }

    if (n->type == LF_HEAD)
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
    struct List* scopes = build_scopes(mod);
    for (size_t i = 0; i < entries_count_list(scopes); i++) {
        Scope* scope = &read_list(Scope, scopes)[i];
        LoopTree* lt = build_loop_tree(scope);
        dump_loop_tree(output, lt);
        destroy_loop_tree(lt);
        dispose_scope(scope);
    }
    destroy_list(scopes);
    fprintf(output, "}\n");
}
