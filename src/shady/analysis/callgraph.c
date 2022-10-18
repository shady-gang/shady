#include "callgraph.h"

#include "list.h"
#include "dict.h"

#include "portability.h"
#include "log.h"

#include "../visit.h"

#include <stdlib.h>
#include <assert.h>

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

KeyHash hash_cgnode(CGNode** n) {
    const Node* fn = (*n)->fn;
    if (!fn) return 0;
    return hash_node(&fn);

}
bool compare_cgnode(CGNode** a, CGNode** b) {
    return (*a)->fn == (*b)->fn;
}

static CGNode* process_node(CallGraph* graph, const Node* fn);

typedef struct {
    Visitor visitor;
    CallGraph* graph;
    CGNode* node;
} CGVisitor;

static const Node* ignore_immediate_fn_addr(const Node* node) {
    if (node->tag == FnAddr_TAG) {
        return node->payload.fn_addr.fn;
    }
    return node;
}

static void visit_node(CGVisitor* visitor, const Node* node) {
    switch (node->tag) {
        case Lambda_TAG: {
            CGNode* target = process_node(visitor->graph, node);
            // Immediate recursion
            if (target == visitor->node)
                visitor->node->is_recursive = true;
            insert_set_get_result(CGNode*, visitor->node->callees, target);
            break;
        }
        case FnAddr_TAG: {
            CGNode* callee_node = process_node(visitor->graph, node->payload.fn_addr.fn);
            callee_node->is_address_captured = true;
            visit_node(visitor, node->payload.fn_addr.fn);
            break;
        }
        case Call_TAG: {
            const Node* callee = node->payload.call_instr.callee;
            callee = ignore_immediate_fn_addr(callee);
            visit_node(visitor, callee);
            visit_nodes(&visitor->visitor, node->payload.call_instr.args);
            break;
        }
        case Let_TAG: {
            const Node* callee = node->payload.let.tail;
            callee = ignore_immediate_fn_addr(callee);
            visit_node(visitor, callee);
            visit_node(visitor, node->payload.let.instruction);
        }
        default: visit_children(&visitor->visitor, node);
    }
}

static CGNode* process_node(CallGraph* graph, const Node* fn) {
    assert(fn && fn->tag == Lambda_TAG);
    CGNode** found = fn ? find_value_dict(const Node*, CGNode*, graph->fn2cgn, fn) : NULL;
    if (found)
        return *found;
    CGNode* new = calloc(1, sizeof(CGNode));
    new->fn = fn;
    new->callees = new_set(CGNode*, (HashFn) hash_cgnode, (CmpFn) compare_cgnode);
    new->tarjan.index = -1;
    insert_dict_and_get_key(const Node*, CGNode*, graph->fn2cgn, fn, new);

    CGVisitor v = {
        .visitor = {
            .visit_fn_scope_rpo = true,
            .visit_fn = (VisitFn) visit_node
        },
        .graph = graph,
        .node = new,
    };

    if (fn)
        visit_children(&v.visitor, fn);

    return new;
}

#ifdef _MSC_VER
// MSVC is the world's worst C11 compiler.
// Standard headers pollute the global namespace, they do have an path to not do so, but they rely on __STDC__ being defined, which it isn't as of the latest Visual Studio 2022 release.
// Of course the Visual Studio docs say __STDC__ should be defined, but after wasting a day to this, I can positively say that's a big fat lie.
// https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
// I could, of course, simply avoid this issue by renaming. That would work and would be practical, and would be the sane thing to do.
// But it's not the right thing :tm: to do, and out of spite, this hack will remain in place until this issue is fixed.
// (Visual Studio 17.3.2, build tools 14.33.31629, windows 10 SDK 10.0.22621.0)
#undef min
#endif

static int min(int a, int b) { return a < b ? a : b; }

// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
static void strongconnect(CGNode* v, int* index, struct List* stack) {
    debug_print("strongconnect(%s) \n", v->fn->payload.lam.name);

    v->tarjan.index = *index;
    v->tarjan.lowlink = *index;
    (*index)++;
    append_list(const Node*, stack, v);
    v->tarjan.on_stack = true;

    // Consider successors of v
    {
        size_t iter = 0;
        CGNode* w;
        debug_print(" has %d successors\n", entries_count_dict(v->callees));
        while (dict_iter(v->callees, &iter, &w, NULL)) {
            debug_print("  %s\n", w->fn->payload.lam.name);
            if (w->tarjan.index == -1) {
                // Successor w has not yet been visited; recurse on it
                strongconnect(w, index, stack);
                v->tarjan.lowlink = min(v->tarjan.lowlink, w->tarjan.lowlink);
            } else if (w->tarjan.on_stack) {
                // Successor w is in stack S and hence in the current SCC
                // If w is not on stack, then (v, w) is an edge pointing to an SCC already found and must be ignored
                // Note: The next line may look odd - but is correct.
                // It says w.index not w.lowlink; that is deliberate and from the original paper
                v->tarjan.lowlink = min(v->tarjan.lowlink, w->tarjan.index);
            }
        }
    }

    // If v is a root node, pop the stack and generate an SCC
    if (v->tarjan.lowlink == v->tarjan.index) {
        LARRAY(CGNode*, scc, entries_count_list(stack));
        size_t scc_size = 0;
        {
            CGNode* w;
            assert(entries_count_list(stack) > 0);
            do {
                w = pop_last_list(CGNode*, stack);
                w->tarjan.on_stack = false;
                scc[scc_size++] = w;
            } while (v != w);
        }

        if (scc_size > 1) {
            for (size_t i = 0; i < scc_size; i++) {
                CGNode* w = scc[i];
                debug_print("Function %s is part of a recursive call chain \n", w->fn->payload.lam.name);
                w->is_recursive = true;
            }
        }
    }
}

static void tarjan(struct Dict* verts) {
    int index = 0;
    struct List* stack = new_list(CGNode*);

    size_t iter = 0;
    CGNode* n;
    while (dict_iter(verts, &iter, NULL, &n)) {
        if (n->tarjan.index == -1)
            strongconnect(n, &index, stack);
    }

    destroy_list(stack);
}

CallGraph* get_callgraph(const Node* root) {
    CallGraph* graph = calloc(sizeof(CallGraph), 1);
    *graph = (CallGraph) {
        .fn2cgn = new_dict(const Node*, CGNode*, (HashFn) hash_node, (CmpFn) compare_node)
    };

    Nodes decls = root->payload.root.declarations;
    for (size_t i = 0; i < decls.count; i++) {
        if (decls.nodes[i]->tag == Lambda_TAG) {
            process_node(graph, decls.nodes[i]);
        }
    }

    debug_print("CallGraph: done with CFG build, contains %d nodes\n", entries_count_dict(graph->fn2cgn));

    tarjan(graph->fn2cgn);

    return graph;
}

void dispose_callgraph(CallGraph* graph) {
    size_t i = 0;
    CGNode* node;
    while (dict_iter(graph->fn2cgn, &i, NULL, &node)) {
        debug_print("Freeing CG node: %s\n", node->fn->payload.lam.name);
        destroy_dict(node->callees);
        free(node);
    }
    destroy_dict(graph->fn2cgn);
    free(graph);
}
