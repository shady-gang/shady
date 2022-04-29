#include "scope.h"
#include "../log.h"

#include "dict.h"

#include <stdlib.h>
#include <assert.h>

struct List* build_scopes(const Node* root) {
    struct List* scopes = new_list(Scope);

    for (size_t i = 0; i < root->payload.root.declarations.count; i++) {
        const Node* decl = root->payload.root.declarations.nodes[i];
        if (decl->tag != Function_TAG) continue;
        Scope scope = build_scope(decl);
        append_list(Scope, scopes, scope);
    }

    return scopes;
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static CFNode* get_or_create_cf_node(struct Dict* d, const Node* n) {
    CFNode** found = find_value_dict(const Node*, CFNode*, d, n);
    if (found) return *found;
    CFNode* new = malloc(sizeof(CFNode));
    *new = (CFNode) {
        .node = n,
        .succs = new_list(CFNode*),
        .preds = new_list(CFNode*),
        .rpo_index = SIZE_MAX,
        .idom = NULL,
        .dominates = NULL,
    };
    insert_dict(const Node*, CFNode*, d, n, new);
    return new;
}

Scope build_scope(const Node* entry) {
    assert(entry->tag == Function_TAG);
    struct List* contents = new_list(CFNode*);

    struct Dict* nodes = new_dict(const Node*, CFNode*, (HashFn) hash_node, (CmpFn) compare_node);

    struct Dict* done = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* queue = new_list(const Node*);

    #define enqueue(node) {                                         \
        if (!find_key_dict(Node*, done, node)) {                    \
            append_list(Node*, queue, node);                        \
            CFNode* cf_node = get_or_create_cf_node(nodes, node);   \
            append_list(CFNode*, contents, cf_node);                \
        }                                                           \
    }

    enqueue(entry);

    #define process_edge(tgt) {                                     \
        assert(tgt);                                                \
        CFNode* src_node = get_or_create_cf_node(nodes, element);   \
        CFNode* tgt_node = get_or_create_cf_node(nodes, tgt);       \
        append_list(CFNode*, src_node->succs, tgt_node);            \
        append_list(CFNode*, tgt_node->preds, src_node);            \
        enqueue(tgt);                                               \
    }

    while (entries_count_list(queue) > 0) {
        const Node* element = pop_last_list(const Node*, queue);
        assert(element->tag == Function_TAG);
        const FnType* fn_type = &element->type->payload.fn_type;
        assert(fn_type->is_continuation || element == entry);

        const Block* block = &element->payload.fn.block->payload.block;
        const Node* terminator = block->terminator;
        switch (terminator->tag) {
            case Jump_TAG: {
                const Node* target = terminator->payload.jump.target;
                process_edge(target)
                break;
            }
            case Branch_TAG: {
                const Node* true_target = terminator->payload.branch.true_target;
                const Node* false_target = terminator->payload.branch.false_target;
                process_edge(true_target);
                process_edge(false_target);
                break;
            }
            case Callf_TAG:
            case Return_TAG:
            case Unreachable_TAG:
            case Merge_TAG: break;
            default: error("scope: unhandled terminator");
        }
    }

    CFNode* entry_node = get_or_create_cf_node(nodes, entry);

    Scope scope = {
        .entry = entry_node,
        .size = entries_count_list(contents),
        .contents = contents,
        .rpo = NULL
    };

    destroy_dict(done);
    destroy_dict(nodes);
    destroy_list(queue);

    compute_rpo(&scope);
    compute_domtree(&scope);

    return scope;
}

static size_t post_order_visit(Scope* scope, CFNode* n, size_t i) {
    n->rpo_index = -2;

    for (size_t j = 0; j < entries_count_list(n->succs); j++) {
        CFNode* succ = read_list(CFNode*, n->succs)[j];
        if (succ->rpo_index == SIZE_MAX)
            i = post_order_visit(scope, succ, i);
    }

    n->rpo_index = i - 1;
    scope->rpo[n->rpo_index] = n;
    return n->rpo_index;
}

void compute_rpo(Scope* scope) {
    scope->rpo = malloc(sizeof(const CFNode*) * scope->size);
    size_t index = post_order_visit(scope,  scope->entry, scope->size);
    assert(index == 0);

    debug_print("RPO: ");
    for (size_t i = 0; i < scope->size; i++) {
        debug_print("%s, ", scope->rpo[i]->node->payload.fn.name);
    }
    debug_print("\n");
}

CFNode* least_common_ancestor(CFNode* i, CFNode* j) {
    assert(i && j);
    while (i->rpo_index != j->rpo_index) {
        while (i->rpo_index < j->rpo_index) j = j->idom;
        while (i->rpo_index > j->rpo_index) i = i->idom;
    }
    return i;
}

void compute_domtree(Scope* scope) {
    for (size_t i = 1; i < scope->size; i++) {
        CFNode* n = read_list(CFNode*, scope->contents)[i];
        for (size_t j = 0; j < entries_count_list(n->preds); j++) {
            CFNode* p = read_list(CFNode*, n->preds)[j];
            if (p->rpo_index < n->rpo_index) {
                n->idom = p;
                goto outer_loop;
            }
        }
        error("no idom found for %s", n->node->payload.fn.name);
        outer_loop:;
    }

    bool todo = true;
    while (todo) {
        todo = false;
        for (size_t i = 1; i < scope->size; i++) {
            CFNode* n = read_list(CFNode*, scope->contents)[i];
            CFNode* new_idom = NULL;
            for (size_t j = 0; j < entries_count_list(n->preds); j++) {
                CFNode* p = read_list(CFNode*, n->preds)[j];
                new_idom = new_idom ? least_common_ancestor(new_idom, p) : p;
            }
            assert(new_idom);
            if (n->idom != new_idom) {
                n->idom = new_idom;
                todo = true;
            }
        }
    }

    for (size_t i = 0; i < scope->size; i++) {
        CFNode* n = read_list(CFNode*, scope->contents)[i];
        n->dominates = new_list(CFNode*);
    }
    for (size_t i = 1; i < scope->size; i++) {
        CFNode* n = read_list(CFNode*, scope->contents)[i];
        append_list(CFNode*, n->idom->dominates, n);
    }
}

void dispose_scope(Scope* scope) {
    for (size_t i = 0; i < scope->size; i++) {
        CFNode* node = read_list(CFNode*, scope->contents)[i];
        destroy_list(node->preds);
        destroy_list(node->succs);
        if (node->dominates)
            destroy_list(node->dominates);
        free(node);
    }
    free(scope->rpo);
    destroy_list(scope->contents);
}

static int extra_uniqueness = 0;

static void dump_cfg_scope(FILE* output, Scope* scope) {
    extra_uniqueness++;

    const Function* entry = &scope->entry->node->payload.fn;
    fprintf(output, "subgraph cluster_%s {\n", entry->name);
    fprintf(output, "label = \"%s\";\n", entry->name);
    for (size_t i = 0; i < entries_count_list(scope->contents); i++) {
        const Function* bb = &read_list(const CFNode*, scope->contents)[i]->node->payload.fn;
        fprintf(output, "%s_%d;\n", bb->name, extra_uniqueness);
    }
    for (size_t i = 0; i < entries_count_list(scope->contents); i++) {
        const CFNode* bb_node = read_list(const CFNode*, scope->contents)[i];
        const Function* bb = &bb_node->node->payload.fn;

        for (size_t j = 0; j < entries_count_list(bb_node->succs); j++) {
            const CFNode* target_node = read_list(CFNode*, bb_node->succs)[j];
            const Function* target_bb = &target_node->node->payload.fn;
            fprintf(output, "%s_%d -> %s_%d;\n", bb->name, extra_uniqueness, target_bb->name, extra_uniqueness);
        }
    }
    fprintf(output, "}\n");
}

void dump_cfg(FILE* output, const Node* root) {
    if (output == NULL)
        output = stderr;

    fprintf(output, "digraph G {\n");
    struct List* scopes = build_scopes(root);
    for (size_t i = 0; i < entries_count_list(scopes); i++) {
        Scope* scope = &read_list(Scope, scopes)[i];
        dump_cfg_scope(output, scope);
        dispose_scope(scope);
    }
    destroy_list(scopes);
    fprintf(output, "}\n");
}
