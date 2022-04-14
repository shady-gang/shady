#include "scope.h"
#include "../implem.h"

#include "dict.h"

#include <assert.h>

static Scope build_scope(const Node* function);

struct List* build_scopes(const Node* root) {
    struct List* scopes = new_list(Scope);

    for (size_t i = 0; i < root->payload.root.variables.count; i++) {
        const Node* element = root->payload.root.definitions.nodes[i];
        if (element == NULL) continue;
        if (element->tag != Function_TAG) continue;
        Scope scope = build_scope(element);
        append_list(Scope, scopes, scope);
    }

    return scopes;
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static struct List* get_node_list(struct Dict* d, const Node* n) {
    struct List** found = find_value_dict(const Node*, struct List*, d, n);
    if (found) return *found;
    struct List* new = new_list(const Node*);
    insert_dict(const Node*, struct List*, d, n, new);
    return new;
}

static Scope build_scope(const Node* entry) {
    assert(entry->tag == Function_TAG);
    struct List* contents = new_list(const Node*);

    struct Dict* succs = new_dict(const Node*, struct List*, (HashFn) hash_node, (CmpFn) compare_node);
    struct Dict* preds = new_dict(const Node*, struct List*, (HashFn) hash_node, (CmpFn) compare_node);

    struct Dict* done = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* queue = new_list(const Node*);

    #define enqueue(node) {                                         \
        if (!find_key_dict(Node*, done, node)) {                    \
            append_list(Node*, queue, node);                        \
            append_list(Node*, contents, node);                     \
            get_node_list(succs, node);                             \
            get_node_list(preds, node);                             \
        }                                                           \
    }

    enqueue(entry);

    #define process_edge(tgt) {                                     \
        struct List* element_succs = get_node_list(succs, element); \
        append_list(Node*, element_succs, tgt);                     \
        struct List* tgt_preds = get_node_list(preds, tgt);         \
        append_list(Node*, tgt_preds, element);                     \
        assert(tgt);                                                \
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
            case Return_TAG:
            case Unreachable_TAG:
            case Join_TAG: break;
            default: error("scope: unhandled terminator");
        }
    }

    Scope scope = {
        .entry = entry,
        .contents = contents,
        .succs = succs,
        .preds = preds,
    };

    destroy_dict(done);
    destroy_list(queue);
    return scope;
}

void dispose_scope(Scope* scope) {
    for (size_t i = 0; i < entries_count_list(scope->contents); i++) {
        const Node* bb = read_list(const Node*, scope->contents)[i];
        struct List* sc = *find_value_dict(Node*, struct List*, scope->succs, bb);
        destroy_list(sc);
        struct List* pd = *find_value_dict(Node*, struct List*, scope->preds, bb);
        destroy_list(pd);
    }
    destroy_dict(scope->succs);
    destroy_dict(scope->preds);
    destroy_list(scope->contents);
}

static int extra_uniqueness = 0;

static void dump_cfg_scope(FILE* output, Scope* scope) {
    extra_uniqueness++;

    const Function* entry = &scope->entry->payload.fn;
    fprintf(output, "subgraph cluster_%s {\n", entry->name);
    fprintf(output, "label = \"%s\";\n", entry->name);
    for (size_t i = 0; i < entries_count_list(scope->contents); i++) {
        const Function* bb = &read_list(const Node*, scope->contents)[i]->payload.fn;
        fprintf(output, "%s_%d;\n", bb->name, extra_uniqueness);
    }
    for (size_t i = 0; i < entries_count_list(scope->contents); i++) {
        const Node* bb_node = read_list(const Node*, scope->contents)[i];
        const Function* bb = &bb_node->payload.fn;

        struct List* sc = *find_value_dict(Node*, struct List*, scope->succs, bb_node);
        for (size_t j = 0; j < entries_count_list(sc); j++) {
            const Node* target_node = read_list(Node*, sc)[j];
            const Function* target_bb = &target_node->payload.fn;
            fprintf(output, "%s_%d -> %s_%d;\n", bb->name, extra_uniqueness, target_bb->name, extra_uniqueness);
        }
        // struct List* pd = *find_value_dict(Node*, struct List*, scope->preds, bb);
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
