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

static struct List* get_node_list(struct Dict* d, Node* n) {
    struct List** found = find_value_dict(const Node*, struct List*, d, n);
    if (found) return *found;
    struct List* new = new_list(const Node*);
    insert_dict_and_get_value(const Node*, struct List*, d, n, new);
    return new;
}

static Scope build_scope(const Node* function) {
    assert(function->tag == Function_TAG);
    struct List* contents = new_list(const Node*);

    struct Dict* succs = new_dict(const Node*, struct List*, (HashFn) hash_node, (CmpFn) compare_node);
    struct Dict* preds = new_dict(const Node*, struct List*, (HashFn) hash_node, (CmpFn) compare_node);

    struct Dict* done = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* queue = new_list(const Node*);

    const Node* entry = function;
    append_list(const Node*, queue, entry);

    #define enqueue(c) { \
    if (!find_key_dict(const Node*, done, c)) \
        append_list(const Node*, queue, c); \
    }

    while (entries_count_list(queue) > 0) {
        const Node* element = pop_last_list(const Node*, queue);
        assert(element->tag == Function_TAG);

        const Block* block = &element->payload.fn.block->payload.block;

        /*for (size_t i = 0; block->instructions.count; i++) {
            const Node* instruction = block->instructions.nodes[i];
            switch (instruction->tag) {
                case Let_TAG: break;
                case StructuredSelection_TAG: {
                    const Node* if_true = instruction->payload.selection.if_true;
                    enqueue(if_true);
                    const Node* if_false = instruction->payload.selection.if_false;
                    if (if_false)
                        enqueue(if_false)
                    break;
                }
                default: error("scope: unhandled instruction");
            }
        }*/
        const Node* terminator = block->terminator;
        switch (terminator->tag) {
            case Jump_TAG: {
                const Node* target = terminator->payload.jump.target;
                // get_node_list(succs, )
                enqueue(target)
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
    };

    destroy_dict(done);
    destroy_list(queue);
    return scope;
}

void dispose_scope(Scope* scope) {
    destroy_list(scope->contents);
}