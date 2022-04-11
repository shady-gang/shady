#include "scope.h"
#include "../implem.h"

#include "dict.h"

#include <assert.h>

Scope build_scope(const Node* function);

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

Scope build_scope(const Node* function) {
    assert(function->tag == Function_TAG);
    struct List* contents = new_list(const Node*);

    struct Dict* done = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* queue = new_list(const Node*);

    const Node* entry = function->payload.fn.block;
    append_list(const Node*, queue, entry);

    #define enqueue(c) { \
    if (!find_key_dict(const Node*, done, c)) \
        append_list(const Node*, queue, c); \
    }

    while (entries_count_list(queue) > 0) {
        const Node* element = pop_last_list(const Node*, queue);
        assert(element->tag == Block_TAG);

        for (size_t i = 0; element->payload.block.instructions.count; i++) {
            const Node* instruction = element->payload.block.instructions.nodes[i];
            switch (instruction->tag) {
                case Let_TAG: break;
                case StructuredSelection_TAG: {
                    const Node* if_true = instruction->payload.selection.ifTrue;
                    enqueue(if_true);
                    const Node* if_false = instruction->payload.selection.ifFalse;
                    if (if_false)
                        enqueue(if_false)
                    break;
                }
                default: error("scope: unhandled instruction");
            }
        }
        const Node* terminator = element->payload.block.terminator;
        switch (terminator->tag) {
            case Jump_TAG: {
                const Node* target = terminator->payload.jump.target;
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