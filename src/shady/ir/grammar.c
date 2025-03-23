#include "shady/ir/grammar.h"

#include "../fold.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <string.h>
#include <assert.h>

Strings _shd_import_strings(IrArena* dst_arena, Strings old_strings);

static void pre_construction_validation(IrArena* arena, Node* node);

const Node* _shd_fold_node_operand(NodeTag tag, NodeClass nc, String opname, const Node* op);

const Type* _shd_check_type_generated(IrArena* a, const Node* node);

Node* _shd_create_node_helper(IrArena* arena, Node node, bool* pfresh) {
    pre_construction_validation(arena, &node);
    if (arena->config.check_types)
        node.type = _shd_check_type_generated(arena, &node);

    if (pfresh)
        *pfresh = false;

    Node* ptr = &node;
    Node** found = shd_dict_find_key(Node*, arena->node_set, ptr);
    // sanity check nominal nodes to be unique, check for duplicates in structural nodes
    if (shd_is_node_nominal(&node))
        assert(!found);
    else if (found)
        return *found;

    if (pfresh)
        *pfresh = true;

    if (arena->config.allow_fold) {
        Node* folded = (Node*) _shd_fold_node(arena, ptr);
        if (folded != ptr) {
            // The folding process simplified the node, we store a mapping to that simplified node and bail out !
            shd_set_insert(Node*, arena->node_set, folded);
            return folded;
        }
    }

    if (arena->config.check_types && node.type)
        assert(is_type(node.type));

    // place the node in the arena and return it
    Node* alloc = (Node*) shd_arena_alloc(arena->arena, sizeof(Node));
    *alloc = node;
    alloc->id = _shd_allocate_node_id(arena, alloc);
    shd_set_insert(const Node*, arena->node_set, alloc);

    return alloc;
}

#include "constructors_generated.c"
