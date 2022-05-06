#include "shady/ir.h"

#include "../portability.h"
#include "../rewrite.h"

#include <stdlib.h>

const Node* import_node(IrArena* arena, const Node* node) {
    Rewriter rewriter = {
        .src_arena = NULL,
        .dst_arena = arena,
        .rewrite_fn = recreate_node_identity,
        .processed = NULL,
    };
    return rewrite_node(&rewriter, node);
}

Nodes import_nodes(IrArena* dst_arena, Nodes old_nodes) {
    size_t count = old_nodes.count;
    LARRAY(const Node*, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = import_node(dst_arena, old_nodes.nodes[i]);
    return nodes(dst_arena, count, arr);
}

Strings import_strings(IrArena* dst_arena, Strings old_strings) {
    size_t count = old_strings.count;
    LARRAY(String, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = string(dst_arena, old_strings.strings[i]);
    return strings(dst_arena, count, arr);
}
