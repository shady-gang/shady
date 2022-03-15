#include "implem.h"
#include "type.h"

#include "murmur3.h"
#include "dict.h"

#include <string.h>

#define LAST_ARG_true(struct_name) ,struct_name in_node
#define LAST_ARG_false(struct_name)

#define CALL_TYPING_METHOD_true(short_name) arena->config.check_types ? check_type_##short_name(arena, in_node) : NULL
#define CALL_TYPING_METHOD_false(short_name) NULL

#define SET_PAYLOAD_true(short_name) .payload = (union NodesUnion) { .short_name = in_node }
#define SET_PAYLOAD_false(_)

#define NODEDEF(has_typing_fn, has_payload, struct_name, short_name) const Node* short_name(IrArena* arena LAST_ARG_##has_payload(struct_name)) { \
    Node node;                                                                                                                                    \
    memset((void*) &node, 0, sizeof(Node));                                                                                                       \
    node = (Node) {                                                                                                                               \
      .type = CALL_TYPING_METHOD_##has_typing_fn(short_name),                                                                                     \
      .tag = struct_name##_TAG,                                                                                                                   \
      SET_PAYLOAD_##has_payload(short_name)                                                                                                       \
    };                                                                                                                                            \
    Node* ptr = &node;                                                                                                                            \
    const Node** found = find_key_dict(const Node*, arena->node_set, ptr);                                                                        \
    if (found)                                                                                                                                    \
        return *found;                                                                                                                            \
    Node* alloc = (Node*) arena_alloc(arena, sizeof(Node));                                                                                       \
    *alloc = node;                                                                                                                                \
    insert_set_get_result(const Node*, arena->node_set, alloc);                                                                                   \
    return alloc;                                                                                                                                 \
}

NODES()
#undef NODEDEF

const char* node_tags[] = {
#define NODEDEF(_, _2, _3, str) #str,
NODES()
#undef NODEDEF
};

const char* primop_names[] = {
#define PRIMOP(str) #str,
PRIMOPS()
#undef PRIMOP
};

KeyHash hash_node(Node** node) {
    uint32_t out[4];
    MurmurHash3_x64_128(*node, sizeof(Node), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    printf("hash of :");
    print_node(*node);
    printf(" = [%u] %u\n", final, final % 32);
    return final;
}

bool compare_node(Node** a, Node** b) {
    return memcmp(*a, *b, sizeof(Node)) == 0;
}
