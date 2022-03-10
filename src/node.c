#include "implem.h"
#include "type.h"

#include "murmur3.h"
#include "dict.h"

#include <string.h>

#define LAST_ARG_true(struct_name) , struct struct_name in_node
#define LAST_ARG_false(struct_name)

#define CALL_TYPING_METHOD_true(short_name) arena->config.check_types ? check_type_##short_name(arena, in_node) : NULL
#define CALL_TYPING_METHOD_false(short_name) NULL

#define SET_PAYLOAD_true(short_name) .payload = (union NodesUnion) { .short_name = in_node }
#define SET_PAYLOAD_false(_)

#define NODEDEF(has_typing_fn, has_payload, struct_name, short_name) const struct Node* short_name(struct IrArena* arena LAST_ARG_##has_payload(struct_name)) { \
    struct Node node;                                                                                                                                           \
    memset((void*) &node, 0, sizeof(struct Node));                                                                                                              \
    struct Node* ptr = &node;                                                                                                                                   \
    const struct Node** found = find_key_dict(const struct Node*, arena->node_set, ptr);                                                                        \
    if (found)                                                                                                                                                  \
        return *found;                                                                                                                                          \
    node = (struct Node) {                                                                                                                                      \
      .type = CALL_TYPING_METHOD_##has_typing_fn(short_name),                                                                                                   \
      .tag = struct_name##_TAG,                                                                                                                                 \
      SET_PAYLOAD_##has_payload(short_name)                                                                                                                     \
    };                                                                                                                                                          \
    struct Node* alloc = (struct Node*) arena_alloc(arena, sizeof(struct Node));                                                                                \
    *alloc = node;                                                                                                                                              \
    insert_set_get_result(const struct Node*, arena->node_set, alloc);                                                                                          \
    return alloc;                                                                                                                                               \
}

NODES()
#undef NODEDEF

KeyHash hash_node(struct Node** node) {
    uint32_t out[4];
    MurmurHash3_x64_128(*node, sizeof(struct Node), 0x1234567, &out);
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

bool compare_node(struct Node** a, struct Node** b) {
    return memcmp(*a, *b, sizeof(struct Node)) == 0;
}
