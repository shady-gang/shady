#include "implem.h"
#include "type.h"

#include "murmur3.h"
#include "dict.h"

#include <string.h>

#define LAST_ARG_1(struct_name) ,struct_name in_node
#define LAST_ARG_0(struct_name)

#define CALL_TYPING_METHOD_11(short_name) arena->config.check_types ? check_type_##short_name(arena, in_node) : NULL
#define CALL_TYPING_METHOD_01(short_name) NULL
#define CALL_TYPING_METHOD_10(short_name) arena->config.check_types ? check_type_##short_name(arena) : NULL
#define CALL_TYPING_METHOD_00(short_name) NULL

#define SET_PAYLOAD_1(short_name) .payload = (union NodesUnion) { .short_name = in_node }
#define SET_PAYLOAD_0(_)

#define NODE_CTOR_1(has_typing_fn, has_payload, struct_name, short_name) const Node* short_name(IrArena* arena LAST_ARG_##has_payload(struct_name)) { \
    Node node;                                                                                                                                    \
    memset((void*) &node, 0, sizeof(Node));                                                                                                       \
    node = (Node) {                                                                                                                               \
      .type = CALL_TYPING_METHOD_##has_typing_fn##has_payload(short_name),                                                                                   \
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

#define NODE_CTOR_0(has_typing_fn, has_payload, struct_name, short_name)
#define NODEDEF(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) NODE_CTOR_##autogen_ctor(has_typing_fn, has_payload, struct_name, short_name)
NODES()
#undef NODEDEF

const Node* var(IrArena* arena, const Type* type, const char* name) {
    Variable variable = {
        .type = type,
        .name = string(arena, name),
        .id = fresh_id(arena)
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
      .type = arena->config.check_types ? check_type_var(arena, variable) : NULL,
      .tag = Variable_TAG,
      .payload.var = variable
    };
    Node* ptr = &node;
    const Node** found = find_key_dict(const Node*, arena->node_set, ptr);
    if (found)
        return *found;
    Node* alloc = (Node*) arena_alloc(arena, sizeof(Node));
    *alloc = node;
    insert_set_get_result(const Node*, arena->node_set, alloc);
    return alloc;
}

const char* node_tags[] = {
#define NODEDEF(_, _2, _3, _4, str) #str,
NODES()
#undef NODEDEF
};

const char* primop_names[] = {
#define PRIMOP(str) #str,
PRIMOPS()
#undef PRIMOP
};

KeyHash hash_node(Node** node) {
    if (is_nominal((*node)->tag)) {
        size_t ptr = (size_t) *node;
        uint32_t upper = ptr >> 32;
        uint32_t lower = ptr;
        return upper ^ lower;
    }

    uint32_t out[4];
    MurmurHash3_x64_128(*node, sizeof(Node), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    //printf("hash of :");
    //print_node(*node);
    //printf(" = [%u] %u\n", final, final % 32);
    return final;
}

bool compare_node(Node** a, Node** b) {
    if ((*a)->tag != (*b)->tag) return false;
    if (is_nominal((*a)->tag))
        return *a == *b;
    return memcmp(*a, *b, sizeof(Node)) == 0;
}
