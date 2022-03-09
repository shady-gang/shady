#include "implem.h"
#include "type.h"

#include "murmur3.h"
#include "dict.h"

#include <string.h>

#define NODEDEF(struct_name, short_name) const struct Node* short_name(struct IrArena* arena, struct struct_name in_node) { \
    struct Node* node = (struct Node*) arena_alloc(arena, sizeof(struct Node));                                             \
    memset((void*) node, 0, sizeof(struct Node));                                                                           \
    *node = (struct Node) {                                                                                                 \
      .type = arena->config.check_types ? check_type_##short_name(arena, in_node) : NULL,                                                     \
      .tag = struct_name##_TAG,                                                                                             \
      .payload = (union NodesUnion) {                                                                                       \
          .short_name = in_node                                                                                             \
      }                                                                                                                     \
    };                                                                                                                      \
    return node;                                                                                                            \
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
