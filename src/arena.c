#include "ir.h"
#include "implem.h"
#include "type.h"

#include "dict.h"
#include "murmur3.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define alloc_size 1024 * 1024

KeyHash hash_nodes(struct Nodes* nodes);
bool compare_nodes(struct Nodes* a, struct Nodes* b);

KeyHash hash_strings(struct Strings* strings);
bool compare_strings(struct Strings* a, struct Strings* b);

KeyHash hash_string(const char** string);
bool compare_string(const char** a, const char** b);

KeyHash hash_node(const struct Node**);
bool compare_node(const struct Node** a, const struct Node** b);

struct IrArena* new_arena(struct IrConfig config) {
    struct IrArena* arena = malloc(sizeof(struct IrArena));
    *arena = (struct IrArena) {
        .nblocks = 0,
        .maxblocks = 256,
        .blocks = malloc(256 * sizeof(size_t)),
        .available = 0,
        .config = config,

        .node_set = new_set(const struct Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .string_set = new_set(const char*, (HashFn) hash_string, (CmpFn) compare_string),

        .nodes_set   = new_set(struct Nodes, (HashFn) hash_nodes, (CmpFn) compare_nodes),
        .strings_set = new_set(struct Strings, (HashFn) hash_strings, (CmpFn) compare_strings),
    };
    for (int i = 0; i < arena->maxblocks; i++)
        arena->blocks[i] = NULL;
    return arena;
}

void destroy_arena(struct IrArena* arena) {
    destroy_dict(arena->strings_set);
    destroy_dict(arena->string_set);
    destroy_dict(arena->node_set);
    for (int i = 0; i < arena->nblocks; i++) {
        free(arena->blocks[i]);
    }
    free(arena->blocks);
    free(arena);
}

void* arena_alloc(struct IrArena* arena, size_t size) {
    if (size == 0)
        return NULL;
    // arena is full
    if (size > arena->available) {
        assert(arena->nblocks <= arena->maxblocks);
        // we need more storage for the block pointers themselves !
        if (arena->nblocks == arena->maxblocks) {
            arena->maxblocks *= 2;
            arena->blocks = realloc(arena->blocks, arena->maxblocks);
        }

        arena->blocks[arena->nblocks++] = malloc(alloc_size);
        arena->available = alloc_size;
    }

    assert(size <= arena->available);

    size_t in_block = alloc_size - arena->available;
    void* allocated = (void*) ((size_t) arena->blocks[arena->nblocks - 1] + in_block);
    memset(allocated, 0, size);
    arena->available -= size;
    return allocated;
}

struct Nodes nodes(struct IrArena* arena, size_t count, const struct Node* in_nodes[]) {
    struct Nodes tmp = {
        .count = count,
        .nodes = in_nodes
    };
    const struct Nodes* found = find_key_dict(struct Nodes, arena->nodes_set, tmp);
    if (found)
        return *found;

    struct Nodes nodes;
    nodes.count = count;
    nodes.nodes = arena_alloc(arena, sizeof(struct Node*) * count);
    for (size_t i = 0; i < count; i++)
        nodes.nodes[i] = in_nodes[i];

    insert_set_get_result(struct Nodes, arena->nodes_set, nodes);
    return nodes;
}

struct Strings strings(struct IrArena* arena, size_t count, const char* in_strs[])  {
    struct Strings tmp = {
        .count = count,
        .strings = in_strs,
    };
    const struct Strings* found = find_key_dict(struct Strings, arena->strings_set, tmp);
    if (found)
        return *found;

    struct Strings strings;
    strings.count = count;
    strings.strings = arena_alloc(arena, sizeof(const char*) * count);
    for (size_t i = 0; i < count; i++)
        strings.strings[i] = in_strs[i];

    insert_set_get_result(struct Strings, arena->strings_set, strings);
    return strings;
}

/// takes care of structural sharing
static const char* string_impl(struct IrArena* arena, size_t size, const char* zero_terminated) {
    const char** ptr = &zero_terminated;
    const char** found = find_key_dict(const char*, arena->string_set, ptr);
    if (found)
        return *found;

    char* new_str = (char*) arena_alloc(arena, strlen(zero_terminated) + 1);
    strncpy(new_str, zero_terminated, size);
    new_str[size] = '\0';

    insert_set_get_result(const char*, arena->string_set, new_str);
    return new_str;
}

const char* string_sized(struct IrArena* arena, size_t size, const char* str) {
    char new_str[size + 1];
    strncpy(new_str, str, size);
    new_str[size] = '\0';
    assert(strlen(new_str) == size);
    return string_impl(arena, size, str);
}

const char* string(struct IrArena* arena, const char* str) {
    return string_impl(arena, strlen(str), str);
}

KeyHash hash_nodes(struct Nodes* nodes) {
    uint32_t out[4];
    MurmurHash3_x64_128((nodes)->nodes, (int) (sizeof(struct Node*) * (nodes)->count), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

bool compare_nodes(struct Nodes* a, struct Nodes* b) {
    return a->count == b->count && memcmp(a->nodes, b->nodes, sizeof(struct Node*) * (a->count)) == 0;
}

KeyHash hash_strings(struct Strings* strings) {
    uint32_t out[4];
    MurmurHash3_x64_128(strings->strings, (int) (sizeof(const char*) * strings->count), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

bool compare_strings(struct Strings* a, struct Strings* b) {
    return a->count == b->count && memcmp(a->strings, b->strings, sizeof(const char*) * a->count) == 0;
}

KeyHash hash_string(const char** string) {
    uint32_t out[4];
    MurmurHash3_x64_128(*string, (int) strlen(*string), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

bool compare_string(const char** a, const char** b) {
    return strlen(*a) == strlen(*b) && strcmp(*a, *b) == 0;
}
