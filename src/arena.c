#include "ir.h"
#include "implem.h"
#include "type.h"

#include "dict.h"
#include "murmur3.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define alloc_size 1024 * 1024

KeyHash hash_nodes(Nodes* nodes);
bool compare_nodes(Nodes* a, Nodes* b);

KeyHash hash_strings(Strings* strings);
bool compare_strings(Strings* a, Strings* b);

KeyHash hash_string(const char** string);
bool compare_string(const char** a, const char** b);

KeyHash hash_node(const Node**);
bool compare_node(const Node** a, const Node** b);

IrArena* new_arena(IrConfig config) {
    IrArena* arena = malloc(sizeof(IrArena));
    *arena = (IrArena) {
        .nblocks = 0,
        .maxblocks = 256,
        .blocks = malloc(256 * sizeof(size_t)),
        .available = 0,
        .config = config,

        .next_free_id = 0,

        .node_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .string_set = new_set(const char*, (HashFn) hash_string, (CmpFn) compare_string),

        .nodes_set   = new_set(Nodes, (HashFn) hash_nodes, (CmpFn) compare_nodes),
        .strings_set = new_set(Strings, (HashFn) hash_strings, (CmpFn) compare_strings),
    };
    for (int i = 0; i < arena->maxblocks; i++)
        arena->blocks[i] = NULL;
    return arena;
}

void destroy_arena(IrArena* arena) {
    destroy_dict(arena->strings_set);
    destroy_dict(arena->string_set);
    destroy_dict(arena->nodes_set);
    destroy_dict(arena->node_set);
    for (int i = 0; i < arena->nblocks; i++) {
        free(arena->blocks[i]);
    }
    free(arena->blocks);
    free(arena);
}

VarId fresh_id(IrArena* arena) {
    return arena->next_free_id++;
}

void* arena_alloc(IrArena* arena, size_t size) {
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

Nodes nodes(IrArena* arena, size_t count, const Node* in_nodes[]) {
    Nodes tmp = {
        .count = count,
        .nodes = in_nodes
    };
    const Nodes* found = find_key_dict(Nodes, arena->nodes_set, tmp);
    if (found)
        return *found;

    Nodes nodes;
    nodes.count = count;
    nodes.nodes = arena_alloc(arena, sizeof(Node*) * count);
    for (size_t i = 0; i < count; i++)
        nodes.nodes[i] = in_nodes[i];

    insert_set_get_result(Nodes, arena->nodes_set, nodes);
    return nodes;
}

Strings strings(IrArena* arena, size_t count, const char* in_strs[])  {
    Strings tmp = {
        .count = count,
        .strings = in_strs,
    };
    const Strings* found = find_key_dict(Strings, arena->strings_set, tmp);
    if (found)
        return *found;

    Strings strings;
    strings.count = count;
    strings.strings = arena_alloc(arena, sizeof(const char*) * count);
    for (size_t i = 0; i < count; i++)
        strings.strings[i] = in_strs[i];

    insert_set_get_result(Strings, arena->strings_set, strings);
    return strings;
}

/// takes care of structural sharing
static const char* string_impl(IrArena* arena, size_t size, const char* zero_terminated) {
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

const char* string_sized(IrArena* arena, size_t size, const char* str) {
    LARRAY(char, new_str, size + 1);

    strncpy(new_str, str, size);
    new_str[size] = '\0';
    assert(strlen(new_str) == size);
    return string_impl(arena, size, str);
}

const char* string(IrArena* arena, const char* str) {
    return string_impl(arena, strlen(str), str);
}

KeyHash hash_nodes(Nodes* nodes) {
    uint32_t out[4];
    MurmurHash3_x64_128((nodes)->nodes, (int) (sizeof(Node*) * (nodes)->count), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

bool compare_nodes(Nodes* a, Nodes* b) {
    return a->count == b->count && memcmp(a->nodes, b->nodes, sizeof(Node*) * (a->count)) == 0;
}

KeyHash hash_strings(Strings* strings) {
    uint32_t out[4];
    MurmurHash3_x64_128(strings->strings, (int) (sizeof(const char*) * strings->count), 0x1234567, &out);
    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

bool compare_strings(Strings* a, Strings* b) {
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
