#include "ir_private.h"
#include "portability.h"

#include "list.h"
#include "dict.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

KeyHash shd_hash_nodes(Nodes* nodes);
bool shd_compare_nodes(Nodes* a, Nodes* b);

KeyHash shd_hash_strings(Strings* strings);
bool shd_compare_strings(Strings* a, Strings* b);

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

KeyHash shd_hash_node(const Node**);
bool shd_compare_node(const Node** a, const Node** b);

IrArena* shd_new_ir_arena(const ArenaConfig* config) {
    IrArena* arena = malloc(sizeof(IrArena));
    *arena = (IrArena) {
        .arena = shd_new_arena(),
        .config = *config,

        .modules = shd_new_list(Module*),

        .node_set = shd_new_set(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .string_set = shd_new_set(const char*, (HashFn) shd_hash_string, (CmpFn) shd_compare_string),

        .nodes_set   = shd_new_set(Nodes, (HashFn) shd_hash_nodes, (CmpFn) shd_compare_nodes),
        .strings_set = shd_new_set(Strings, (HashFn) shd_hash_strings, (CmpFn) shd_compare_strings),

        .ids = shd_new_growy(),
    };
    return arena;
}

const Node* shd_get_node_by_id(const IrArena* a, NodeId id) {
    return ((const Node**) shd_growy_data(a->ids))[id];
}

const ArenaConfig* shd_ir_arena_get_config(const IrArena* a) {
    return &a->config;
}

void shd_destroy_ir_arena(IrArena* arena) {
    for (size_t i = 0; i < shd_list_count(arena->modules); i++) {
        shd_destroy_module(shd_read_list(Module*, arena->modules)[i]);
    }

    shd_destroy_list(arena->modules);
    shd_destroy_dict(arena->strings_set);
    shd_destroy_dict(arena->string_set);
    shd_destroy_dict(arena->nodes_set);
    shd_destroy_dict(arena->node_set);
    shd_destroy_arena(arena->arena);
    shd_destroy_growy(arena->ids);
    free(arena);
}

const ArenaConfig* shd_get_arena_config(const IrArena* a) {
    return &a->config;
}

NodeId _shd_allocate_node_id(IrArena* arena, const Node* n) {
    shd_growy_append_object(arena->ids, n);
    return shd_growy_size(arena->ids) / sizeof(const Node*);
}

Nodes shd_nodes(IrArena* arena, size_t count, const Node* in_nodes[]) {
    Nodes tmp = {
        .count = count,
        .nodes = in_nodes
    };
    const Nodes* found = shd_dict_find_key(Nodes, arena->nodes_set, tmp);
    if (found)
        return *found;

    Nodes nodes;
    nodes.count = count;
    nodes.nodes = shd_arena_alloc(arena->arena, sizeof(Node*) * count);
    for (size_t i = 0; i < count; i++)
        nodes.nodes[i] = in_nodes[i];

    shd_set_insert_get_result(Nodes, arena->nodes_set, nodes);
    return nodes;
}

Strings shd_strings(IrArena* arena, size_t count, const char* in_strs[]) {
    Strings tmp = {
        .count = count,
        .strings = in_strs,
    };
    const Strings* found = shd_dict_find_key(Strings, arena->strings_set, tmp);
    if (found)
        return *found;

    Strings strings;
    strings.count = count;
    strings.strings = shd_arena_alloc(arena->arena, sizeof(const char*) * count);
    for (size_t i = 0; i < count; i++)
        strings.strings[i] = in_strs[i];

    shd_set_insert_get_result(Strings, arena->strings_set, strings);
    return strings;
}

Nodes shd_empty(IrArena* a) {
    return shd_nodes(a, 0, NULL);
}

Nodes shd_singleton(const Node* n) {
    IrArena* arena = n->arena;
    const Type* arr[] = { n };
    return shd_nodes(arena, 1, arr);
}

const Node* shd_first(Nodes nodes) {
    assert(nodes.count > 0);
    return nodes.nodes[0];
}

Nodes shd_nodes_append(IrArena* arena, Nodes old, const Node* new) {
    LARRAY(const Node*, tmp, old.count + 1);
    for (size_t i = 0; i < old.count; i++)
        tmp[i] = old.nodes[i];
    tmp[old.count] = new;
    return shd_nodes(arena, old.count + 1, tmp);
}

Nodes shd_nodes_prepend(IrArena* arena, Nodes old, const Node* new) {
    LARRAY(const Node*, tmp, old.count + 1);
    for (size_t i = 0; i < old.count; i++)
        tmp[i + 1] = old.nodes[i];
    tmp[0] = new;
    return shd_nodes(arena, old.count + 1, tmp);
}

Nodes shd_concat_nodes(IrArena* arena, Nodes a, Nodes b) {
    LARRAY(const Node*, tmp, a.count + b.count);
    size_t j = 0;
    for (size_t i = 0; i < a.count; i++)
        tmp[j++] = a.nodes[i];
    for (size_t i = 0; i < b.count; i++)
        tmp[j++] = b.nodes[i];
    assert(j == a.count + b.count);
    return shd_nodes(arena, j, tmp);
}

Nodes shd_change_node_at_index(IrArena* arena, Nodes old, size_t i, const Node* n) {
    LARRAY(const Node*, tmp, old.count);
    for (size_t j = 0; j < old.count; j++)
        tmp[j] = old.nodes[j];
    tmp[i] = n;
    return shd_nodes(arena, old.count, tmp);
}

bool shd_find_in_nodes(Nodes nodes, const Node* n) {
    for (size_t i = 0; i < nodes.count; i++)
        if (nodes.nodes[i] == n)
            return true;
    return false;
}

/// takes care of structural sharing
static const char* string_impl(IrArena* arena, size_t size, const char* zero_terminated) {
    if (!zero_terminated)
        return NULL;
    const char* ptr = zero_terminated;
    const char** found = shd_dict_find_key(const char*, arena->string_set, ptr);
    if (found)
        return *found;

    char* new_str = (char*) shd_arena_alloc(arena->arena, strlen(zero_terminated) + 1);
    strncpy(new_str, zero_terminated, size);
    new_str[size] = '\0';

    shd_set_insert_get_result(const char*, arena->string_set, new_str);
    return new_str;
}

const char* shd_string_sized(IrArena* arena, size_t size, const char* str) {
    LARRAY(char, new_str, size + 1);

    strncpy(new_str, str, size);
    new_str[size] = '\0';
    assert(strlen(new_str) == size);
    return string_impl(arena, size, str);
}

const char* shd_string(IrArena* arena, const char* str) {
    if (!str)
        return NULL;
    return string_impl(arena, strlen(str), str);
}

// TODO merge with strings()
Strings _shd_import_strings(IrArena* dst_arena, Strings old_strings) {
    size_t count = old_strings.count;
    LARRAY(String, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = shd_string(dst_arena, old_strings.strings[i]);
    return shd_strings(dst_arena, count, arr);
}

void shd_format_string_internal(const char* str, va_list args, void* uptr, void callback(void*, size_t, char*));

typedef struct {
    IrArena* a;
    const char** result;
} InternInArenaPayload;

static void intern_in_arena(InternInArenaPayload* uptr, size_t len, char* tmp) {
    const char* interned = string_impl(uptr->a, len, tmp);
    *uptr->result = interned;
}

String shd_fmt_string_irarena(IrArena* arena, const char* str, ...) {
    String result = NULL;
    InternInArenaPayload p = { .a = arena, .result = &result };
    va_list args;
    va_start(args, str);
    shd_format_string_internal(str, args, &p, (void (*)(void*, size_t, char*)) intern_in_arena);
    va_end(args);
    return result;
}

const char* shd_make_unique_name(IrArena* arena, const char* str) {
    return shd_fmt_string_irarena(arena, "%s_%d", str, _shd_allocate_node_id(arena, NULL));
}

KeyHash shd_hash_nodes(Nodes* nodes) {
    return shd_hash(nodes->nodes, sizeof(const Node*) * nodes->count);
}

bool shd_compare_nodes(Nodes* a, Nodes* b) {
    if (a->count != b->count) return false;
    if (a->count == 0) return true;
    assert(a->nodes != NULL && b->nodes != NULL);
    return memcmp(a->nodes, b->nodes, sizeof(Node*) * (a->count)) == 0; // actually compare the data
}

KeyHash shd_hash_strings(Strings* strings) {
    return shd_hash(strings->strings, sizeof(char*) * strings->count);
}

bool shd_compare_strings(Strings* a, Strings* b) {
    if (a->count != b->count) return false;
    if (a->count == 0) return true;
    assert(a->strings != NULL && b->strings != NULL);
    return memcmp(a->strings, b->strings, sizeof(const char*) * a->count) == 0;
}

KeyHash shd_hash_string(const char** string) {
    if (!*string)
        return 0;
    return shd_hash(*string, strlen(*string));
}

bool shd_compare_string(const char** a, const char** b) {
    if (*a == NULL || *b == NULL)
        return (!*a) == (!*b);
    return strlen(*a) == strlen(*b) && strcmp(*a, *b) == 0;
}

Nodes shd_list_to_nodes(IrArena* arena, struct List* list) {
    return shd_nodes(arena, shd_list_count(list), shd_read_list(const Node*, list));
}
