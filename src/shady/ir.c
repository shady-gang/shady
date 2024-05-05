#include "ir_private.h"
#include "portability.h"

#include "list.h"
#include "dict.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

static KeyHash hash_nodes(Nodes* nodes);
bool compare_nodes(Nodes* a, Nodes* b);

static KeyHash hash_strings(Strings* strings);
static bool compare_strings(Strings* a, Strings* b);

KeyHash hash_string(const char** string);
bool compare_string(const char** a, const char** b);

KeyHash hash_node(const Node**);
bool compare_node(const Node** a, const Node** b);

IrArena* new_ir_arena(ArenaConfig config) {
    IrArena* arena = malloc(sizeof(IrArena));
    *arena = (IrArena) {
        .arena = new_arena(),
        .config = config,

        .modules = new_list(Module*),

        .node_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .string_set = new_set(const char*, (HashFn) hash_string, (CmpFn) compare_string),

        .nodes_set   = new_set(Nodes, (HashFn) hash_nodes, (CmpFn) compare_nodes),
        .strings_set = new_set(Strings, (HashFn) hash_strings, (CmpFn) compare_strings),

        .ids = new_growy(),
    };
    return arena;
}

const Node* get_node_by_id(const IrArena* a, NodeId id) {
    return ((const Node**) growy_data(a->ids))[id];
}

void destroy_ir_arena(IrArena* arena) {
    for (size_t i = 0; i < entries_count_list(arena->modules); i++) {
        destroy_module(read_list(Module*, arena->modules)[i]);
    }

    destroy_list(arena->modules);
    destroy_dict(arena->strings_set);
    destroy_dict(arena->string_set);
    destroy_dict(arena->nodes_set);
    destroy_dict(arena->node_set);
    destroy_arena(arena->arena);
    destroy_growy(arena->ids);
    free(arena);
}

ArenaConfig get_arena_config(const IrArena* a) {
    return a->config;
}

NodeId allocate_node_id(IrArena* arena, const Node* n) {
    growy_append_object(arena->ids, n);
    return growy_size(arena->ids) / sizeof(const Node*);
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
    nodes.nodes = arena_alloc(arena->arena, sizeof(Node*) * count);
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
    strings.strings = arena_alloc(arena->arena, sizeof(const char*) * count);
    for (size_t i = 0; i < count; i++)
        strings.strings[i] = in_strs[i];

    insert_set_get_result(Strings, arena->strings_set, strings);
    return strings;
}

Nodes empty(IrArena* a) {
    return nodes(a, 0, NULL);
}

Nodes singleton(const Type* type) {
    IrArena* arena = type->arena;
    const Type* arr[] = { type };
    return nodes(arena, 1, arr);
}

const Node* first(Nodes nodes) {
    assert(nodes.count > 0);
    return nodes.nodes[0];
}

Nodes append_nodes(IrArena* arena, Nodes old, const Node* new) {
    LARRAY(const Node*, tmp, old.count + 1);
    for (size_t i = 0; i < old.count; i++)
        tmp[i] = old.nodes[i];
    tmp[old.count] = new;
    return nodes(arena, old.count + 1, tmp);
}

Nodes prepend_nodes(IrArena* arena, Nodes old, const Node* new) {
    LARRAY(const Node*, tmp, old.count + 1);
    for (size_t i = 0; i < old.count; i++)
        tmp[i + 1] = old.nodes[i];
    tmp[0] = new;
    return nodes(arena, old.count + 1, tmp);
}

Nodes concat_nodes(IrArena* arena, Nodes a, Nodes b) {
    LARRAY(const Node*, tmp, a.count + b.count);
    size_t j = 0;
    for (size_t i = 0; i < a.count; i++)
        tmp[j++] = a.nodes[i];
    for (size_t i = 0; i < b.count; i++)
        tmp[j++] = b.nodes[i];
    assert(j == a.count + b.count);
    return nodes(arena, j, tmp);
}

Nodes change_node_at_index(IrArena* arena, Nodes old, size_t i, const Node* n) {
    LARRAY(const Node*, tmp, old.count);
    for (size_t j = 0; j < old.count; j++)
        tmp[j] = old.nodes[j];
    tmp[i] = n;
    return nodes(arena, old.count, tmp);
}

bool find_in_nodes(Nodes nodes, const Node* n) {
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
    const char** found = find_key_dict(const char*, arena->string_set, ptr);
    if (found)
        return *found;

    char* new_str = (char*) arena_alloc(arena->arena, strlen(zero_terminated) + 1);
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
    if (!str)
        return NULL;
    return string_impl(arena, strlen(str), str);
}

// TODO merge with strings()
Strings import_strings(IrArena* dst_arena, Strings old_strings) {
    size_t count = old_strings.count;
    LARRAY(String, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = string(dst_arena, old_strings.strings[i]);
    return strings(dst_arena, count, arr);
}

void format_string_internal(const char* str, va_list args, void* uptr, void callback(void*, size_t, char*));

typedef struct { IrArena* a; const char** result; } InternInArenaPayload;

static void intern_in_arena(InternInArenaPayload* uptr, size_t len, char* tmp) {
    const char* interned = string_impl(uptr->a, len, tmp);
    *uptr->result = interned;
}

String format_string_interned(IrArena* arena, const char* str, ...) {
    String result = NULL;
    InternInArenaPayload p = { .a = arena, .result = &result };
    va_list args;
    va_start(args, str);
    format_string_internal(str, args, &p, (void(*)(void*, size_t, char*)) intern_in_arena);
    va_end(args);
    return result;
}

const char* unique_name(IrArena* arena, const char* str) {
    return format_string_interned(arena, "%s_%d", str, allocate_node_id(arena, NULL));
}

KeyHash hash_nodes(Nodes* nodes) {
    return hash_murmur(nodes->nodes, sizeof(const Node*) * nodes->count);
}

bool compare_nodes(Nodes* a, Nodes* b) {
    if (a->count != b->count) return false;
    if (a->count == 0) return true;
    assert(a->nodes != NULL && b->nodes != NULL);
    return memcmp(a->nodes, b->nodes, sizeof(Node*) * (a->count)) == 0; // actually compare the data
}

KeyHash hash_strings(Strings* strings) {
    return hash_murmur(strings->strings, sizeof(char*) * strings->count);
}

bool compare_strings(Strings* a, Strings* b) {
    if (a->count != b->count) return false;
    if (a->count == 0) return true;
    assert(a->strings != NULL && b->strings != NULL);
    return memcmp(a->strings, b->strings, sizeof(const char*) * a->count) == 0;
}

KeyHash hash_string(const char** string) {
    if (!*string)
        return 0;
    return hash_murmur(*string, strlen(*string));
}

bool compare_string(const char** a, const char** b) {
    if (*a == NULL || *b == NULL)
        return (!*a) == (!*b);
    return strlen(*a) == strlen(*b) && strcmp(*a, *b) == 0;
}

Nodes list_to_nodes(IrArena* arena, struct List* list) {
    return nodes(arena, entries_count_list(list), read_list(const Node*, list));
}
