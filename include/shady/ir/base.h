#ifndef SHADY_IR_BASE_H
#define SHADY_IR_BASE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __GNUC__
#define SHADY_DESIGNATED_INIT __attribute__((designated_init))
#else
#define SHADY_DESIGNATED_INIT
#endif

typedef struct IrArena_ IrArena;
typedef struct Module_ Module;
typedef struct Node_ Node;
typedef struct Node_ Type;
typedef uint32_t NodeId;
typedef const char* String;

typedef struct Nodes_ {
    size_t count;
    const Node** nodes;
} Nodes;

typedef struct Strings_ {
    size_t count;
    String* strings;
} Strings;

Nodes shd_nodes(IrArena*, size_t count, const Node*[]);
Strings shd_strings(IrArena* arena, size_t count, const char** in_strs);

Nodes shd_empty(IrArena* a);
Nodes shd_singleton(const Node* n);

#define mk_nodes(arena, ...) shd_nodes(arena, sizeof((const Node*[]) { __VA_ARGS__ }) / sizeof(const Node*), (const Node*[]) { __VA_ARGS__ })

const Node* shd_first(Nodes nodes);

Nodes shd_nodes_append(IrArena*, Nodes, const Node*);
Nodes shd_nodes_prepend(IrArena*, Nodes, const Node*);
Nodes shd_concat_nodes(IrArena* arena, Nodes a, Nodes b);
Nodes shd_change_node_at_index(IrArena* arena, Nodes old, size_t i, const Node* n);
bool shd_find_in_nodes(Nodes nodes, const Node* n);

String shd_string_sized(IrArena*, size_t size, const char* start);
String shd_string(IrArena*, const char*);

// see also: format_string in util.h
String shd_fmt_string_irarena(IrArena* arena, const char* str, ...);
String shd_make_unique_name(IrArena* arena, const char* str);

#endif
