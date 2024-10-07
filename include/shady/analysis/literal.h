#ifndef SHADY_ANALYSIS_LITERAL_H
#define SHADY_ANALYSIS_LITERAL_H

#include "shady/ir/base.h"

const char* shd_get_string_literal(IrArena* arena, const Node* node);

typedef struct {
    bool enter_loads;
    bool allow_incompatible_types;
    bool assume_globals_immutability;
} NodeResolveConfig;

NodeResolveConfig shd_default_node_resolve_config(void);
const Node* shd_chase_ptr_to_source(const Node* ptr, NodeResolveConfig config);
const Node* shd_resolve_ptr_to_value(const Node* ptr, NodeResolveConfig config);

const Node* shd_resolve_node_to_definition(const Node* node, NodeResolveConfig config);

#endif
