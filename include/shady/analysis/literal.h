#ifndef SHADY_ANALYSIS_LITERAL_H
#define SHADY_ANALYSIS_LITERAL_H

#include "shady/ir/base.h"

const char* get_string_literal(IrArena*, const Node*);

typedef struct {
    bool enter_loads;
    bool allow_incompatible_types;
    bool assume_globals_immutability;
} NodeResolveConfig;

NodeResolveConfig default_node_resolve_config(void);
const Node* chase_ptr_to_source(const Node*, NodeResolveConfig config);
const Node* resolve_ptr_to_value(const Node* node, NodeResolveConfig config);

const Node* resolve_node_to_definition(const Node* node, NodeResolveConfig config);

#endif
