#ifndef SHADY_IR_GRAMMAR_H
#define SHADY_IR_GRAMMAR_H

#include "shady/ir/base.h"
#include "shady/ir/enum.h"

// These enums and structs are used in the node payloads so they must live here
// instead of in the relevant header

typedef struct BodyBuilder_ BodyBuilder;

// see primops.json
#include "primops_generated.h"

// see grammar.json
#include "grammar_generated.h"

bool shd_is_node_nominal(const Node* node);
bool shd_is_node_tag_recursive(NodeTag tag);
const char* shd_get_node_tag_string(NodeTag tag);

#endif
