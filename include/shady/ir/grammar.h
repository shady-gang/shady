#ifndef SHADY_IR_GRAMMAR_H
#define SHADY_IR_GRAMMAR_H

#include "shady/ir/base.h"

// These enums and structs are used in the node payloads so they must live here
// instead of in the relevant header

typedef enum {
    IntTy8,
    IntTy16,
    IntTy32,
    IntTy64,
} IntSizes;

enum {
    IntSizeMin = IntTy8,
    IntSizeMax = IntTy64,
};

typedef enum {
    FloatTy16,
    FloatTy32,
    FloatTy64
} FloatSizes;

typedef enum {
    NotSpecial,
    /// for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else
    MultipleReturn,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    DecorateBlock
} RecordSpecialFlag;

typedef struct BodyBuilder_ BodyBuilder;

// see primops.json
#include "primops_generated.h"

// see grammar.json
#include "grammar_generated.h"

bool shd_is_node_nominal(const Node* node);
const char* shd_get_node_tag_string(NodeTag tag);

#endif
