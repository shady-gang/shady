#ifndef SHADY_IR_GRAMMAR_H
#define SHADY_IR_GRAMMAR_H

#include "shady/ir/base.h"

// These structs are referenced in the actual grammar

typedef enum {
    NotSpecial,
    /// for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else
    MultipleReturn,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    DecorateBlock
} RecordSpecialFlag;

#include "shady/ir/int.h"
#include "shady/ir/float.h"
#include "shady/ir/primop.h"

typedef struct BodyBuilder_ BodyBuilder;

// see grammar.json
#include "grammar_generated.h"

#endif
