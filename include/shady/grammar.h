#ifndef SHADY_IR_H
#error "do not include this file by itself, include shady/ir.h instead"
#endif

typedef enum DivergenceQualifier_ {
    Unknown,
    Uniform,
    Varying
} DivergenceQualifier;

typedef enum {
    NotSpecial,
    /// for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else
    MultipleReturn,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    DecorateBlock
} RecordSpecialFlag;

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

#define EXECUTION_MODELS(EM) \
EM(Compute,  1) \
EM(Fragment, 0) \
EM(Vertex,   0) \

typedef enum {
    EmNone,
#define EM(name, _) Em##name,
EXECUTION_MODELS(EM)
#undef EM
} ExecutionModel;

ExecutionModel execution_model_from_string(const char*);

//////////////////////////////// Generated definitions ////////////////////////////////

// see grammar.json
#include "grammar_generated.h"

extern const char* node_tags[];
extern const bool node_type_has_payload[];

//////////////////////////////// Node categories ////////////////////////////////

inline static bool is_nominal(const Node* node) {
    NodeTag tag = node->tag;
    if (node->tag == PrimOp_TAG && has_primop_got_side_effects(node->payload.prim_op.op))
        return true;
    return tag == Function_TAG || tag == BasicBlock_TAG || tag == Constant_TAG || tag == Variable_TAG || tag == GlobalVariable_TAG || tag == NominalType_TAG || tag == AnonLambda_TAG;
}

inline static bool is_anonymous_lambda(const Node* node) { return node->tag == AnonLambda_TAG; }
inline static bool is_function(const Node* node) { return node->tag == Function_TAG; }
