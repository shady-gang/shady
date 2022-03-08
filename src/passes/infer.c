#include "passes.h"

#include "../implem.h"
#include "../type.h"

#include <assert.h>

#define ctxparams struct IrArena* src_arena, struct IrArena* dst_arena
#define ctx src_arena, dst_arena

const struct Node* type_value(ctxparams, const struct Node* value, const struct Type* expected_type) {
    switch (value->tag) {
        case Variable_TAG: {
            const struct Type* inferred_type = expected_type;
            assert(value->payload.var.type);
            if (value->payload.var.type) {
                check_subtype(expected_type, value->payload.var.type);
                inferred_type = value->payload.var.type;
            }
            assert(resolve_divergence(inferred_type) != Unknown);
            return var(dst_arena, (struct Variable) {
                .name = value->payload.var.name,
                .type = inferred_type
            });
        }
        case UntypedNumber_TAG: {
            if (!expected_type) error("it is impossible to infer numbers without context");
        }
        default: error("not a value");
    }
}

const struct Node* type_decl(ctxparams, const struct Node* decl) {
    switch (decl->tag) {
        case VariableDecl_TAG:
        case Function_TAG:
        default: error("not a decl");
    }
}

struct Program infer_types(ctxparams, struct Program src_program) {
    /*const struct Node* new_top_level[src_program.declarations_and_definitions.count];
    for (size_t i = 0; i < src_program.declarations_and_definitions.count; i++) {
        new_top_level[i] = type_decl(ctx, src_program.declarations_and_definitions.nodes[i]);
    }
    return (struct Program) { nodes(dst_arena, src_program.declarations_and_definitions.count, new_top_level) };*/
    SHADY_NOT_IMPLEM
}
