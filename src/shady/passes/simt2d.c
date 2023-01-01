#include "passes.h"

#include "../type.h"
#include "../rewrite.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    size_t width;
    const Node* mask;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case QualifiedType_TAG: {
            if (!node->payload.qualified_type.is_uniform) return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = pack_type(arena, (PackType) { .width = ctx->width, .element_type = rewrite_node(&ctx->rewriter, node->payload.qualified_type.type )})
            });
            goto rewrite;
        }
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            switch (op) {
                case quote_op: goto rewrite;
                case alloca_logical_op:
                case alloca_op: {
                    const Node* type = rewrite_node(&ctx->rewriter, first(node->payload.prim_op.type_arguments));
                    return prim_op(arena, (PrimOp) {
                        .op = op,
                        .type_arguments = singleton(maybe_packed_type_helper(type, ctx->width)),
                        .operands = empty(arena)
                    });
                }
                case subgroup_local_id_op: {
                    error("TODO")
                }
                default: break;
            }

            bool was_uniform = true;
            Nodes old_operands = node->payload.prim_op.operands;
            for (size_t i = 0; i < old_operands.count; i++)
                was_uniform &= is_qualified_type_uniform(old_operands.nodes[i]->type);
            Nodes new_type_arguments = rewrite_nodes(&ctx->rewriter, node->payload.prim_op.type_arguments);

            LARRAY(const Node*, new_operands, old_operands.count);
            // Nodes new_operands = rewrite_nodes(&ctx->rewriter, node->payload.prim_op.operands);
            for (size_t i = 0; i < old_operands.count; i++) {
                const Node* old_operand = old_operands.nodes[i];
                const Type* old_operand_type = old_operand->type;
                bool op_was_uniform = deconstruct_qualified_type(&old_operand_type);
                // assert(was_uniform || !op_was_uniform && "result was uniform implies=> operand was uniform");
                new_operands[i] = rewrite_node(&ctx->rewriter, old_operand);
                const Type* new_operand_type = new_operands[i]->type;
                if (op_was_uniform) {
                    LARRAY(const Node*, copies, ctx->width);
                    for (size_t j = 0; j < ctx->width; j++)
                        copies[j] = new_operands[i];
                    new_operand_type = pack_type(arena, (PackType) { .width = ctx->width, .element_type = get_unqualified_type(new_operand_type)});
                    new_operands[i] = composite(arena, new_operand_type, nodes(arena, ctx->width, copies));
                }
            }
            return prim_op(arena, (PrimOp) {
                .op = op,
                .type_arguments = new_type_arguments,
                .operands = nodes(arena, old_operands.count, new_operands)
            });
        }
        rewrite:
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void simt2d(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .width = config->subgroup_size,
        .mask = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
