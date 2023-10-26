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

static const Node* widen(Context* ctx, const Node* value) {
    IrArena* a = ctx->rewriter.dst_arena;
    LARRAY(const Node*, copies, ctx->width);
    for (size_t j = 0; j < ctx->width; j++)
        copies[j] = value;
    const Type* type = pack_type(a, (PackType) { .width = ctx->width, .element_type = get_unqualified_type(value->type)});
    return composite_helper(a, type, nodes(a, ctx->width, copies));
}

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case QualifiedType_TAG: {
            if (!node->payload.qualified_type.is_uniform) return qualified_type(a, (QualifiedType) {
                .is_uniform = true,
                .type = pack_type(a, (PackType) { .width = ctx->width, .element_type = rewrite_node(&ctx->rewriter, node->payload.qualified_type.type )})
            });
            goto rewrite;
        }
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            switch (op) {
                case quote_op: goto rewrite;
                case alloca_logical_op: {
                    BodyBuilder* bb = begin_body(a);
                    const Node* type = rewrite_node(&ctx->rewriter, first(node->payload.prim_op.type_arguments));
                    LARRAY(const Node*, allocated, ctx->width);
                    for (size_t i = 0; i < ctx->width; i++) {
                        allocated[i] = first(bind_instruction_named(bb, prim_op(a, (PrimOp) {
                                .op = op,
                                .type_arguments = singleton(type),
                                //.type_arguments = singleton(maybe_packed_type_helper(type, ctx->width)),
                                .operands = empty(a)
                        }), (String[]) {"allocated"}));
                    }
                    //return yield_values_and_wrap_in_control(bb, singleton(widen(ctx, allocated)));
                    const Node* result_type = maybe_packed_type_helper(ptr_type(a, (PtrType) { .address_space = AsSPVFunctionLogical, .pointed_type = type }), ctx->width);
                    const Node* packed = composite_helper(a, result_type, nodes(a, ctx->width, allocated));
                    return yield_values_and_wrap_in_block(bb, singleton(packed));
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
                if (op_was_uniform)
                    new_operands[i] = widen(ctx, new_operands[i]);
            }
            return prim_op(a, (PrimOp) {
                .op = op,
                .type_arguments = new_type_arguments,
                .operands = nodes(a, old_operands.count, new_operands)
            });
        }
        rewrite:
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

Module* simt2d(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    aconfig.is_simt = false;
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .width = config->specialization.subgroup_size,
        .mask = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
