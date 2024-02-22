#include "shady/ir.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    BodyBuilder* bb;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* force_to_be_value(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;
    IrArena* a = ctx->rewriter.dst_arena;

    if (is_instruction(node)) {
        const Node* let_bound;
        let_bound = process_node(ctx, node);
        return first(bind_instruction_outputs_count(ctx->bb, let_bound, 1, NULL));
    }

    switch (node->tag) {
        // All decls map to refdecl/fnaddr
        case Constant_TAG:
        case GlobalVariable_TAG: {
            return ref_decl_helper(a, process_node(ctx, node));
        }
        case Function_TAG: {
            return fn_addr_helper(a, process_node(ctx, node));
        }
        case Variable_TAG: return find_processed(&ctx->rewriter, node);
        default:
            break;
    }

    assert(is_value(node));
    const Node* value = process_node(ctx, node);
    assert(is_value(value));
    return value;
}

static const Node* process_op(Context* ctx, NodeClass op_class, SHADY_UNUSED String op_name, const Node* node) {
    if (node == NULL) return NULL;
    IrArena* a = ctx->rewriter.dst_arena;
    switch (op_class) {
        case NcType: {
            switch (node->tag) {
                case NominalType_TAG: {
                    return type_decl_ref(ctx->rewriter.dst_arena, (TypeDeclRef) {
                            .decl = process_node(ctx, node),
                    });
                }
                default: break;
            }
            assert(is_type(node));
            const Node* type = process_node(ctx, node);
            assert(is_type(type));
            return type;
        }
        case NcValue:
            return force_to_be_value(ctx, node);
        case NcVariable:
            break;
        case NcInstruction: {
            if (is_instruction(node))
                return process_node(ctx, node);
            const Node* val = force_to_be_value(ctx, node);
            return quote_helper(a, singleton(val));
        }
        case NcTerminator:
            break;
        case NcDeclaration:
            break;
        case NcCase:
            break;
        case NcBasic_block:
            break;
        case NcAnnotation:
            break;
        case NcJump:
            break;
    }
    return process_node(ctx, node);
}

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* a = ctx->rewriter.dst_arena;

    // add a builder to each abstraction...
    switch (node->tag) {
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

            new->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
            return new;
        }
        case BasicBlock_TAG: {
            Node* new = basic_block(a, (Node*) rewrite_node(&ctx->rewriter, node->payload.basic_block.fn), recreate_variables(&ctx->rewriter, node->payload.basic_block.params), node->payload.basic_block.name);
            register_processed(&ctx->rewriter, node, new);
            register_processed_list(&ctx->rewriter, node->payload.basic_block.params, new->payload.basic_block.params);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;
            new->payload.basic_block.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.basic_block.body));
            return new;
        }
        case Case_TAG: {
            Nodes new_params = recreate_variables(&ctx->rewriter, node->payload.case_.params);
            register_processed_list(&ctx->rewriter, node->payload.case_.params, new_params);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

            const Node* new_body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.case_.body));
            return case_(a, new_params, new_body);
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* normalize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    aconfig.check_op_classes = true;
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) NULL),
        .bb = NULL,
    };

    ctx.rewriter.config.search_map = false;
    ctx.rewriter.config.write_map = false;
    ctx.rewriter.rewrite_op_fn = (RewriteOpFn) process_op;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
