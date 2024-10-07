#include "shady/pass.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* force_to_be_value(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        // All decls map to refdecl/fnaddr
        case Constant_TAG:
        case GlobalVariable_TAG: {
            return ref_decl_helper(a, process_node(ctx, node));
        }
        case Function_TAG: {
            return fn_addr_helper(a, process_node(ctx, node));
        }
        case Param_TAG: return shd_find_processed(&ctx->rewriter, node);
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
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
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
        case NcParam:
            break;
        case NcInstruction: {
            if (is_instruction(node)) {
                const Node* new = process_node(ctx, node);
                //register_processed(r, node, new);
                return new;
            }
            const Node* val = force_to_be_value(ctx, node);
            return val;
        }
        case NcTerminator:
            break;
        case NcDeclaration:
            break;
        case NcBasic_block:
            break;
        case NcAnnotation:
            break;
        case NcJump:
            break;
        case NcStructured_construct:
            break;
    }
    return process_node(ctx, node);
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node** already_done = shd_search_processed(&ctx->rewriter, node);
    if (already_done)
        return *already_done;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    // add a builder to each abstraction...
    switch (node->tag) {
        // case Let_TAG: {
        //     const Node* ninstr = rewrite_op(r, NcInstruction, "instruction", get_let_instruction(node));
        //     register_processed(r, get_let_instruction(node), ninstr);
        //     return let(a, ninstr, rewrite_op(r, NcTerminator, "in", node->payload.let.in));
        // }
        default: break;
    }

    const Node* new = shd_recreate_node(&ctx->rewriter, node);
    if (is_instruction(new))
        shd_register_processed(r, node, new);
    return new;
}

Module* slim_pass_normalize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.check_op_classes = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process_op),
    };

    ctx.rewriter.config.search_map = false;
    ctx.rewriter.config.write_map = false;

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
