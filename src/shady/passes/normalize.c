#include "pass.h"

#include "../type.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct Context_ {
    Rewriter rewriter;
    BodyBuilder* bb;
    struct Dict* bound;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* force_to_be_value(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;
    IrArena* a = ctx->rewriter.dst_arena;

    if (is_instruction(node)) {
        const Node** found = find_value_dict(const Node*, const Node*, ctx->bound, node);
        if (found)
            return *found;
        const Node* let_bound = process_node(ctx, node);
        insert_dict_and_get_result(const Node*, const Node*, ctx->bound, node, let_bound);
        return first(bind_instruction_outputs_count(ctx->bb, let_bound, 1));
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
        case Param_TAG: return find_processed(&ctx->rewriter, node);
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
        case NcStructured_construct:
            break;
    }
    return process_node(ctx, node);
}

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    // add a builder to each abstraction...
    switch (node->tag) {
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;
            ctx2.bound = new_dict(const Node*, const Node*, (HashFn) hash_node, (CmpFn) compare_node);
            new->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
            destroy_dict(ctx2.bound);
            return new;
        }
        case BasicBlock_TAG: {
            Node* new = basic_block(a, recreate_params(&ctx->rewriter, node->payload.basic_block.params), node->payload.basic_block.name);
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
            Nodes new_params = recreate_params(&ctx->rewriter, node->payload.case_.params);
            register_processed_list(&ctx->rewriter, node->payload.case_.params, new_params);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

            const Node* new_body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.case_.body));
            return case_(a, new_params, new_body);
        }
        case Let_TAG: {
            const Node* oinstr = get_let_instruction(node);
            //const Node* found = search_processed(r, oinstr);
            const Node** found = find_value_dict(const Node*, const Node*, ctx->bound, node);
            if (found)
                return rewrite_node(r, node->payload.let.in);
            const Node* ninstr = rewrite_op(r, NcInstruction, "instruction", oinstr);
            insert_dict_and_get_result(const Node*, const Node*, ctx->bound, oinstr, ninstr);
            register_processed(r, oinstr, ninstr);
            bind_instruction_outputs_count(ctx->bb, ninstr, 0);
            return rewrite_node(r, node->payload.let.in);

            //const Node* new = recreate_node_identity(r, node);
            //register_processed(r, node, new);
            //return new;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* normalize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.check_op_classes = true;
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_op_rewriter(src, dst, (RewriteOpFn) process_op),
        .bb = NULL,
        .bound = NULL,
    };

    ctx.rewriter.config.search_map = false;
    ctx.rewriter.config.write_map = false;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
