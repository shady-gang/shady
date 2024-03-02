#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    Node** builtins;
} Context;

static const Type* get_req_cast(Context* ctx, const Node* src) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (src->tag) {
        case GlobalVariable_TAG: {
            GlobalVariable global_variable = src->payload.global_variable;
            const Node* ba = lookup_annotation_list(global_variable.annotations, "Builtin");
            if (ba) {
                Builtin b = get_builtin_by_name(get_annotation_string_payload(ba));
                assert(b != BuiltinsCount);
                const Type* expected_t = get_builtin_type(a, b);
                const Type* actual_t = rewrite_node(&ctx->rewriter, src)->payload.global_variable.type;
                if (expected_t != actual_t) {
                    log_string(INFO, "normalize_builtins: found builtin decl '%s' not matching expected type: '", global_variable.name);
                    log_node(INFO, expected_t);
                    log_string(INFO, "', got '");
                    log_node(INFO, actual_t);
                    log_string(INFO, "'.");
                    return actual_t;
                }
            }
            break;
        }
        case RefDecl_TAG: return get_req_cast(ctx, src->payload.ref_decl.decl);
        case PrimOp_TAG: {
            PrimOp prim_op = src->payload.prim_op;
            if (prim_op.op == lea_op) {
                const Type* src_req_cast = get_req_cast(ctx, first(prim_op.operands));
                if (src_req_cast) {
                    bool u = deconstruct_qualified_type(&src_req_cast);
                    enter_composite(&src_req_cast, &u, nodes(a, prim_op.operands.count - 2, &prim_op.operands.nodes[2]), false);
                    return src_req_cast;
                }
            }
        }
        default: break;
    }

    return NULL;
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case GlobalVariable_TAG: {
            GlobalVariable global_variable = node->payload.global_variable;
            const Node* ba = lookup_annotation_list(global_variable.annotations, "Builtin");
            if (ba) {
                Builtin b = get_builtin_by_name(get_annotation_string_payload(ba));
                assert(b != BuiltinsCount);
                if (ctx->builtins[b])
                    return ctx->builtins[b];
                const Type* t = get_builtin_type(a, b);
                Node* ndecl = global_var(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, global_variable.annotations), t, global_variable.name, get_builtin_as(b));
                register_processed(&ctx->rewriter, node, ndecl);
                // no 'init' for builtins, right ?
                assert(!global_variable.init);
                ctx->builtins[b] = ndecl;
                return ndecl;
            }
        }
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            switch (op) {
                case load_op: {
                    const Type* req_cast = get_req_cast(ctx, first(node->payload.prim_op.operands));
                    if (req_cast) {
                        assert(is_data_type(req_cast));
                        BodyBuilder* bb = begin_body(a);
                        const Node* r = first(bind_instruction(bb, recreate_node_identity(&ctx->rewriter, node)));
                        const Node* r2 = first(gen_primop(bb, reinterpret_op, singleton(req_cast), singleton(r)));
                        return yield_values_and_wrap_in_block(bb, singleton(r2));
                    }
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* normalize_builtins(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    aconfig.validate_builtin_types = true;
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .builtins = calloc(sizeof(Node*), BuiltinsCount)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    free(ctx.builtins);
    return dst;
}
