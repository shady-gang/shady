#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

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
        case Lea_TAG: {
            const Type* src_req_cast = get_req_cast(ctx, src->payload.lea.ptr);
            if (src_req_cast) {
                bool u = deconstruct_qualified_type(&src_req_cast);
                enter_composite(&src_req_cast, &u, src->payload.lea.indices, false);
                return src_req_cast;
            }
            break;
        }
        default: break;
    }

    return NULL;
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

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
                Node* ndecl = global_var(r->dst_module, rewrite_nodes(r, global_variable.annotations), t, global_variable.name, get_builtin_as(b));
                register_processed(r, node, ndecl);
                // no 'init' for builtins, right ?
                assert(!global_variable.init);
                ctx->builtins[b] = ndecl;
                return ndecl;
            }
            break;
        }
        case Load_TAG: {
            const Type* req_cast = get_req_cast(ctx, node->payload.load.ptr);
            if (req_cast) {
                assert(is_data_type(req_cast));
                BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, node->payload.load.mem));
                const Node* r1 = first(bind_instruction(bb, recreate_node_identity(r, node)));
                const Node* r2 = first(gen_primop(bb, reinterpret_op, singleton(req_cast), singleton(r1)));
                return yield_values_and_wrap_in_block(bb, singleton(r2));
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* normalize_builtins(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.validate_builtin_types = true;
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .builtins = calloc(sizeof(Node*), BuiltinsCount)
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    free(ctx.builtins);
    return dst;
}
