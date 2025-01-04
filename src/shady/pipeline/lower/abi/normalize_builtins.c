#include "shady/pass.h"
#include "shady/ir/builtin.h"

#include "ir_private.h"

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
            const Node* ba = shd_lookup_annotation_list(global_variable.annotations, "Builtin");
            if (ba) {
                Builtin b = shd_get_builtin_by_name(shd_get_annotation_string_payload(ba));
                assert(b != BuiltinsCount);
                const Type* expected_t = shd_get_builtin_type(a, b);
                const Type* actual_t = shd_rewrite_node(&ctx->rewriter, src)->payload.global_variable.type;
                if (expected_t != actual_t) {
                    shd_log_fmt(INFO, "normalize_builtins: found builtin decl '%s' not matching expected type: '", global_variable.name);
                    shd_log_node(INFO, expected_t);
                    shd_log_fmt(INFO, "', got '");
                    shd_log_node(INFO, actual_t);
                    shd_log_fmt(INFO, "'.");
                    return actual_t;
                }
            }
            break;
        }
        case PtrCompositeElement_TAG: {
            const Type* src_req_cast = get_req_cast(ctx, src->payload.ptr_composite_element.ptr);
            if (src_req_cast) {
                bool u = shd_deconstruct_qualified_type(&src_req_cast);
                shd_enter_composite_type(&src_req_cast, &u, src->payload.ptr_composite_element.index, false);
                return src_req_cast;
            }
            break;
        }
        default: break;
    }

    return NULL;
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case GlobalVariable_TAG: {
            GlobalVariable global_variable = node->payload.global_variable;
            const Node* ba = shd_lookup_annotation_list(global_variable.annotations, "Builtin");
            if (ba) {
                Builtin b = shd_get_builtin_by_name(shd_get_annotation_string_payload(ba));
                assert(b != BuiltinsCount);
                if (ctx->builtins[b])
                    return ctx->builtins[b];
                const Type* t = shd_get_builtin_type(a, b);
                Node* ndecl = global_variable_helper(r->dst_module, shd_rewrite_nodes(r, global_variable.annotations), t, global_variable.name,
                                         shd_get_builtin_address_space(b), global_variable.is_ref);
                shd_register_processed(r, node, ndecl);
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
                assert(shd_is_data_type(req_cast));
                BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, node->payload.load.mem));
                const Node* r1 = shd_bld_add_instruction(bb, shd_recreate_node(r, node));
                const Node* r2 = prim_op_helper(a, reinterpret_op, shd_singleton(req_cast), shd_singleton(r1));
                return shd_bld_to_instr_yield_values(bb, shd_singleton(r2));
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_normalize_builtins(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.validate_builtin_types = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .builtins = calloc(sizeof(Node*), BuiltinsCount)
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    free(ctx.builtins);
    return dst;
}
