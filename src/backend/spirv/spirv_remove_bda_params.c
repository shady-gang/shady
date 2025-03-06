#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/decl.h"
#include "shady/ir/annotation.h"
#include "shady/ir/function.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static Nodes rewrite_args(Context* ctx, const Nodes old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    const TargetConfig* target = &shd_get_arena_config(a)->target;
    LARRAY(const Node*, arr, old.count);
    for (size_t i = 0; i < old.count; i++) {
        const Node* new = shd_rewrite_node(r, old.nodes[i]);
        const Type* t = old.nodes[i]->type;
        shd_deconstruct_qualified_type(&t);
        if (t->tag == PtrType_TAG && t->payload.ptr_type.address_space == AsGlobal) {
            new = bit_cast_helper(a, int_type_helper(a, target->memory.ptr_size, false), new);
        }
        arr[i] = new;
    }
    return shd_nodes(a, old.count, arr);
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    const TargetConfig* target = &shd_get_arena_config(a)->target;

    switch (node->tag) {
        case Function_TAG: {
            Function payload = node->payload.fun;
            payload = shd_rewrite_function_head_payload(r, payload);
            for (size_t i = 0; i < payload.params.count; i++) {
                const Type* t = payload.params.nodes[i]->type;
                ShdScope s = shd_deconstruct_qualified_type(&t);
                if (t->tag == PtrType_TAG && t->payload.ptr_type.address_space == AsGlobal) {
                    const Node* param = param_helper(a, qualified_type_helper(a, s, int_type_helper(a, target->memory.ptr_size, false)));
                    const Node* reinterpreted = bit_cast_helper(a, t, param);
                    payload.params = shd_change_node_at_index(a, payload.params, i, param);
                    shd_register_processed(r, get_abstraction_params(node).nodes[i], reinterpreted);
                } else {
                    shd_register_processed(r, get_abstraction_params(node).nodes[i], payload.params.nodes[i]);
                }
            }

            Node* new = shd_function(r->dst_module, payload);
            shd_register_processed(r, node, new);
            shd_rewrite_annotations(&ctx->rewriter, node, new);
            shd_recreate_node_body(r, node, new);
            return new;
        }
        case Call_TAG: {
            Call payload = node->payload.call;
            return call(a, (Call) {
                .mem = shd_rewrite_node(r, payload.mem),
                .callee = shd_rewrite_node(r, payload.callee),
                .args = rewrite_args(ctx, payload.args),
            });
        }
        case IndirectCall_TAG: {
            IndirectCall payload = node->payload.indirect_call;
            return indirect_call(a, (IndirectCall) {
                .mem = shd_rewrite_node(r, payload.mem),
                .callee = shd_rewrite_node(r, payload.callee),
                .args = rewrite_args(ctx, payload.args),
            });
        }
        case IndirectTailCall_TAG: {
            IndirectTailCall payload = node->payload.indirect_tail_call;
            return indirect_tail_call(a, (IndirectTailCall) {
                .mem = shd_rewrite_node(r, payload.mem),
                .callee = shd_rewrite_node(r, payload.callee),
                .args = rewrite_args(ctx, payload.args),
            });
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* shd_spvbe_pass_remove_bda_params(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
