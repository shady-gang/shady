#include <shady/ir/composite.h>

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

static Nodes rewrite_results(Context* ctx, Nodes result_types, Nodes results) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    LARRAY(const Node*, arr, result_types.count);
    for (size_t i = 0; i < result_types.count; i++) {
        const Node* new = results.nodes[i];
        const Type* t = result_types.nodes[i];
        shd_deconstruct_qualified_type(&t);
        if (t->tag == PtrType_TAG && t->payload.ptr_type.address_space == AsGlobal) {
            new = bit_cast_helper(a, t, new);
        }
        arr[i] = new;
    }
    return shd_nodes(a, result_types.count, arr);
}

static const Node* rewrite_result(Context* ctx, const Node* oldcall, const Node* result) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Nodes result_types = shd_rewrite_nodes(r, shd_unwrap_multiple_yield_types(oldcall->arena, oldcall->type));
    return shd_maybe_tuple_helper(a, rewrite_results(ctx, result_types, shd_deconstruct_composite(a, result, result_types.count)));
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
            for (size_t i = 0; i < payload.return_types.count; i++) {
                const Type* t = payload.return_types.nodes[i];
                ShdScope s = shd_deconstruct_qualified_type(&t);
                const Type* npt = NULL;
                if (t->tag == PtrType_TAG && t->payload.ptr_type.address_space == AsGlobal) {
                    npt = qualified_type_helper(a, s, int_type_helper(a, target->memory.ptr_size, false));
                    payload.return_types = shd_change_node_at_index(a, payload.return_types, i, npt);
                }
            }

            Node* new = shd_function(r->dst_module, payload);
            shd_register_processed(r, node, new);
            shd_rewrite_annotations(&ctx->rewriter, node, new);
            shd_recreate_node_body(r, node, new);
            return new;
        }
        case Return_TAG: {
            Return payload = node->payload.fn_ret;
            payload.mem = shd_rewrite_node(r, payload.mem);
            payload.args = rewrite_args(ctx, payload.args);
            return fn_ret(a, payload);
        }
        case FnType_TAG: {
            FnType payload = node->payload.fn_type;
            for (size_t i = 0; i < payload.param_types.count; i++) {
                const Type* t = payload.param_types.nodes[i];
                ShdScope s = shd_deconstruct_qualified_type(&t);
                const Type* npt = NULL;
                if (t->tag == PtrType_TAG && t->payload.ptr_type.address_space == AsGlobal) {
                    npt = qualified_type_helper(a, s, int_type_helper(a, target->memory.ptr_size, false));
                } else {
                    npt = shd_rewrite_node(r, payload.param_types.nodes[i]);
                }
                payload.param_types = shd_change_node_at_index(a, payload.param_types, i, npt);
            }
            for (size_t i = 0; i < payload.return_types.count; i++) {
                const Type* t = payload.return_types.nodes[i];
                ShdScope s = shd_deconstruct_qualified_type(&t);
                const Type* npt = NULL;
                if (t->tag == PtrType_TAG && t->payload.ptr_type.address_space == AsGlobal) {
                    npt = qualified_type_helper(a, s, int_type_helper(a, target->memory.ptr_size, false));
                } else {
                    npt = shd_rewrite_node(r, payload.return_types.nodes[i]);
                }
                payload.return_types = shd_change_node_at_index(a, payload.return_types, i, npt);
            }
            return fn_type(a, payload);
        }
        case Call_TAG: {
            Call payload = node->payload.call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes results = shd_bld_call(bb, shd_rewrite_node(r, payload.callee), rewrite_args(ctx, payload.args));
            Nodes expected_result_types = shd_rewrite_nodes(r, shd_unwrap_multiple_yield_types(node->arena, node->type));
            results = rewrite_results(ctx, expected_result_types, results);
            return shd_bld_to_instr_yield_values(bb, results);
        }
        case IndirectCall_TAG: {
            IndirectCall payload = node->payload.indirect_call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes results = shd_bld_indirect_call(bb, shd_rewrite_node(r, payload.callee), rewrite_args(ctx, payload.args));
            Nodes expected_result_types = shd_rewrite_nodes(r, shd_unwrap_multiple_yield_types(node->arena, node->type));
            results = rewrite_results(ctx, expected_result_types, results);
            return shd_bld_to_instr_yield_values(bb, results);
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

Module* shd_spvbe_pass_remove_bda_params(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED void* unused, Module* src) {
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
