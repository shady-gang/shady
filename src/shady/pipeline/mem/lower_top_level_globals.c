#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/function.h"
#include "shady/ir/builtin.h"
#include "shady/ir/annotation.h"
#include "shady/ir/debug.h"
#include "shady/ir/decl.h"
#include "shady/dict.h"

#include "log.h"
#include "portability.h"
#include "shady/ir/mem.h"

typedef struct {
    Rewriter rewriter;
    AddressSpace as;
    AddressSpace new_as;
    const Type* t;

    const Node* param;
    Node2Node backing;
} Context;

static void store_init_values(Context* ctx, BodyBuilder* bb) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Nodes oglobals = shd_module_collect_reachable_globals(ctx->rewriter.src_module);

    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* oglobal = oglobals.nodes[i];
        if (oglobal->payload.global_variable.address_space != ctx->as)
            continue;
        const Node* oinit = oglobal->payload.global_variable.init;
        if (!oinit)
            continue;
        const Node* new_ptr = shd_rewrite_node(r, oglobal);
        const Node* ninit = shd_rewrite_node(r, oinit);
        shd_bld_store(bb, new_ptr, ninit);
    }
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.rewriter = shd_create_children_rewriter(r);
            Function payload = shd_rewrite_function_head_payload(r, node->payload.fun);
            shd_register_processed_list(&fn_ctx.rewriter, get_abstraction_params(node), payload.params);
            const Node* param = NULL;
            if (!shd_lookup_annotation(node, "EntryPoint")) {
                param = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, ptr_type_helper(a, ctx->new_as, ctx->t, true)));
                payload.params = shd_nodes_prepend(a, payload.params, param);
                fn_ctx.param = param;
            }
            Node* new = shd_function(r->dst_module, payload);
            shd_register_processed(r, node, new);
            shd_rewrite_annotations(&ctx->rewriter, node, new);
            if (get_abstraction_body(node)) {
                if (!param) {
                    const Node* mem0 = shd_get_abstraction_mem(new);
                    BodyBuilder* bb = shd_bld_begin(a, mem0);
                    const Node* alloc = shd_bld_add_instruction(bb, local_alloc_helper(a, mem0, ctx->t));
                    fn_ctx.param = alloc;
                    store_init_values(&fn_ctx, bb);
                    shd_register_processed(&ctx->rewriter, shd_get_abstraction_mem(node), shd_bld_mem(bb));
                    shd_set_abstraction_body(new, shd_bld_finish(bb, shd_rewrite_node(&fn_ctx.rewriter, get_abstraction_body(node))));
                } else
                    shd_set_abstraction_body(new, shd_rewrite_node(&fn_ctx.rewriter, get_abstraction_body(node)));
            }
            shd_destroy_rewriter(&fn_ctx.rewriter);
            return new;
        }
        case Call_TAG: {
            assert(ctx->param);
            Call payload = node->payload.call;
            payload.mem = shd_rewrite_node(r, payload.mem);
            payload.callee = shd_rewrite_node(r, payload.callee);
            payload.args = shd_rewrite_nodes(r, payload.args);
            payload.args = shd_nodes_prepend(a, payload.args, ctx->param);
            return call(a, payload);
        }
        case IndirectCall_TAG: {
            assert(ctx->param);
            IndirectCall payload = node->payload.indirect_call;
            payload.mem = shd_rewrite_node(r, payload.mem);
            payload.callee = shd_rewrite_node(r, payload.callee);
            payload.args = shd_rewrite_nodes(r, payload.args);
            payload.args = shd_nodes_prepend(a, payload.args, ctx->param);
            return indirect_call(a, payload);
        }
        case IndirectTailCall_TAG: {
            assert(ctx->param);
            IndirectTailCall payload = node->payload.indirect_tail_call;
            payload.mem = shd_rewrite_node(r, payload.mem);
            payload.callee = shd_rewrite_node(r, payload.callee);
            payload.args = shd_rewrite_nodes(r, payload.args);
            payload.args = shd_nodes_prepend(a, payload.args, ctx->param);
            return indirect_tail_call(a, payload);
        }
        case PtrType_TAG: {
            PtrType payload = node->payload.ptr_type;
            if (payload.address_space == ctx->as) {
                payload.address_space = ctx->new_as;
                return ptr_type(a, payload);
            }
            break;
        }
        case FnType_TAG: {
            FnType payload = shd_recreate_node(r, node)->payload.fn_type;
            payload.param_types = shd_nodes_prepend(a, payload.param_types, ptr_type_helper(a, ctx->new_as, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, ctx->t), true));
            return fn_type(a, payload);
        }
        case GlobalVariable_TAG: {
            GlobalVariable payload = node->payload.global_variable;
            if (payload.address_space == ctx->as) {
                const Node* index = shd_node2node_find(ctx->backing, node);
                assert(index);
                return ptr_composite_element_helper(a, ctx->param, index);
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

static Rewriter* rewrite_globals_in_local_ctx(Context* ctx, const Node* n) {
    if (n->tag == GlobalVariable_TAG && n->payload.global_variable.address_space == ctx->as)
        return &ctx->rewriter;
    return shd_default_rewriter_selector(&ctx->rewriter, n);
}

Module* shd_pass_lower_top_level_globals(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .as = AsPrivate,
        .new_as = AsFunction,
        .backing = shd_new_node2node(),
    };
    ctx.rewriter.select_rewriter_fn = (SelectRewriterFn*) rewrite_globals_in_local_ctx;

    Nodes oglobals = shd_module_collect_reachable_globals(src);
    LARRAY(const Type*, members, oglobals.count);
    size_t count = 0;
    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* oglobal = oglobals.nodes[i];
        if (oglobal->payload.global_variable.address_space != ctx.as)
            continue;
        members[count] = shd_rewrite_node(&ctx.rewriter, oglobal->payload.global_variable.type);
        const Node* index = shd_uint32_literal(a, count);
        shd_node2node_insert(ctx.backing, oglobal, index);
        count++;
    }

    ctx.t = record_type_helper(a, shd_nodes(a, count, members), shd_strings(a, 0, NULL), NotSpecial);

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_node2node(ctx.backing);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
