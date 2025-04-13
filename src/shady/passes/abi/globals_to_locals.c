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
    AddressSpace src_as;
    AddressSpace dst_as;
} Global2LocalsPassConfig;

typedef struct {
    Rewriter rewriter;
    Global2LocalsPassConfig pass_config;

    bool real_entry_point;

    struct {
        Nodes old;
        Nodes new;
    } promoted_to_copy;
    struct {
        Nodes old;
        Nodes new;
    } promoted_to_alloca;
} Context;

static void store_init_values(Context* ctx, BodyBuilder* bb) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Nodes oglobals = shd_module_collect_reachable_globals(ctx->rewriter.src_module);

    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* oglobal = oglobals.nodes[i];
        if (oglobal->payload.global_variable.address_space != ctx->pass_config.src_as)
            continue;
        const Node* oinit = oglobal->payload.global_variable.init;
        if (!oinit)
            continue;
        const Node* new_ptr = shd_rewrite_node(r, oglobal);
        const Node* ninit = shd_rewrite_node(r, oinit);
        shd_bld_store(bb, new_ptr, ninit);
    }
}

static Nodes pre_call(Context* ctx, BodyBuilder* bb, Nodes args) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    for (size_t i = 0; i < ctx->promoted_to_copy.old.count; i++) {
        const Node* in_value = shd_bld_load(bb, ctx->promoted_to_copy.new.nodes[i]);
        args = shd_nodes_prepend(a, args, in_value);
    }
    for (size_t i = 0; i < ctx->promoted_to_alloca.old.count; i++) {
        const Node* ptr = ctx->promoted_to_alloca.new.nodes[i];
        args = shd_nodes_prepend(a, args, ptr);
    }
    return args;
}

static Nodes post_call(Context* ctx, BodyBuilder* bb, Nodes retvals) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    for (size_t i = 0; i < ctx->promoted_to_copy.old.count; i++) {
        const Node* opromoted = ctx->promoted_to_copy.old.nodes[i];
        const Node* out_value = retvals.nodes[ctx->promoted_to_copy.old.count - 1 - i];
        shd_bld_store(bb, ctx->promoted_to_copy.new.nodes[i], out_value);

        String debug_name = shd_get_node_name_unsafe(opromoted);
        if (debug_name)
            shd_set_debug_name(out_value, debug_name);
    }
    return shd_nodes(a, retvals.count - ctx->promoted_to_copy.old.count, &retvals.nodes[ctx->promoted_to_copy.old.count]);
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    bool physical = shd_get_arena_config(a)->target.memory.address_spaces[ctx->pass_config.dst_as].physical;

    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.rewriter = shd_create_children_rewriter(r);
            Function payload = shd_rewrite_function_head_payload(r, node->payload.fun);
            shd_register_processed_list(&fn_ctx.rewriter, get_abstraction_params(node), payload.params);
            const Node* param = NULL;

            // callable shaders aren't real entry point for our purposes, we can modify their interface
            fn_ctx.real_entry_point = shd_lookup_annotation(node, "EntryPoint") && shd_execution_model_from_entry_point(node) != EmCallable;

            /*if (real_entry_point) {
                param = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, ptr_type_helper(a, ctx->pass_config.dst_as, ctx->t, !physical)));
                //payload.params = shd_nodes_prepend(a, payload.params, param);
                //fn_ctx.param = param;
            }*/
            Nodes copy_in = shd_empty(a);
            if(!fn_ctx.real_entry_point) {
                for (size_t i = 0; i < ctx->promoted_to_copy.old.count; i++) {
                    const Node* opromoted = ctx->promoted_to_copy.old.nodes[i];
                    const Type* t = shd_rewrite_node(r, opromoted->payload.global_variable.type);
                    t = qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, t);
                    param = param_helper(a, t);
                    payload.params = shd_nodes_prepend(a, payload.params, param);
                    copy_in = shd_nodes_append(a, copy_in, param);
                    payload.return_types = shd_nodes_prepend(a, payload.return_types, t);
                    String debug_name = shd_get_node_name_unsafe(opromoted);
                    if (debug_name)
                        shd_set_debug_name(param, debug_name);
                }
                for (size_t i = 0; i < ctx->promoted_to_alloca.old.count; i++) {
                    const Node* opromoted = ctx->promoted_to_alloca.old.nodes[i];
                    const Type* t = shd_rewrite_node(r, opromoted->payload.global_variable.type);
                    t = ptr_type_helper(a, ctx->pass_config.dst_as, t, !physical);
                    t = qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, t);
                    param = param_helper(a, t);
                    payload.params = shd_nodes_prepend(a, payload.params, param);
                    shd_register_processed(&fn_ctx.rewriter, opromoted, param);
                    fn_ctx.promoted_to_alloca.new = shd_nodes_append(a, fn_ctx.promoted_to_alloca.new, param);
                    String debug_name = shd_get_node_name_unsafe(opromoted);
                    if (debug_name)
                        shd_set_debug_name(param, debug_name);
                }
            }

            Node* new = shd_function(r->dst_module, payload);
            shd_register_processed(r, node, new);
            shd_rewrite_annotations(&ctx->rewriter, node, new);
            if (get_abstraction_body(node)) {
                const Node* mem0 = shd_get_abstraction_mem(new);
                BodyBuilder* bb = shd_bld_begin(a, mem0);
                //bool physical = shd_get_arena_config(a)->target.memory.address_spaces[ctx->pass_config.dst_as].physical;
                //const Node* alloc;
                //if (physical)
                //    alloc = shd_bld_add_instruction(bb, stack_alloc_helper(a, mem0, ctx->t));
                //else
                //    alloc = shd_bld_add_instruction(bb, local_alloc_helper(a, mem0, ctx->t));
                //fn_ctx.param = alloc;

                // inout params are also allocas inside the fn, but they are not allowed to leak
                for (size_t i = 0; i < ctx->promoted_to_copy.old.count; i++) {
                    const Node* opromoted = ctx->promoted_to_copy.old.nodes[i];
                    const Type* t = shd_rewrite_node(r, opromoted->payload.global_variable.type);
                    const Node* alloca = physical ? shd_bld_stack_alloc(bb, t) : shd_bld_local_alloc(bb, t);
                    if (copy_in.count > 0)
                        shd_bld_store(bb, alloca, copy_in.nodes[i]);
                    shd_register_processed(&fn_ctx.rewriter, opromoted, alloca);
                    fn_ctx.promoted_to_copy.new = shd_nodes_append(a, fn_ctx.promoted_to_copy.new, alloca);
                    String debug_name = shd_get_node_name_unsafe(opromoted);
                    if (debug_name)
                        shd_set_debug_name(alloca, debug_name);
                }
                if (fn_ctx.real_entry_point) {
                    for (size_t i = 0; i < ctx->promoted_to_alloca.old.count; i++) {
                        const Node* opromoted = ctx->promoted_to_alloca.old.nodes[i];
                        const Type* t = shd_rewrite_node(r, opromoted->payload.global_variable.type);
                        const Node* alloca = physical ? shd_bld_stack_alloc(bb, t) : shd_bld_local_alloc(bb, t);
                        if (copy_in.count > 0)
                            shd_bld_store(bb, alloca, copy_in.nodes[i]);
                        shd_register_processed(&fn_ctx.rewriter, opromoted, alloca);
                        fn_ctx.promoted_to_alloca.new = shd_nodes_append(a, fn_ctx.promoted_to_alloca.new, alloca);
                        String debug_name = shd_get_node_name_unsafe(opromoted);
                        if (debug_name)
                            shd_set_debug_name(alloca, debug_name);
                    }
                }

                if (fn_ctx.real_entry_point)
                   store_init_values(&fn_ctx, bb);

                shd_register_processed(&ctx->rewriter, shd_get_abstraction_mem(node), shd_bld_mem(bb));
                shd_set_abstraction_body(new, shd_bld_finish(bb, shd_rewrite_node(&fn_ctx.rewriter, get_abstraction_body(node))));
            }
            shd_destroy_rewriter(&fn_ctx.rewriter);
            return new;
        }
        case Return_TAG: {
            Return payload = node->payload.fn_ret;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            payload.args = shd_rewrite_nodes(r, payload.args);
            if (!ctx->real_entry_point) {
                for (size_t i = 0; i < ctx->promoted_to_copy.old.count; i++) {
                    const Node* return_value = shd_bld_load(bb, ctx->promoted_to_copy.new.nodes[i]);
                    payload.args = shd_nodes_prepend(a, payload.args, return_value);
                }
            }
            return shd_bld_return(bb, payload.args);
        }
        case Call_TAG: {
            Call payload = node->payload.call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            payload.callee = shd_rewrite_node(r, payload.callee);
            payload.args = shd_rewrite_nodes(r, payload.args);
            payload.args = pre_call(ctx, bb, payload.args);
            Nodes results = shd_bld_call(bb, payload.callee, payload.args);
            return shd_bld_to_instr_yield_values(bb, post_call(ctx, bb, results));
        }
        case IndirectCall_TAG: {
            IndirectCall payload = node->payload.indirect_call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            payload.callee = shd_rewrite_node(r, payload.callee);
            payload.args = shd_rewrite_nodes(r, payload.args);
            payload.args = pre_call(ctx, bb, payload.args);
            Nodes results = shd_bld_indirect_call(bb, payload.callee, payload.args);
            return shd_bld_to_instr_yield_values(bb, post_call(ctx, bb, results));
        }
        case IndirectTailCall_TAG: {
            IndirectTailCall payload = node->payload.indirect_tail_call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            payload.callee = shd_rewrite_node(r, payload.callee);
            payload.args = shd_rewrite_nodes(r, payload.args);
            payload.args = pre_call(ctx, bb, payload.args);
            return shd_bld_indirect_tail_call(bb, payload.callee, payload.args);
        }
        case PtrType_TAG: {
            PtrType payload = node->payload.ptr_type;
            if (payload.address_space == ctx->pass_config.src_as) {
                payload.address_space = ctx->pass_config.dst_as;
                payload.pointed_type = shd_rewrite_node(r, payload.pointed_type);
                return ptr_type(a, payload);
            }
            break;
        }
        case FnType_TAG: {
            FnType payload = shd_recreate_node(r, node)->payload.fn_type;

            for (size_t i = 0; i < ctx->promoted_to_copy.old.count; i++) {
                const Type* t = shd_rewrite_node(r, ctx->promoted_to_copy.old.nodes[i]->type);
                assert(t->tag == QualifiedType_TAG);
                payload.param_types = shd_nodes_prepend(a, payload.param_types, t);
                payload.return_types = shd_nodes_prepend(a, payload.return_types, t);
            }
            for (size_t i = 0; i < ctx->promoted_to_alloca.old.count; i++) {
                const Node* opromoted = ctx->promoted_to_alloca.old.nodes[i];
                const Type* t = shd_rewrite_node(r, opromoted->type);
                t = shd_get_unqualified_type(t);
                t = ptr_type_helper(a, ctx->pass_config.dst_as, t, !physical);
                payload.param_types = shd_nodes_prepend(a, payload.param_types, t);
            }

            return fn_type(a, payload);
        }
        case GlobalVariable_TAG: {
            GlobalVariable payload = node->payload.global_variable;
            if (payload.address_space == ctx->pass_config.src_as) {
                assert(false);
                //const Node* index = shd_node2node_find(ctx->backing, node);
                //assert(index);
                //return ptr_composite_element_helper(a, ctx->param, index);
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

static Rewriter* rewrite_globals_in_local_ctx(Context* ctx, const Node* n) {
    if (n->tag == GlobalVariable_TAG && n->payload.global_variable.address_space == ctx->pass_config.src_as)
        return &ctx->rewriter;
    return shd_default_rewriter_selector(&ctx->rewriter, n);
}

Module* shd_pass_globals_to_locals(SHADY_UNUSED const CompilerConfig* config, const Global2LocalsPassConfig* pass_config, Module* src) {
    IrArena* oa = shd_module_get_arena(src);
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .pass_config = *pass_config,
    };
    ctx.rewriter.select_rewriter_fn = (SelectRewriterFn*) rewrite_globals_in_local_ctx;

    Nodes oglobals = shd_module_collect_reachable_globals(src);
    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* oglobal = oglobals.nodes[i];
        if (oglobal->payload.global_variable.address_space != ctx.pass_config.src_as)
            continue;
        bool promote_to_ref = !shd_lookup_annotation(oglobal, "Inout") && !config->use_rt_pipelines_for_calls;
        if (promote_to_ref)
            ctx.promoted_to_alloca.old = shd_nodes_append(oa, ctx.promoted_to_alloca.old, oglobal);
        else
            ctx.promoted_to_copy.old = shd_nodes_append(oa, ctx.promoted_to_copy.old, oglobal);
        //members[count] = shd_rewrite_node(&ctx.rewriter, oglobal->payload.global_variable.type);
        //members_names[count] = shd_get_node_name_safe(oglobal);
        //const Node* index = shd_uint32_literal(a, count);
        //shd_node2node_insert(ctx.backing, oglobal, index);
        //count++;
    }

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
