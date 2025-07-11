#include "shady/pass.h"
#include "shady/ir/annotation.h"
#include "shady/ir/debug.h"
#include "shady/ir/function.h"
#include "shady/ir/mem.h"
#include "shady/ir/decl.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/composite.h"
#include "shady/dict.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
    Node2Node lifted_globals;
    Nodes extra_details;
    Nodes extra_params;
    Nodes extra_globals;
} Context;

static bool affected(const Node* n) {
    if (n->tag != GlobalVariable_TAG)
        return false;
    if (n->payload.global_variable.address_space == AsGlobal)
        return true;
    return shd_lookup_annotation(n, "AllocateInScratchMemory");
}

static const Node* add(const Node* a, const Node* b) {
    IrArena* arena = a->arena;
    return prim_op_helper(arena, add_op, mk_nodes(arena, a, b));
}

static const Node* mul(const Node* a, const Node* b) {
    IrArena* arena = a->arena;
    return prim_op_helper(arena, mul_op, mk_nodes(arena, a, b));
}

static OpRewriteResult* process(Context* ctx, SHADY_UNUSED NodeClass use, SHADY_UNUSED String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            if (!get_abstraction_body(node))
                break;
            Function payload = shd_rewrite_function_head_payload(r, node->payload.fun);
            Context fn_ctx = *ctx;
            shd_register_processed_list(&fn_ctx.rewriter, get_abstraction_params(node), payload.params);
            if (shd_lookup_annotation(node, "EntryPoint")) {
                payload.params = shd_concat_nodes(a, ctx->extra_params, payload.params);
            }
            Node* newfun = shd_function(r->dst_module, payload);
            shd_register_processed(r, node, newfun);
            shd_rewrite_annotations(r, node, newfun);
            fn_ctx.rewriter = shd_create_children_rewriter(r);
            fn_ctx.bb = shd_bld_begin(a, shd_get_abstraction_mem(newfun));
            if (shd_lookup_annotation(node, "EntryPoint")) {
                // copy the params
                for (size_t i = 0; i < ctx->extra_globals.count; i++) {
                    const Node* value = ctx->extra_params.nodes[i];
                    const Node* old_global = ctx->extra_details.nodes[i];
                    assert(affected(old_global));
                    const Node* scratch = shd_lookup_annotation(old_global, "AllocateInScratchMemory");
                    if (scratch) {
                        // we need to map to the correct stack...
                        const Node* global_thread_offset;
                        if (shd_is_rt_execution_model(shd_get_arena_config(ctx->rewriter.src_arena)->target.execution_model)) {
                            const Node* launch_size = shd_bld_builtin_load(r->dst_module, fn_ctx.bb, ShdBuiltinLaunchSizeKHR);
                            //const Node* launch_width = prim_op_helper(a, extract_op, mk_nodes(a, launch_size, shd_uint32_literal(a, 0)));
                            const Node* launch_height = shd_extract_literal(a, launch_size, 1);
                            const Node* launch_depth = shd_extract_literal(a, launch_size, 2);

                            const Node* launch_id = shd_bld_builtin_load(r->dst_module, fn_ctx.bb, ShdBuiltinLaunchIdKHR);
                            const Node* launch_x = shd_extract_literal(a, launch_id, 0);
                            const Node* launch_y = shd_extract_literal(a, launch_id, 1);
                            const Node* launch_z = shd_extract_literal(a, launch_id, 2);
                            global_thread_offset = add(launch_z, mul(launch_depth, add(launch_y, mul(launch_height, launch_x))));
                        } else {
                            const Node* total_workgroup_size = shd_uint32_literal(a, shd_get_arena_config(a)->specializations.workgroup_size[0] * shd_get_arena_config(a)->specializations.workgroup_size[1] * shd_get_arena_config(a)->specializations.workgroup_size[2]);
                            const Node* num_workgroups = shd_bld_builtin_load(r->dst_module, fn_ctx.bb, ShdBuiltinNumWorkgroups);
                            //const Node* num_workgroups_x = prim_op_helper(a, extract_op, mk_nodes(a, num_workgroups, shd_uint32_literal(a, 0)));
                            const Node* num_workgroups_y = shd_extract_literal(a, num_workgroups, 1);
                            const Node* num_workgroups_z = shd_extract_literal(a, num_workgroups, 2);

                            const Node* workgroup_id = shd_bld_builtin_load(r->dst_module, fn_ctx.bb, ShdBuiltinWorkgroupId);
                            const Node* workgroup_id_x = shd_extract_literal(a, workgroup_id, 0);
                            const Node* workgroup_id_y = shd_extract_literal(a, workgroup_id, 1);
                            const Node* workgroup_id_z = shd_extract_literal(a, workgroup_id, 2);
                            const Node* linear_workgroup_id = add(workgroup_id_z, mul(num_workgroups_z, add(workgroup_id_y, mul(num_workgroups_y, workgroup_id_x))));

                            const Node* subgroup_size = shd_bld_builtin_load(r->dst_module, fn_ctx.bb, ShdBuiltinSubgroupSize);
                            const Node* subgroup_id = shd_bld_builtin_load(r->dst_module, fn_ctx.bb, ShdBuiltinSubgroupId);

                            // const Node* thread_size = size_of_helper(a, shd_rewrite_op(r, NcType, "type", old_global->payload.global_variable.type));
                            // thread_size = shd_bld_convert_int_zero_extend(fn_ctx.bb, shd_get_unqualified_type(linear_workgroup_id->type), thread_size);
                            // const Node* thread_size = shd_get_annotation_value(scratch);
                            // TODO: make this more robust wrt builtin sizes
                            // TypeMemLayout layout = shd_get_mem_layout(a, old_global->payload.global_variable.type);
                            // const Node* thread_size = shd_uint32_literal(a, shd_bytes_to_words_static(a, layout.size_in_bytes));
                            global_thread_offset = add(mul(subgroup_id, subgroup_size), mul(linear_workgroup_id, total_workgroup_size));
                        }
                        value = ptr_array_element_offset_helper(a, value, global_thread_offset);
                        if (ctx->config->printf_trace.scratch_base_addr) {
                            const Node* ptr = bit_cast_helper(a, shd_uint64_type(a), value);
                            shd_bld_debug_printf(fn_ctx.bb, "GTO=%d ptr=%lx\n", mk_nodes(a, global_thread_offset, ptr));
                        }
                    }
                    shd_bld_store(fn_ctx.bb, ctx->extra_globals.nodes[i], value);
                }
            }
            Node* post_prelude = basic_block_helper(a, shd_empty(a));
            shd_set_debug_name(post_prelude, "post-prelude");
            shd_register_processed(&fn_ctx.rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(post_prelude));
            shd_set_abstraction_body(post_prelude, shd_rewrite_op(&fn_ctx.rewriter, NcTerminator, "body", get_abstraction_body(node)));
            shd_set_abstraction_body(newfun, shd_bld_finish(fn_ctx.bb, jump_helper(a, shd_bld_mem(fn_ctx.bb), post_prelude,
                                                                                   shd_empty(a))));
            shd_destroy_rewriter(&fn_ctx.rewriter);
            return shd_new_rewrite_result(r, newfun);
        }
        case GlobalVariable_TAG: {
            if (!affected(node))
                break;
            assert(ctx->bb && "this Global isn't appearing in an abstraction - we cannot replace it with a load!");
            const Node* ptr_addr = shd_node2node_find(ctx->lifted_globals, node);
            const Node* ptr = shd_bld_load(ctx->bb, ptr_addr);
            ptr = scope_cast_helper(a, shd_get_qualified_type_scope(node->type), ptr);
            OpRewriteResult* result = shd_new_rewrite_result_none(r);
            shd_rewrite_result_add_mask_rule(result, NcValue, ptr);
            return result;
        }
        default: break;
    }

    return shd_new_rewrite_result(r, shd_recreate_node(r, node));
}

static Rewriter* rewrite_globals_in_local_ctx(Rewriter* r, const Node* n) {
    if (affected(n))
        return r;
    return shd_default_rewriter_selector(r, n);
}

Module* shd_pass_globals_to_params(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process),
        .config = config,
        .lifted_globals = shd_new_node2node(),
    };
    ctx.rewriter.select_rewriter_fn = rewrite_globals_in_local_ctx;

    Nodes oglobals = shd_module_collect_reachable_globals(src);
    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* odecl = oglobals.nodes[i];
        if (!affected(odecl))
            continue;

        const Type* t = shd_get_unqualified_type(shd_rewrite_op(&ctx.rewriter, NcType, "type", odecl->type));
        String name = shd_get_node_name_unsafe(odecl);

        const Node* g = shd_global_var(dst, (GlobalVariable) {
            .address_space = AsPrivate,
            .type = t,
            .is_ref = true,
        });
        const Node* p = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.constants, t));
        if (name) {
            shd_set_debug_name(p, name);
            shd_set_debug_name(g, name);
        }

        shd_node2node_insert(ctx.lifted_globals, odecl, g);
        ctx.extra_details = shd_nodes_append(a, ctx.extra_details, odecl);
        ctx.extra_params = shd_nodes_append(a, ctx.extra_params, p);
        ctx.extra_globals = shd_nodes_append(a, ctx.extra_globals, g);
    }

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_node2node(ctx.lifted_globals);

    size_t j = 0;
    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* odecl = oglobals.nodes[i];
        if (!affected(odecl))
            continue;
        assert(j < ctx.extra_params.count);
        if (shd_lookup_annotation(odecl, "AllocateInScratchMemory")) {
            TypeMemLayout layout = shd_get_mem_layout(a, odecl->payload.global_variable.type);
            const Node* thread_size = shd_uint32_literal(a, layout.size_in_bytes);
            shd_add_annotation(ctx.extra_params.nodes[j], annotation_values(a, (AnnotationValues) {
                .name = "RuntimeProvideScratch",
                .values = mk_nodes(a, thread_size)
            }));
        } else if (odecl->payload.global_variable.init) {
            shd_add_annotation(ctx.extra_params.nodes[j], annotation_values(a, (AnnotationValues) {
                    .name = "RuntimeProvideMem",
                    .values = mk_nodes(a, shd_rewrite_op(&ctx.rewriter, NcValue, "init", odecl->payload.global_variable.init))
            }));
        } else {
            shd_error("TODO: implement just reserving scratch space for unintialized constants");
        }
        j++;
    }

    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
