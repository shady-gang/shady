#include "shady/pass.h"
#include "shady/ir/builtin.h"

#include "ir_private.h"

#include "util.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    Node** globals;
    bool is_entry_point;
} Context;

static void add_bounds_check(IrArena* a, BodyBuilder* bb, const Node* i, const Node* max) {
    Node* out_of_bounds_case = basic_block_helper(a, shd_empty(a));
    shd_set_abstraction_body(out_of_bounds_case, merge_break(a, (MergeBreak) {
        .args = shd_empty(a),
        .mem = shd_get_abstraction_mem(out_of_bounds_case)
    }));
    shd_bld_if(bb, shd_empty(a), prim_op_helper(a, gte_op, shd_empty(a), mk_nodes(a, i, max)), out_of_bounds_case, NULL);
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;

    switch (node->tag) {
        case BuiltinRef_TAG: {
            Builtin b = node->payload.builtin_ref.builtin;
            switch (b) {
                case BuiltinSubgroupId:
                case BuiltinWorkgroupId:
                case BuiltinGlobalInvocationId:
                case BuiltinLocalInvocationId: {
                    Node* global = global_variable_helper(m, shd_get_builtin_type(a, b), AsPrivate);
                    shd_module_add_export(m, shd_fmt_string_irarena(a, "builtin_%s", shd_get_builtin_name(b)), global);
                    return global;
                } case BuiltinNumWorkgroups: {
                    Node* global = global_variable_helper(m, shd_get_builtin_type(a, b), AsExternal);
                    shd_module_add_export(m, shd_fmt_string_irarena(a, "builtin_%s", shd_get_builtin_name(b)), global);
                    return global;
                } default:
                    break;
            }
            break;
        }
        case Function_TAG: {
            Context ctx2 = *ctx;
            ctx2.is_entry_point = false;
            const Node* epa = shd_lookup_annotation(node, "EntryPoint");
            if (epa && strcmp(shd_get_annotation_string_payload(epa), "Compute") == 0) {
                ctx2.is_entry_point = true;
                assert(node->payload.fun.return_types.count == 0 && "entry points do not return at this stage");

                Nodes wparams = shd_recreate_params(&ctx->rewriter, node->payload.fun.params);
                Node* wrapper = function_helper(m, wparams, shd_empty(a));
                shd_rewrite_annotations(r, node, wrapper);
                shd_register_processed(&ctx->rewriter, node, wrapper);

                // recreate the old entry point, but this time it's not the entry point anymore
                Nodes nparams = shd_recreate_params(&ctx->rewriter, node->payload.fun.params);
                Node* inner = function_helper(m, nparams, shd_empty(a));
                shd_set_debug_name(inner, shd_format_string_arena(a->arena, "%s_wrapped", shd_get_node_name_safe(node)));
                shd_add_annotation_named(inner, "Leaf");
                shd_register_processed_list(&ctx->rewriter, node->payload.fun.params, nparams);
                shd_register_processed(&ctx->rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(inner));
                shd_set_abstraction_body(inner, shd_recreate_node(&ctx->rewriter, node->payload.fun.body));

                BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(wrapper));
                const Node* num_workgroups_var = shd_rewrite_node(&ctx->rewriter, shd_get_or_create_builtin(ctx->rewriter.src_module, BuiltinNumWorkgroups));
                const Node* workgroup_num_vec3 = shd_bld_load(bb, num_workgroups_var);

                // prepare variables for iterating over workgroups
                String names[] = { "gx", "gy", "gz" };
                const Node* workgroup_id[3];
                const Node* num_workgroups[3];
                for (int dim = 0; dim < 3; dim++) {
                    workgroup_id[dim] = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, shd_uint32_type(a)));
                    shd_set_debug_name(workgroup_id[dim], names[dim]);
                    num_workgroups[dim] = shd_extract_helper(a, workgroup_num_vec3, shd_singleton(shd_uint32_literal(a, dim)));
                }

                // Prepare variables for iterating inside workgroups
                const Node* subgroup_id[3];
                uint32_t num_subgroups[3];
                const Node* num_subgroups_literals[3];
                assert(ctx->config->target.subgroup_size);
                assert(a->config.specializations.workgroup_size[0] && a->config.specializations.workgroup_size[1] && a->config.specializations.workgroup_size[2]);
                num_subgroups[0] = a->config.specializations.workgroup_size[0] / ctx->config->target.subgroup_size;
                num_subgroups[1] = a->config.specializations.workgroup_size[1];
                num_subgroups[2] = a->config.specializations.workgroup_size[2];
                String names2[] = { "sgx", "sgy", "sgz" };
                for (int dim = 0; dim < 3; dim++) {
                    subgroup_id[dim] = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, shd_uint32_type(a)));
                    shd_set_debug_name(subgroup_id[dim], names2[dim]);
                    num_subgroups_literals[dim] = shd_uint32_literal(a, num_subgroups[dim]);
                }

                Node* cases[6];
                BodyBuilder* builders[6];
                for (int scope = 0; scope < 2; scope++) {
                    const Node** params;
                    const Node** maxes;
                    if (scope == 1) {
                        params = subgroup_id;
                        maxes = num_subgroups_literals;
                    } else if (scope == 0) {
                        params = workgroup_id;
                        maxes = num_workgroups;
                    } else
                        assert(false);
                    for (int dim = 0; dim < 3; dim++) {
                        Node* loop_body = basic_block_helper(a, shd_singleton(params[dim]));
                        cases[scope * 3 + dim] = loop_body;
                        BodyBuilder* loop_bb = shd_bld_begin(a, shd_get_abstraction_mem(loop_body));
                        builders[scope * 3 + dim] = loop_bb;
                        add_bounds_check(a, loop_bb, params[dim], maxes[dim]);
                    }
                }

                // BodyBuilder* bb2 = begin_block_with_side_effects(a, bb_mem(builders[5]));
                BodyBuilder* bb2 = builders[5];
                // write the workgroup ID
                shd_bld_store(bb2, shd_rewrite_node(&ctx->rewriter, shd_get_or_create_builtin(ctx->rewriter.src_module, BuiltinWorkgroupId)), composite_helper(a, pack_type(a, (PackType) { .element_type = shd_uint32_type(a), .width = 3 }), mk_nodes(a, workgroup_id[0], workgroup_id[1], workgroup_id[2])));
                // write the local ID
                const Node* local_id[3];
                // local_id[0] = SUBGROUP_SIZE * subgroup_id[0] + subgroup_local_id
                local_id[0] = prim_op_helper(a, add_op, shd_empty(a), mk_nodes(a, prim_op_helper(a, mul_op, shd_empty(a), mk_nodes(a, shd_uint32_literal(a, ctx->config->target.subgroup_size), subgroup_id[0])), shd_bld_builtin_load(m, bb, BuiltinSubgroupLocalInvocationId)));
                local_id[1] = subgroup_id[1];
                local_id[2] = subgroup_id[2];
                shd_bld_store(bb2, shd_rewrite_node(&ctx->rewriter, shd_get_or_create_builtin(ctx->rewriter.src_module, BuiltinLocalInvocationId)), composite_helper(a, pack_type(a, (PackType) { .element_type = shd_uint32_type(a), .width = 3 }), mk_nodes(a, local_id[0], local_id[1], local_id[2])));
                // write the global ID
                const Node* global_id[3];
                for (int dim = 0; dim < 3; dim++)
                    global_id[dim] = prim_op_helper(a, add_op, shd_empty(a), mk_nodes(a, prim_op_helper(a, mul_op, shd_empty(a), mk_nodes(a, shd_uint32_literal(a, a->config.specializations.workgroup_size[dim]), workgroup_id[dim])), local_id[dim]));
                shd_bld_store(bb2, shd_rewrite_node(&ctx->rewriter, shd_get_or_create_builtin(ctx->rewriter.src_module, BuiltinGlobalInvocationId)), composite_helper(a, pack_type(a, (PackType) { .element_type = shd_uint32_type(a), .width = 3 }), mk_nodes(a, global_id[0], global_id[1], global_id[2])));
                // TODO: write the subgroup ID
                shd_bld_call(bb2, inner, wparams);

                // Wrap in 3 loops for iterating over subgroups, then again for workgroups
                for (unsigned scope = 1; scope < 2; scope--) {
                    const Node** params;
                    if (scope == 0) {
                        params = workgroup_id;
                    } else if (scope == 1) {
                        params = subgroup_id;
                    } else
                        assert(false);
                    for (unsigned dim = 2; dim < 3; dim--) {
                        size_t depth = scope * 3 + dim;
                        Node* loop_body = cases[depth];
                        BodyBuilder* body_bb = builders[depth];

                        shd_set_abstraction_body(loop_body, shd_bld_finish(body_bb, merge_continue(a, (MergeContinue) {
                            .args = shd_singleton(prim_op_helper(a, add_op, shd_empty(a), mk_nodes(a, params[dim], shd_uint32_literal(a, 1)))),
                            .mem = shd_bld_mem(body_bb)
                        })));
                        shd_bld_loop(depth > 0 ? builders[depth - 1] : bb, shd_empty(a), shd_singleton(shd_uint32_literal(a, 0)), loop_body);
                    }
                }

                shd_set_abstraction_body(wrapper, shd_bld_finish(bb, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_bld_mem(bb) })));
                return wrapper;
            }
            return shd_recreate_node(&ctx2.rewriter, node);
        }
        case Load_TAG: {
            Load payload = node->payload.load;
            const Node* ptr = payload.ptr;
            if (ptr == shd_get_or_create_builtin(ctx->rewriter.src_module, BuiltinSubgroupId)) {
                BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
                const Node* loaded = shd_bld_add_instruction(bb, shd_recreate_node(&ctx->rewriter, node));
                const Node* uniformized = prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(loaded));
                return shd_bld_to_instr_yield_values(bb, shd_singleton(uniformized));
            }
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_lower_workgroups(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .globals = calloc(sizeof(Node*), PRIMOPS_COUNT),
    };
    shd_rewrite_module(&ctx.rewriter);
    free(ctx.globals);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
