#include "passes.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include <stdlib.h>
#include <assert.h>

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
    Node** globals;
    bool is_entry_point;
} Context;

static const Node* get_global_var(Context* ctx, Op op) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    const Node* node = prim_op(ctx->rewriter.src_arena, (PrimOp) { .op = op });

    String name = primop_names[op];
    const Type* t = rewrite_node(&ctx->rewriter, node->type);
    deconstruct_qualified_type(&t);
    Node* decl = ctx->globals[node->payload.prim_op.op];
    AddressSpace as;

    switch (op) {
        case global_id_op:
        case workgroup_local_id_op:
        case workgroup_id_op:
            as = AsPrivatePhysical;
            break;
        case workgroup_num_op:
            as = AsExternal;
            break;
        default: break;
    }

    if (!decl) {
        decl = ctx->globals[op] = global_var(m, empty(a), t, name, as);
    }
    return decl;
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;

    switch (node->tag) {
        case PrimOp_TAG: {
            switch (node->payload.prim_op.op) {
                case subgroup_id_op:
                case workgroup_id_op:
                case workgroup_local_id_op:
                case workgroup_num_op:
                case workgroup_size_op:
                case global_id_op: {
                    const Node* ref = ref_decl_helper(a,  get_global_var(ctx, node->payload.prim_op.op));
                    return prim_op(a, (PrimOp) { .op = load_op, .operands = singleton(ref) });
                }
                default: break;
            }
            break;
        }
        case Function_TAG: {
            Context ctx2 = *ctx;
            ctx2.is_entry_point = false;
            if (lookup_annotation(node, "EntryPoint")) {
                ctx2.is_entry_point = true;
                assert(node->payload.fun.return_types.count == 0 && "entry points do not return at this stage");

                Nodes wannotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
                Nodes wparams = recreate_variables(&ctx->rewriter, node->payload.fun.params);
                Node* wrapper = function(m, wparams, get_abstraction_name(node), wannotations, empty(a));
                register_processed(&ctx->rewriter, node, wrapper);

                // recreate the old entry point, but this time it's not the entry point anymore
                Nodes nannotations = filter_out_annotation(a, wannotations, "EntryPoint");
                Nodes nparams = recreate_variables(&ctx->rewriter, node->payload.fun.params);
                Node* inner = function(m, nparams, format_string(a, "%s_wrapped", get_abstraction_name(node)), nannotations, empty(a));
                register_processed_list(&ctx->rewriter, node->payload.fun.params, nparams);
                inner->payload.fun.body = recreate_node_identity(&ctx->rewriter, node->payload.fun.body);

                BodyBuilder* bb = begin_body(a);
                const Node* workgroup_num_vec3 = gen_load(bb, ref_decl_helper(a, get_global_var(ctx, workgroup_num_op)));

                // prepare variables for iterating over workgroups
                String names[] = { "gx", "gy", "gz" };
                const Node* workgroup_id[3];
                const Node* num_workgroups[3];
                for (int dim = 0; dim < 3; dim++) {
                    workgroup_id[dim] = var(a, qualified_type_helper(uint32_type(a), false), names[dim]);
                    num_workgroups[dim] = gen_extract(bb, workgroup_num_vec3, singleton(uint32_literal(a, dim)));
                }

                // Prepare variables for iterating inside workgroups
                const Node* subgroup_id[3];
                uint32_t num_subgroups[3];
                const Node* num_subgroups_literals[3];
                assert(a->config.specializations.subgroup_size);
                assert(a->config.specializations.workgroup_size[0] && a->config.specializations.workgroup_size[1] && a->config.specializations.workgroup_size[2]);
                num_subgroups[0] = a->config.specializations.workgroup_size[0] / a->config.specializations.subgroup_size;
                num_subgroups[1] = a->config.specializations.workgroup_size[1];
                num_subgroups[2] = a->config.specializations.workgroup_size[2];
                String names2[] = { "sgx", "sgy", "sgz" };
                for (int dim = 0; dim < 3; dim++) {
                    subgroup_id[dim] = var(a, qualified_type_helper(uint32_type(a), false), names2[dim]);
                    num_subgroups_literals[dim] = uint32_literal(a, num_subgroups[dim]);
                }

                BodyBuilder* bb2 = begin_body(a);
                // write the workgroup ID
                gen_store(bb2, ref_decl_helper(a, get_global_var(ctx, workgroup_id_op)), composite(a, pack_type(a, (PackType) { .element_type = uint32_type(a), .width = 3 }), mk_nodes(a, workgroup_id[0], workgroup_id[1], workgroup_id[2])));
                // write the local ID
                const Node* local_id[3];
                // local_id[0] = SUBGROUP_SIZE * subgroup_id[0] + subgroup_local_id
                local_id[0] = gen_primop_e(bb2, add_op, empty(a), mk_nodes(a, gen_primop_e(bb2, mul_op, empty(a), mk_nodes(a, uint32_literal(a, a->config.specializations.subgroup_size), subgroup_id[0])), gen_primop_e(bb2, subgroup_local_id_op, empty(a), empty(a))));
                local_id[1] = subgroup_id[1];
                local_id[2] = subgroup_id[2];
                gen_store(bb2, ref_decl_helper(a, get_global_var(ctx, workgroup_local_id_op)), composite(a, pack_type(a, (PackType) { .element_type = uint32_type(a), .width = 3 }), mk_nodes(a, local_id[0], local_id[1], local_id[2])));
                // write the global ID
                const Node* global_id[3];
                for (int dim = 0; dim < 3; dim++)
                    global_id[dim] = gen_primop_e(bb2, add_op, empty(a), mk_nodes(a, gen_primop_e(bb2, mul_op, empty(a), mk_nodes(a, uint32_literal(a, a->config.specializations.workgroup_size[dim]), workgroup_id[dim])), local_id[dim]));
                gen_store(bb2, ref_decl_helper(a, get_global_var(ctx, global_id_op)), composite(a, pack_type(a, (PackType) { .element_type = uint32_type(a), .width = 3 }), mk_nodes(a, global_id[0], global_id[1], global_id[2])));
                // TODO: write the subgroup ID

                bind_instruction(bb2, call(a, (Call) { .callee = fn_addr_helper(a, inner), .args = wparams }));
                const Node* instr = yield_values_and_wrap_in_block(bb2, empty(a));

                // Wrap in 3 loops for iterating over subgroups, then again for workgroups
                for (int scope = 0; scope < 2; scope++) {
                    const Node** params;
                    const Node** maxes;
                    if (scope == 0) {
                        params = subgroup_id;
                        maxes = num_subgroups_literals;
                    } else if (scope == 1) {
                        params = workgroup_id;
                        maxes = num_workgroups;
                    } else
                        assert(false);
                    for (int dim = 0; dim < 3; dim++) {
                        BodyBuilder* body_bb = begin_body(a);
                        bind_instruction(body_bb, if_instr(a, (If) {
                                .yield_types = empty(a),
                                .condition = gen_primop_e(body_bb, gte_op, empty(a), mk_nodes(a, params[dim], maxes[dim])),
                                .if_true = lambda(a, empty(a), merge_break(a, (MergeBreak) {.args = empty(a)})),
                                .if_false = NULL
                        }));
                        bind_instruction(body_bb, instr);
                        const Node* loop = loop_instr(a, (Loop) {
                                .yield_types = empty(a),
                                .initial_args = singleton(uint32_literal(a, 0)),
                                .body = lambda(a, singleton(params[dim]), finish_body(body_bb, merge_continue(a, (MergeContinue) {.args = singleton(gen_primop_e(body_bb, add_op, empty(a), mk_nodes(a, params[dim], uint32_literal(a, 1))))})))
                        });
                        instr = loop;
                    }
                }

                bind_instruction(bb, instr);
                wrapper->payload.fun.body = finish_body(bb, fn_ret(a, (Return) { .fn = wrapper, .args = empty(a) }));
                return wrapper;
            }
            return recreate_node_identity(&ctx2.rewriter, node);
        }
        /*case Return_TAG: {
            if (ctx->is_entry_point) {

            }
        }*/
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_workgroups(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
        .globals = calloc(sizeof(Node*), PRIMOPS_COUNT),
    };
    rewrite_module(&ctx.rewriter);
    free(ctx.globals);
    destroy_rewriter(&ctx.rewriter);
}
