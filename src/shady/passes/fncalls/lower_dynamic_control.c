#include "join_point_ops.h"
#include "ir_private.h"

#include "shady/pass.h"
#include "shady/ir/stack.h"
#include "shady/ir/cast.h"
#include "shady/ir/builtin.h"
#include "shady/analysis/uses.h"

#include "analysis/cfg.h"
#include "analysis/leak.h"

#include "log.h"
#include "portability.h"
#include "util.h"
#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>
#include "shady/ir/memory_layout.h"

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;
    bool disable_lowering;

    CFG* cfg;
    const UsesMap* uses;
} Context;

static const Node* process(Context* ctx, const Node* old);

/// Turn a function into a top-level entry point, calling into the top dispatch function.
static const Node* lift_entry_point(Context* ctx, const Node* old, const Node* fun) {
    assert(old->tag == Function_TAG && fun->tag == Function_TAG);
    Context ctx2 = *ctx;
    Rewriter* r = &ctx2.rewriter;
    IrArena* a = ctx->rewriter.dst_arena;
    // For the lifted entry point, we keep _all_ annotations
    Nodes rewritten_params = shd_recreate_params(&ctx2.rewriter, old->payload.fun.params);
    Node* new_entry_pt = function_helper(ctx2.rewriter.dst_module, rewritten_params, shd_nodes(a, 0, NULL));
    shd_rewrite_annotations(r, old, new_entry_pt);

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new_entry_pt));

    shd_bld_call(bb, shd_find_or_process_decl(&ctx->rewriter, "builtin_init_scheduler"), shd_empty(a));

    // shove the arguments on the stack
    for (size_t i = rewritten_params.count - 1; i < rewritten_params.count; i--) {
        shd_bld_stack_push_value(bb, rewritten_params.nodes[i]);
    }

    // Initialise next_fn/next_mask to the entry function
    const Node* entry_point_addr = fn_addr_helper(a, shd_rewrite_node(r, old));
    entry_point_addr = shd_bld_bitcast(bb, int_type_helper(a, shd_get_arena_config(a)->target.memory.fn_ptr_size, false), entry_point_addr);
    shd_bld_call(bb, shd_find_or_process_decl(&ctx->rewriter, "builtin_fork"), shd_singleton(entry_point_addr));
    shd_bld_add_instruction(bb, ext_instr(a, (ExtInstr) {
        .result_t = unit_type(a),
        .mem = shd_bld_mem(bb),
        .set = "shady.internal",
        .opcode = ShadyOpDispatcherEnterFn,
        .operands = shd_empty(a),
    }));
    shd_set_abstraction_body(new_entry_pt, shd_bld_return(bb, shd_empty(a)));
    return new_entry_pt;
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (old->tag) {
        case Function_TAG: {
            Context ctx2 = *ctx;
            ctx2.cfg = build_fn_cfg(old);
            ctx2.uses = shd_new_uses_map_fn(old, (NcFunction | NcType));
            ctx = &ctx2;

            const Node* entry_point_annotation = shd_lookup_annotation(old, "EntryPoint");
            String exported_name = shd_get_exported_name(old);

            // Leave leaf-calls alone :)
            ctx2.disable_lowering = shd_lookup_annotation(old, "Leaf") || !old->payload.fun.body;
            if (ctx2.disable_lowering) {
                Node* fun = shd_recreate_node_head(&ctx2.rewriter, old);
                if (old->payload.fun.body) {
                    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));
                    shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(old), shd_bld_mem(bb));
                    shd_set_abstraction_body(fun, shd_bld_finish(bb, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old))));
                }

                shd_destroy_uses_map(ctx2.uses);
                shd_destroy_cfg(ctx2.cfg);
                return fun;
            }

            assert(ctx->config->dynamic_scheduling && "Dynamic scheduling is disabled, but we encountered a non-leaf function");
            shd_remove_annotation_by_name(old, "EntryPoint");
            shd_remove_annotation_by_name(old, "Exported");

            Node* fun = shd_recreate_node_head(r, old);
            shd_set_debug_name(fun, shd_format_string_arena(a->arena, "%s_indirect", shd_get_node_name_safe(old)));
            shd_rewrite_annotations(r, old, fun);

            shd_register_processed(&ctx->rewriter, old, fun);

            if (entry_point_annotation) {
                assert(exported_name);
                const Node* new_entry_pt = lift_entry_point(ctx, old, fun);
                shd_add_annotation(new_entry_pt, shd_rewrite_node(r, entry_point_annotation));
                shd_add_annotation_named(new_entry_pt, "Leaf");
                shd_module_add_export(ctx->rewriter.dst_module, exported_name, new_entry_pt);
            }
            BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));
            shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(old), shd_bld_mem(bb));
            shd_set_abstraction_body(fun, shd_bld_finish(bb, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old))));
            shd_destroy_uses_map(ctx2.uses);
            shd_destroy_cfg(ctx2.cfg);
            return fun;
        }
        case JoinPointType_TAG: return shd_find_or_process_decl(&ctx->rewriter, "JoinPoint");
        case ExtInstr_TAG: {
            ExtInstr payload = old->payload.ext_instr;
            if (strcmp(payload.set, "shady.internal") == 0) {
                String callee_name = NULL;
                Nodes args = shd_rewrite_nodes(r, payload.operands);
                switch ((ShadyJoinPointOpcodes ) payload.opcode) {
                    case ShadyOpDefaultJoinPoint:
                        callee_name = "builtin_entry_join_point";
                        break;
                    case ShadyOpCreateJoinPoint:
                        callee_name = "builtin_create_control_point";
                        const Node* dst = bit_cast_helper(a, int_type_helper(a, shd_get_arena_config(a)->target.memory.fn_ptr_size, false), args.nodes[0]);
                        args = shd_change_node_at_index(a, args, 0, dst);
                        break;
                    default: goto rebuild;
                }
                return call(a, (Call) {
                    .mem = shd_rewrite_node(r, payload.mem),
                    .callee = shd_find_or_process_decl(r, callee_name),
                    .args = args,
                });
            }
            break;
        }
        case IndirectTailCall_TAG: {
            IndirectTailCall payload = old->payload.indirect_tail_call;
            if (shd_get_qualified_type_scope(payload.callee->type) <= shd_get_arena_config(a)->target.scopes.gang) {
                const Node* mem0 = shd_get_original_mem(payload.mem);
                assert(mem0->tag == AbsMem_TAG);
                // checking that the payload is uniform is not sufficient: we could be branching uniformingly in non-uniform control flow
                // this check is a bit conservative and heavy-handed but should be correct
                if (mem0->payload.abs_mem.abs->tag == Function_TAG)
                    break;
            }

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_stack_push_values(bb, shd_rewrite_nodes(&ctx->rewriter, payload.args));
            const Node* target = shd_rewrite_node(&ctx->rewriter, payload.callee);
            target = shd_bld_bitcast(bb, int_type_helper(a, shd_get_arena_config(a)->target.memory.fn_ptr_size, false), target);

            shd_bld_call(bb, shd_find_or_process_decl(&ctx->rewriter, "builtin_fork"), shd_singleton(target));
            return shd_bld_finish(bb, ext_terminator(a, (ExtTerminator) {
                .mem = shd_bld_mem(bb),
                .set = "shady.internal",
                .opcode = ShadyOpDispatcherContinue,
                .operands = shd_empty(a),
            }));
        }
        case Join_TAG: {
            Join payload = old->payload.join;

            const Node* jp = shd_rewrite_node(&ctx->rewriter, old->payload.join.join_point);
            if (shd_get_unqualified_type(jp->type)->tag == JoinPointType_TAG)
                break;

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_stack_push_values(bb, shd_rewrite_nodes(&ctx->rewriter, old->payload.join.args));
            const Node* jp_payload = prim_op_helper(a, extract_op, mk_nodes(a, jp, shd_int32_literal(a, 2)));
            shd_bld_stack_push_value(bb, jp_payload);
            const Node* dst = prim_op_helper(a, extract_op, mk_nodes(a, jp, shd_int32_literal(a, 1)));
            const Node* tree_node = prim_op_helper(a, extract_op, mk_nodes(a, jp, shd_int32_literal(a, 0)));

            shd_bld_call(bb, shd_find_or_process_decl(&ctx->rewriter, "builtin_join"), mk_nodes(a, dst, tree_node));
            return shd_bld_finish(bb, ext_terminator(a, (ExtTerminator) {
                .mem = shd_bld_mem(bb),
                .set = "shady.internal",
                .opcode = ShadyOpDispatcherContinue,
                .operands = shd_empty(a),
            }));
        }
        case Control_TAG: {
            Control payload = old->payload.control;
            if (shd_is_control_static(ctx->uses, old)) {
                const Node* old_jp = shd_first(get_abstraction_params(payload.inside));
                assert(old_jp->tag == Param_TAG);
                const Node* old_jp_type = old_jp->type;
                shd_deconstruct_qualified_type(&old_jp_type);
                assert(old_jp_type->tag == JoinPointType_TAG);
                const Node* new_jp_type = shd_recreate_node(r, old_jp_type);
                const Node* new_jp = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.gang, new_jp_type));
                shd_rewrite_annotations(r, old_jp, new_jp);
                shd_register_processed(&ctx->rewriter, old_jp, new_jp);
                Node* new_control_case = basic_block_helper(a, shd_singleton(new_jp));
                shd_register_processed(r, payload.inside, new_control_case);
                shd_set_abstraction_body(new_control_case, shd_rewrite_node(&ctx->rewriter, get_abstraction_body(payload.inside)));
                Nodes nyield_types = shd_rewrite_nodes(&ctx->rewriter, payload.yield_types);
                return control(a, (Control) {
                    .yield_types = nyield_types,
                    .inside = new_control_case,
                    .tail = shd_rewrite_node(r, get_structured_construct_tail(old)),
                    .mem = shd_rewrite_node(r, payload.mem),
                });
            }
            break;
        }
        default:
            break;
    }
    rebuild:
    return shd_recreate_node(&ctx->rewriter, old);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

Module* shd_pass_lower_dynamic_control(const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .disable_lowering = false,
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
