#include "join_point_ops.h"

#include "shady/pass.h"
#include "shady/ir/stack.h"
#include "shady/ir/cast.h"
#include "shady/ir/builtin.h"

#include "ir_private.h"
#include "analysis/cfg.h"
#include "analysis/uses.h"
#include "analysis/leak.h"

#include "log.h"
#include "portability.h"
#include "util.h"
#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>
#include "shady/ir/memory_layout.h"

typedef uint64_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;
    bool disable_lowering;
    struct Dict* assigned_fn_ptrs;
    FnPtr* next_fn_ptr;

    CFG* cfg;
    const UsesMap* uses;

    Node** top_dispatcher_fn;
} Context;

static const Node* process(Context* ctx, const Node* old);

static const Node* get_fn(Rewriter* rewriter, const char* name) {
    const Node* decl = shd_find_or_process_decl(rewriter, name);
    return fn_addr_helper(rewriter->dst_arena, decl);
}

static const Type* lowered_fn_type(Context* ctx) {
    IrArena* a = ctx->rewriter.dst_arena;
    return shd_int_type_helper(a, false, ctx->config->target.memory.ptr_size);
}

static const Node* fn_ptr_as_value(Context* ctx, FnPtr ptr) {
    IrArena* a = ctx->rewriter.dst_arena;
    return int_literal(a, (IntLiteral) {
        .is_signed = false,
        .width = ctx->config->target.memory.ptr_size,
        .value = ptr
    });
}

static FnPtr get_fn_ptr(Context* ctx, const Node* the_function) {
    assert(the_function->arena == ctx->rewriter.src_arena);
    assert(the_function->tag == Function_TAG);

    FnPtr* found = shd_dict_find_value(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function);
    if (found) return *found;

    FnPtr ptr = (*ctx->next_fn_ptr)++;
    bool r = shd_dict_insert_get_result(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function, ptr);
    assert(r);
    return ptr;
}

static const Node* lower_fn_addr(Context* ctx, const Node* the_function) {
    return fn_ptr_as_value(ctx, get_fn_ptr(ctx, the_function));
}

/// Turn a function into a top-level entry point, calling into the top dispatch function.
static void lift_entry_point(Context* ctx, const Node* old, const Node* fun) {
    assert(old->tag == Function_TAG && fun->tag == Function_TAG);
    Context ctx2 = *ctx;
    Rewriter* r = &ctx2.rewriter;
    IrArena* a = ctx->rewriter.dst_arena;
    // For the lifted entry point, we keep _all_ annotations
    Nodes rewritten_params = shd_recreate_params(&ctx2.rewriter, old->payload.fun.params);
    Node* new_entry_pt = function_helper(ctx2.rewriter.dst_module, rewritten_params, old->payload.fun.name, shd_nodes(a, 0, NULL));
    shd_rewrite_annotations(r, old, new_entry_pt);

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new_entry_pt));

    shd_bld_call(bb, get_fn(&ctx->rewriter, "builtin_init_scheduler"), shd_empty(a));

    // shove the arguments on the stack
    for (size_t i = rewritten_params.count - 1; i < rewritten_params.count; i--) {
        shd_bld_stack_push_value(bb, rewritten_params.nodes[i]);
    }

    // Initialise next_fn/next_mask to the entry function
    const Node* fork_fn = get_fn(&ctx->rewriter, "builtin_fork");
    const Node* entry_point_addr = shd_uint32_literal(a, get_fn_ptr(ctx, old));
    // fn_addr = gen_conversion(bb, lowered_fn_type(ctx), fn_addr);
    shd_bld_call(bb, fork_fn, shd_singleton(entry_point_addr));

    if (!*ctx->top_dispatcher_fn) {
        *ctx->top_dispatcher_fn = function_helper(ctx->rewriter.dst_module, shd_nodes(a, 0, NULL), "top_dispatcher", shd_nodes(a, 0, NULL));
        shd_add_annotation_named(*ctx->top_dispatcher_fn, "Generated");
        shd_add_annotation_named(*ctx->top_dispatcher_fn, "Leaf");
    }

    shd_bld_call(bb, fn_addr_helper(a, *ctx->top_dispatcher_fn), shd_empty(a));

    shd_set_abstraction_body(new_entry_pt, shd_bld_return(bb, shd_empty(a)));
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (old->tag) {
        case Function_TAG: {
            Context ctx2 = *ctx;
            ctx2.cfg = build_fn_cfg(old);
            ctx2.uses = shd_new_uses_map_fn(old, (NcDeclaration | NcType));
            ctx = &ctx2;

            const Node* entry_point_annotation = shd_lookup_annotation(old, "EntryPoint");

            // Leave leaf-calls alone :)
            ctx2.disable_lowering = shd_lookup_annotation(old, "Leaf") || !old->payload.fun.body;
            if (ctx2.disable_lowering) {
                Node* fun = shd_recreate_node_head(&ctx2.rewriter, old);
                if (old->payload.fun.body) {
                    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));
                    if (entry_point_annotation) {
                        // shd_bld_call(bb, fn_addr_helper(a, ctx2.init_fn), shd_empty(a));
                    }
                    shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(old), shd_bld_mem(bb));
                    shd_set_abstraction_body(fun, shd_bld_finish(bb, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old))));
                }

                shd_destroy_uses_map(ctx2.uses);
                shd_destroy_cfg(ctx2.cfg);
                return fun;
            }

            assert(ctx->config->dynamic_scheduling && "Dynamic scheduling is disabled, but we encountered a non-leaf function");


            String new_name = shd_format_string_arena(a->arena, "%s_indirect", old->payload.fun.name);

            shd_remove_annotation_by_name(old, "EntryPoint");
            Node* fun = function_helper(ctx->rewriter.dst_module, shd_nodes(a, 0, NULL), new_name, shd_nodes(a, 0, NULL));
            shd_rewrite_annotations(r, old, fun);
            shd_add_annotation_named(fun, "Leaf");
            shd_add_annotation(fun, annotation_value(a, (AnnotationValue) { .name = "FnId", .value = lower_fn_addr(ctx, old) }));

            shd_register_processed(&ctx->rewriter, old, fun);

            if (entry_point_annotation)
                lift_entry_point(ctx, old, fun);

            BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));
            // Params become stack pops !
            for (size_t i = 0; i < old->payload.fun.params.count; i++) {
                const Node* old_param = old->payload.fun.params.nodes[i];
                const Type* new_param_type = shd_rewrite_node(&ctx->rewriter, shd_get_unqualified_type(old_param->type));
                const Node* popped = shd_bld_stack_pop_value(bb, new_param_type);
                // TODO use the uniform stack instead ? or no ?
                if (shd_is_qualified_type_uniform(old_param->type))
                    popped = prim_op(a, (PrimOp) { .op = subgroup_assume_uniform_op, .type_arguments = shd_empty(a), .operands = shd_singleton(popped) });
                if (old_param->payload.param.name)
                    shd_set_value_name((Node*) popped, old_param->payload.param.name);
                shd_register_processed(&ctx->rewriter, old_param, popped);
            }
            shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(old), shd_bld_mem(bb));
            shd_set_abstraction_body(fun, shd_bld_finish(bb, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old))));
            shd_destroy_uses_map(ctx2.uses);
            shd_destroy_cfg(ctx2.cfg);
            return fun;
        }
        case FnAddr_TAG: return lower_fn_addr(ctx, old->payload.fn_addr.fn);
        case Call_TAG: {
            Call payload = old->payload.call;
            assert(payload.callee->tag == FnAddr_TAG && "Only direct calls should survive this pass");
            FnAddr callee = payload.callee->payload.fn_addr;
            const Node* ncallee = shd_rewrite_node(&ctx->rewriter, payload.callee->payload.fn_addr.fn);
            if (!ctx->disable_lowering) {
                return shd_rewrite_node(r, payload.mem);
            }
            return call(a, (Call) {
                .callee = fn_addr_helper(a, ncallee),
                .args = shd_rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = shd_rewrite_node(r, payload.mem)
            });
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
                        args = shd_change_node_at_index(a, args, 0, prim_op_helper(a, convert_op, shd_singleton(shd_uint32_type(a)), shd_singleton(args.nodes[0])));
                        break;
                }
                return call(a, (Call) {
                    .mem = shd_rewrite_node(r, payload.mem),
                    .callee = get_fn(r, callee_name),
                    .args = args,
                });
            }
            break;
        }
        case TailCall_TAG: {
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);
            TailCall payload = old->payload.tail_call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_stack_push_values(bb, shd_rewrite_nodes(&ctx->rewriter, payload.args));
            const Node* target = shd_rewrite_node(&ctx->rewriter, payload.callee);
            target = shd_bld_conversion(bb, shd_uint32_type(a), target);

            shd_bld_call(bb, get_fn(&ctx->rewriter, "builtin_fork"), shd_singleton(target));
            return shd_bld_finish(bb, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_bld_mem(bb) }));
        }
        case Join_TAG: {
            Join payload = old->payload.join;
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);

            const Node* jp = shd_rewrite_node(&ctx->rewriter, old->payload.join.join_point);
            const Node* jp_type = jp->type;
            shd_deconstruct_qualified_type(&jp_type);
            if (jp_type->tag == JoinPointType_TAG)
                break;

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_stack_push_values(bb, shd_rewrite_nodes(&ctx->rewriter, old->payload.join.args));
            const Node* jp_payload = prim_op_helper(a, extract_op, shd_empty(a), mk_nodes(a, jp, shd_int32_literal(a, 2)));
            shd_bld_stack_push_value(bb, jp_payload);
            const Node* dst = prim_op_helper(a, extract_op, shd_empty(a), mk_nodes(a, jp, shd_int32_literal(a, 1)));
            const Node* tree_node = prim_op_helper(a, extract_op, shd_empty(a), mk_nodes(a, jp, shd_int32_literal(a, 0)));

            shd_bld_call(bb, get_fn(&ctx->rewriter, "builtin_join"), mk_nodes(a, dst, tree_node));
            return shd_bld_finish(bb, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_bld_mem(bb) }));
        }
        case PtrType_TAG: {
            const Node* pointee = old->payload.ptr_type.pointed_type;
            if (pointee->tag == FnType_TAG) {
                const Type* emulated_fn_ptr_type = shd_uint64_type(a);
                return emulated_fn_ptr_type;
            }
            break;
        }
        case Control_TAG: {
            Control payload = old->payload.control;
            if (shd_is_control_static(ctx->uses, old)) {
                // const Node* old_inside = old->payload.control.inside;
                const Node* old_jp = shd_first(get_abstraction_params(payload.inside));
                assert(old_jp->tag == Param_TAG);
                const Node* old_jp_type = old_jp->type;
                shd_deconstruct_qualified_type(&old_jp_type);
                assert(old_jp_type->tag == JoinPointType_TAG);
                const Node* new_jp_type = join_point_type(a, (JoinPointType) {
                    .yield_types = shd_rewrite_nodes(&ctx->rewriter, old_jp_type->payload.join_point_type.yield_types),
                });
                const Node* new_jp = param_helper(a, shd_as_qualified_type(new_jp_type, true), old_jp->payload.param.name);
                shd_register_processed(&ctx->rewriter, old_jp, new_jp);
                Node* new_control_case = case_(a, shd_singleton(new_jp));
                shd_register_processed(r, payload.inside, new_control_case);
                shd_set_abstraction_body(new_control_case, shd_rewrite_node(&ctx->rewriter, get_abstraction_body(payload.inside)));
                // BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, payload.mem));
                Nodes nyield_types = shd_rewrite_nodes(&ctx->rewriter, old->payload.control.yield_types);
                return control(a, (Control) {
                    .yield_types = nyield_types,
                    .inside = new_control_case,
                    .tail = shd_rewrite_node(r, get_structured_construct_tail(old)),
                    .mem = shd_rewrite_node(r, payload.mem),
                });
                //return yield_values_and_wrap_in_block(bb, gen_control(bb, nyield_types, new_body));
            }
            break;
        }
        default:
            break;
    }
    return shd_recreate_node(&ctx->rewriter, old);
}

static void generate_top_level_dispatch_fn(Context* ctx) {
    assert(ctx->config->dynamic_scheduling);
    assert(*ctx->top_dispatcher_fn);
    assert((*ctx->top_dispatcher_fn)->tag == Function_TAG);
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    BodyBuilder* dispatcher_body_builder = shd_bld_begin(a, shd_get_abstraction_mem(*ctx->top_dispatcher_fn));

    bool count_iterations = ctx->config->shader_diagnostics.max_top_iterations > 0;

    const Node* iterations_count_param = NULL;
    // if (count_iterations)
    //     iterations_count_param = param(a, qualified_type(a, (QualifiedType) { .type = int32_type(a), .is_uniform = true }), "iterations");

    // Node* loop_inside_case = case_(a, count_iterations ? singleton(iterations_count_param) : shd_nodes(a, 0, NULL));
    // gen_loop(dispatcher_body_builder, empty(a), count_iterations ? singleton(int32_literal(a, 0)) : empty(a), loop_inside_case);
    begin_loop_helper_t l = shd_bld_begin_loop_helper(dispatcher_body_builder, shd_empty(a), count_iterations ? shd_singleton(shd_int32_type(a)) : shd_empty(a), count_iterations ? shd_singleton(shd_int32_literal(a, 0)) : shd_empty(a));
    Node* loop_inside_case = l.loop_body;
    if (count_iterations)
        iterations_count_param = shd_first(l.params);
    BodyBuilder* loop_body_builder = shd_bld_begin(a, shd_get_abstraction_mem(loop_inside_case));

    const Node* next_function = shd_bld_load(loop_body_builder, shd_find_or_process_decl(r, "next_fn"));
    const Node* get_active_branch_fn = get_fn(r, "builtin_get_active_branch");
    const Node* next_mask = shd_first(shd_bld_call(loop_body_builder, get_active_branch_fn, shd_empty(a)));
    const Node* local_id = shd_bld_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupLocalInvocationId);
    const Node* should_run = prim_op_helper(a, mask_is_thread_active_op, shd_empty(a), mk_nodes(a, next_mask, local_id));

    const Node* sid = shd_bld_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupId);
    if (ctx->config->printf_trace.top_function) {
        const Node* resume_at = shd_bld_load(loop_body_builder, lea_helper(a, shd_find_or_process_decl(r, "resume_at"), shd_uint32_literal(a, 0), mk_nodes(a, local_id)));
        if (count_iterations)
            shd_bld_debug_printf(loop_body_builder, "trace: top loop, thread:%d:%d iteration=%d next_fn=%d next_mask=%lx resume_at=%d\n", mk_nodes(a, sid, local_id, iterations_count_param, next_function, next_mask, resume_at));
        else
            shd_bld_debug_printf(loop_body_builder, "trace: top loop, thread:%d:%d next_fn=%d next_mask=%lx resume_at=%d\n", mk_nodes(a, sid, local_id, next_function, next_mask, resume_at));
    }

    const Node* iteration_count_plus_one = NULL;
    if (count_iterations)
        iteration_count_plus_one = prim_op_helper(a, add_op, shd_empty(a), mk_nodes(a, iterations_count_param, shd_int32_literal(a, 1)));

    if (ctx->config->shader_diagnostics.max_top_iterations > 0) {
        begin_control_t c = shd_bld_begin_control(loop_body_builder, shd_empty(a));
        const Node* bail_condition = prim_op_helper(a, gt_op, shd_empty(a), mk_nodes(a, iterations_count_param, shd_int32_literal(a, ctx->config->shader_diagnostics.max_top_iterations)));
        Node* bail_case = case_(a, shd_empty(a));
        const Node* break_terminator = join(a, (Join) { .args = shd_empty(a), .join_point = l.break_jp, .mem = shd_get_abstraction_mem(bail_case) });
        shd_set_abstraction_body(bail_case, break_terminator);
        Node* proceed_case = case_(a, shd_empty(a));
        shd_set_abstraction_body(proceed_case, join(a, (Join) {
            .join_point = c.jp,
            .mem = shd_get_abstraction_mem(proceed_case),
            .args = shd_empty(a),
        }));
        shd_set_abstraction_body(c.case_, branch(a, (Branch) {
            .mem = shd_get_abstraction_mem(c.case_),
            .condition = bail_condition,
            .true_jump = jump_helper(a, shd_get_abstraction_mem(c.case_), bail_case, shd_empty(a)),
            .false_jump = jump_helper(a, shd_get_abstraction_mem(c.case_), proceed_case, shd_empty(a)),
        }));
        // gen_if(loop_body_builder, empty(a), bail_condition, bail_case, NULL);
    }

    struct List* literals = shd_new_list(const Node*);
    struct List* jumps = shd_new_list(const Node*);

    // Build 'zero' case (exits the program)
    Node* zero_case_lam = case_(a, shd_nodes(a, 0, NULL));
    Node* zero_if_true_lam = case_(a, shd_empty(a));
    BodyBuilder* zero_if_case_builder = shd_bld_begin(a, shd_get_abstraction_mem(zero_if_true_lam));
    if (ctx->config->printf_trace.top_function) {
        shd_bld_debug_printf(zero_if_case_builder, "trace: kill thread %d:%d\n", mk_nodes(a, sid, local_id));
    }
    shd_set_abstraction_body(zero_if_true_lam, shd_bld_join(zero_if_case_builder, l.break_jp, shd_empty(a)));

    Node* zero_if_false = case_(a, shd_empty(a));
    BodyBuilder* zero_false_builder = shd_bld_begin(a, shd_get_abstraction_mem(zero_if_false));
    if (ctx->config->printf_trace.top_function) {
        shd_bld_debug_printf(zero_false_builder, "trace: thread %d:%d escaped death!\n", mk_nodes(a, sid, local_id));
    }
    shd_set_abstraction_body(zero_if_false, shd_bld_join(zero_false_builder, l.continue_jp, count_iterations ? shd_singleton(iteration_count_plus_one) : shd_empty(a)));

    shd_set_abstraction_body(zero_case_lam, branch(a, (Branch) {
        .mem = shd_get_abstraction_mem(zero_case_lam),
        .condition = should_run,
        .true_jump = jump_helper(a, shd_get_abstraction_mem(zero_case_lam), zero_if_true_lam, shd_empty(a)),
        .false_jump = jump_helper(a, shd_get_abstraction_mem(zero_case_lam), zero_if_false, shd_empty(a)),
    }));

    const Node* zero_lit = shd_uint64_literal(a, 0);
    shd_list_append(const Node*, literals, zero_lit);
    const Node* zero_jump = jump_helper(a, shd_bld_mem(loop_body_builder), zero_case_lam, shd_empty(a));
    shd_list_append(const Node*, jumps, zero_jump);

    Nodes old_decls = shd_module_get_declarations(ctx->rewriter.src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (decl->tag == Function_TAG) {
            if (shd_lookup_annotation(decl, "Leaf"))
                continue;

            const Node* fn_lit = shd_uint32_literal(a, get_fn_ptr(ctx, decl));

            Node* if_true_case = case_(a, shd_empty(a));
            BodyBuilder* if_builder = shd_bld_begin(a, shd_get_abstraction_mem(if_true_case));
            if (ctx->config->printf_trace.top_function) {
                shd_bld_debug_printf(if_builder, "trace: thread %d:%d will run fn %u with mask = %lx\n", mk_nodes(a, sid, local_id, fn_lit, next_mask));
            }
            shd_bld_call(if_builder, fn_addr_helper(a, shd_rewrite_node(r, decl)), shd_empty(a));
            shd_set_abstraction_body(if_true_case, shd_bld_join(if_builder, l.continue_jp, count_iterations ? shd_singleton(iteration_count_plus_one) : shd_empty(a)));

            Node* if_false = case_(a, shd_empty(a));
            shd_set_abstraction_body(if_false, join(a, (Join) {
                .mem = shd_get_abstraction_mem(if_false),
                .join_point = l.continue_jp,
                .args = count_iterations ? shd_singleton(iteration_count_plus_one) : shd_empty(a)
            }));

            Node* fn_case = case_(a, shd_nodes(a, 0, NULL));
            shd_set_abstraction_body(fn_case, branch(a, (Branch) {
                .mem = shd_get_abstraction_mem(fn_case),
                .condition = should_run,
                .true_jump = jump_helper(a, shd_get_abstraction_mem(fn_case), if_true_case, shd_empty(a)),
                .false_jump = jump_helper(a, shd_get_abstraction_mem(fn_case), if_false, shd_empty(a)),
            }));

            shd_list_append(const Node*, literals, fn_lit);
            const Node* j = jump_helper(a, shd_bld_mem(loop_body_builder), fn_case, shd_empty(a));
            shd_list_append(const Node*, jumps, j);
        }
    }

    Node* default_case = case_(a, shd_nodes(a, 0, NULL));
    shd_set_abstraction_body(default_case, unreachable(a, (Unreachable) { .mem = shd_get_abstraction_mem(default_case) }));

    shd_set_abstraction_body(loop_inside_case, shd_bld_finish(loop_body_builder, br_switch(a, (Switch) {
        .mem = shd_bld_mem(loop_body_builder),
        .switch_value = next_function,
        .case_values = shd_nodes(a, shd_list_count(literals), shd_read_list(const Node*, literals)),
        .case_jumps = shd_nodes(a, shd_list_count(jumps), shd_read_list(const Node*, jumps)),
        .default_jump = jump_helper(a, shd_bld_mem(loop_body_builder), default_case, shd_empty(a))
    })));

    shd_destroy_list(literals);
    shd_destroy_list(jumps);

    if (ctx->config->printf_trace.top_function)
        shd_bld_debug_printf(dispatcher_body_builder, "trace: end of top\n", shd_empty(a));

    shd_set_abstraction_body(*ctx->top_dispatcher_fn, shd_bld_finish(dispatcher_body_builder, fn_ret(a, (Return) {
        .args = shd_nodes(a, 0, NULL),
        .mem = shd_bld_mem(dispatcher_body_builder),
    })));
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

Module* shd_pass_lower_tailcalls(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    struct Dict* ptrs = shd_new_dict(const Node*, FnPtr, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);

    FnPtr next_fn_ptr = 1;

    Node* top_dispatcher_fn = NULL;

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .disable_lowering = false,
        .assigned_fn_ptrs = ptrs,
        .next_fn_ptr = &next_fn_ptr,

        .top_dispatcher_fn = &top_dispatcher_fn,
    };

    shd_rewrite_module(&ctx.rewriter);

    // Generate the top dispatcher, but only if it is used for realsies
    if (*ctx.top_dispatcher_fn)
        generate_top_level_dispatch_fn(&ctx);

    shd_destroy_dict(ptrs);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
