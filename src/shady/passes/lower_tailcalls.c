#include "shady/pass.h"
#include "join_point_ops.h"

#include "../ir_private.h"

#include "../analysis/cfg.h"
#include "../analysis/uses.h"
#include "../analysis/leak.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"
#include "util.h"
#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

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
    Node* init_fn;
} Context;

static const Node* process(Context* ctx, const Node* old);

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
    IrArena* a = ctx->rewriter.dst_arena;
    // For the lifted entry point, we keep _all_ annotations
    Nodes rewritten_params = shd_recreate_params(&ctx2.rewriter, old->payload.fun.params);
    Node* new_entry_pt = function(ctx2.rewriter.dst_module, rewritten_params, old->payload.fun.name, shd_rewrite_nodes(&ctx2.rewriter, old->payload.fun.annotations), shd_nodes(a, 0, NULL));

    BodyBuilder* bb = begin_body_with_mem(a, shd_get_abstraction_mem(new_entry_pt));

    gen_call(bb, fn_addr_helper(a, ctx->init_fn), shd_empty(a));
    gen_call(bb, access_decl(&ctx->rewriter, "builtin_init_scheduler"), shd_empty(a));

    // shove the arguments on the stack
    for (size_t i = rewritten_params.count - 1; i < rewritten_params.count; i--) {
        gen_push_value_stack(bb, rewritten_params.nodes[i]);
    }

    // Initialise next_fn/next_mask to the entry function
    const Node* jump_fn = access_decl(&ctx->rewriter, "builtin_fork");
    const Node* fn_addr = shd_uint32_literal(a, get_fn_ptr(ctx, old));
    // fn_addr = gen_conversion(bb, lowered_fn_type(ctx), fn_addr);
    gen_call(bb, jump_fn, shd_singleton(fn_addr));

    if (!*ctx->top_dispatcher_fn) {
        *ctx->top_dispatcher_fn = function(ctx->rewriter.dst_module, shd_nodes(a, 0, NULL), "top_dispatcher", mk_nodes(a, annotation(a, (Annotation) { .name = "Generated" }), annotation(a, (Annotation) { .name = "Leaf" })), shd_nodes(a, 0, NULL));
    }

    gen_call(bb, fn_addr_helper(a, *ctx->top_dispatcher_fn), shd_empty(a));

    shd_set_abstraction_body(new_entry_pt, finish_body(bb, fn_ret(a, (Return) {
        .args = shd_nodes(a, 0, NULL),
        .mem = bb_mem(bb),
    })));
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (old->tag) {
        case Function_TAG: {
            Context ctx2 = *ctx;
            ctx2.cfg = build_fn_cfg(old);
            ctx2.uses = create_fn_uses_map(old, (NcDeclaration | NcType));
            ctx = &ctx2;

            const Node* entry_point_annotation = shd_lookup_annotation_list(old->payload.fun.annotations, "EntryPoint");

            // Leave leaf-calls alone :)
            ctx2.disable_lowering = shd_lookup_annotation(old, "Leaf") || !old->payload.fun.body;
            if (ctx2.disable_lowering) {
                Node* fun = shd_recreate_node_head(&ctx2.rewriter, old);
                if (old->payload.fun.body) {
                    BodyBuilder* bb = begin_body_with_mem(a, shd_get_abstraction_mem(fun));
                    if (entry_point_annotation) {
                        gen_call(bb, fn_addr_helper(a, ctx2.init_fn), shd_empty(a));
                    }
                    shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(old), bb_mem(bb));
                    shd_set_abstraction_body(fun, finish_body(bb, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old))));
                }

                destroy_uses_map(ctx2.uses);
                destroy_cfg(ctx2.cfg);
                return fun;
            }

            assert(ctx->config->dynamic_scheduling && "Dynamic scheduling is disabled, but we encountered a non-leaf function");

            Nodes new_annotations = shd_rewrite_nodes(&ctx->rewriter, old->payload.fun.annotations);
            new_annotations = shd_nodes_append(a, new_annotations, annotation_value(a, (AnnotationValue) { .name = "FnId", .value = lower_fn_addr(ctx, old) }));
            new_annotations = shd_nodes_append(a, new_annotations, annotation(a, (Annotation) { .name = "Leaf" }));

            String new_name = shd_format_string_arena(a->arena, "%s_indirect", old->payload.fun.name);

            Node* fun = function(ctx->rewriter.dst_module, shd_nodes(a, 0, NULL), new_name, shd_filter_out_annotation(a, new_annotations, "EntryPoint"), shd_nodes(a, 0, NULL));
            shd_register_processed(&ctx->rewriter, old, fun);

            if (entry_point_annotation)
                lift_entry_point(ctx, old, fun);

            BodyBuilder* bb = begin_body_with_mem(a, shd_get_abstraction_mem(fun));
            // Params become stack pops !
            for (size_t i = 0; i < old->payload.fun.params.count; i++) {
                const Node* old_param = old->payload.fun.params.nodes[i];
                const Type* new_param_type = shd_rewrite_node(&ctx->rewriter, get_unqualified_type(old_param->type));
                const Node* popped = gen_pop_value_stack(bb, new_param_type);
                // TODO use the uniform stack instead ? or no ?
                if (is_qualified_type_uniform(old_param->type))
                    popped = prim_op(a, (PrimOp) { .op = subgroup_assume_uniform_op, .type_arguments = shd_empty(a), .operands = shd_singleton(popped) });
                if (old_param->payload.param.name)
                    shd_set_value_name((Node*) popped, old_param->payload.param.name);
                shd_register_processed(&ctx->rewriter, old_param, popped);
            }
            shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(old), bb_mem(bb));
            shd_set_abstraction_body(fun, finish_body(bb, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old))));
            destroy_uses_map(ctx2.uses);
            destroy_cfg(ctx2.cfg);
            return fun;
        }
        case FnAddr_TAG: return lower_fn_addr(ctx, old->payload.fn_addr.fn);
        case Call_TAG: {
            Call payload = old->payload.call;
            assert(payload.callee->tag == FnAddr_TAG && "Only direct calls should survive this pass");
            return call(a, (Call) {
                .callee = fn_addr_helper(a, shd_rewrite_node(&ctx->rewriter, payload.callee->payload.fn_addr.fn)),
                .args = shd_rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = shd_rewrite_node(r, payload.mem)
            });
        }
        case JoinPointType_TAG: return type_decl_ref(a, (TypeDeclRef) {
            .decl = find_or_process_decl(&ctx->rewriter, "JoinPoint"),
        });
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
                    .callee = access_decl(r, callee_name),
                    .args = args,
                });
            }
            break;
        }
        case TailCall_TAG: {
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);
            TailCall payload = old->payload.tail_call;
            BodyBuilder* bb = begin_body_with_mem(a, shd_rewrite_node(r, payload.mem));
            gen_push_values_stack(bb, shd_rewrite_nodes(&ctx->rewriter, payload.args));
            const Node* target = shd_rewrite_node(&ctx->rewriter, payload.callee);
            target = gen_conversion(bb, shd_uint32_type(a), target);

            gen_call(bb, access_decl(&ctx->rewriter, "builtin_fork"), shd_singleton(target));
            return finish_body(bb, fn_ret(a, (Return) { .args = shd_empty(a), .mem = bb_mem(bb) }));
        }
        case Join_TAG: {
            Join payload = old->payload.join;
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);

            const Node* jp = shd_rewrite_node(&ctx->rewriter, old->payload.join.join_point);
            const Node* jp_type = jp->type;
            deconstruct_qualified_type(&jp_type);
            if (jp_type->tag == JoinPointType_TAG)
                break;

            BodyBuilder* bb = begin_body_with_mem(a, shd_rewrite_node(r, payload.mem));
            gen_push_values_stack(bb, shd_rewrite_nodes(&ctx->rewriter, old->payload.join.args));
            const Node* jp_payload = gen_primop_e(bb, extract_op, shd_empty(a), mk_nodes(a, jp, shd_int32_literal(a, 2)));
            gen_push_value_stack(bb, jp_payload);
            const Node* dst = gen_primop_e(bb, extract_op, shd_empty(a), mk_nodes(a, jp, shd_int32_literal(a, 1)));
            const Node* tree_node = gen_primop_e(bb, extract_op, shd_empty(a), mk_nodes(a, jp, shd_int32_literal(a, 0)));

            gen_call(bb, access_decl(&ctx->rewriter, "builtin_join"), mk_nodes(a, dst, tree_node));
            return finish_body(bb, fn_ret(a, (Return) { .args = shd_empty(a), .mem = bb_mem(bb) }));
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
            if (is_control_static(ctx->uses, old)) {
                // const Node* old_inside = old->payload.control.inside;
                const Node* old_jp = shd_first(get_abstraction_params(payload.inside));
                assert(old_jp->tag == Param_TAG);
                const Node* old_jp_type = old_jp->type;
                deconstruct_qualified_type(&old_jp_type);
                assert(old_jp_type->tag == JoinPointType_TAG);
                const Node* new_jp_type = join_point_type(a, (JoinPointType) {
                    .yield_types = shd_rewrite_nodes(&ctx->rewriter, old_jp_type->payload.join_point_type.yield_types),
                });
                const Node* new_jp = param(a, shd_as_qualified_type(new_jp_type, true), old_jp->payload.param.name);
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
    IrArena* a = ctx->rewriter.dst_arena;

    BodyBuilder* dispatcher_body_builder = begin_body_with_mem(a, shd_get_abstraction_mem(*ctx->top_dispatcher_fn));

    bool count_iterations = ctx->config->shader_diagnostics.max_top_iterations > 0;

    const Node* iterations_count_param = NULL;
    // if (count_iterations)
    //     iterations_count_param = param(a, qualified_type(a, (QualifiedType) { .type = int32_type(a), .is_uniform = true }), "iterations");

    // Node* loop_inside_case = case_(a, count_iterations ? singleton(iterations_count_param) : shd_nodes(a, 0, NULL));
    // gen_loop(dispatcher_body_builder, empty(a), count_iterations ? singleton(int32_literal(a, 0)) : empty(a), loop_inside_case);
    begin_loop_helper_t l = begin_loop_helper(dispatcher_body_builder, shd_empty(a), count_iterations ? shd_singleton(shd_int32_type(a)) : shd_empty(a), count_iterations ? shd_singleton(shd_int32_literal(a, 0)) : shd_empty(a));
    Node* loop_inside_case = l.loop_body;
    if (count_iterations)
        iterations_count_param = shd_first(l.params);
    BodyBuilder* loop_body_builder = begin_body_with_mem(a, shd_get_abstraction_mem(loop_inside_case));

    const Node* next_function = gen_load(loop_body_builder, access_decl(&ctx->rewriter, "next_fn"));
    const Node* get_active_branch_fn = access_decl(&ctx->rewriter, "builtin_get_active_branch");
    const Node* next_mask = shd_first(gen_call(loop_body_builder, get_active_branch_fn, shd_empty(a)));
    const Node* local_id = gen_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupLocalInvocationId);
    const Node* should_run = gen_primop_e(loop_body_builder, mask_is_thread_active_op, shd_empty(a), mk_nodes(a, next_mask, local_id));

    if (ctx->config->printf_trace.god_function) {
        const Node* sid = gen_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupId);
        if (count_iterations)
            gen_debug_printf(loop_body_builder, "trace: top loop, thread:%d:%d iteration=%d next_fn=%d next_mask=%lx\n", mk_nodes(a, sid, local_id, iterations_count_param, next_function, next_mask));
        else
            gen_debug_printf(loop_body_builder, "trace: top loop, thread:%d:%d next_fn=%d next_mask=%lx\n", mk_nodes(a, sid, local_id, next_function, next_mask));
    }

    const Node* iteration_count_plus_one = NULL;
    if (count_iterations)
        iteration_count_plus_one = gen_primop_e(loop_body_builder, add_op, shd_empty(a), mk_nodes(a, iterations_count_param, shd_int32_literal(a, 1)));

    if (ctx->config->shader_diagnostics.max_top_iterations > 0) {
        begin_control_t c = begin_control(loop_body_builder, shd_empty(a));
        const Node* bail_condition = gen_primop_e(loop_body_builder, gt_op, shd_empty(a), mk_nodes(a, iterations_count_param, shd_int32_literal(a, ctx->config->shader_diagnostics.max_top_iterations)));
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
    BodyBuilder* zero_if_case_builder = begin_body_with_mem(a, shd_get_abstraction_mem(zero_if_true_lam));
    if (ctx->config->printf_trace.god_function) {
        const Node* sid = gen_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupId);
        gen_debug_printf(zero_if_case_builder, "trace: kill thread %d:%d\n", mk_nodes(a, sid, local_id));
    }
    shd_set_abstraction_body(zero_if_true_lam, finish_body_with_join(zero_if_case_builder, l.break_jp, shd_empty(a)));

    Node* zero_if_false = case_(a, shd_empty(a));
    BodyBuilder* zero_false_builder = begin_body_with_mem(a, shd_get_abstraction_mem(zero_if_false));
    if (ctx->config->printf_trace.god_function) {
        const Node* sid = gen_builtin_load(ctx->rewriter.dst_module, zero_false_builder, BuiltinSubgroupId);
        gen_debug_printf(zero_false_builder, "trace: thread %d:%d escaped death!\n", mk_nodes(a, sid, local_id));
    }
    shd_set_abstraction_body(zero_if_false, finish_body_with_join(zero_false_builder, l.continue_jp, count_iterations ? shd_singleton(iteration_count_plus_one) : shd_empty(a)));

    shd_set_abstraction_body(zero_case_lam, branch(a, (Branch) {
        .mem = shd_get_abstraction_mem(zero_case_lam),
        .condition = should_run,
        .true_jump = jump_helper(a, shd_get_abstraction_mem(zero_case_lam), zero_if_true_lam, shd_empty(a)),
        .false_jump = jump_helper(a, shd_get_abstraction_mem(zero_case_lam), zero_if_false, shd_empty(a)),
    }));

    const Node* zero_lit = shd_uint64_literal(a, 0);
    shd_list_append(const Node*, literals, zero_lit);
    const Node* zero_jump = jump_helper(a, bb_mem(loop_body_builder), zero_case_lam, shd_empty(a));
    shd_list_append(const Node*, jumps, zero_jump);

    Nodes old_decls = shd_module_get_declarations(ctx->rewriter.src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (decl->tag == Function_TAG) {
            if (shd_lookup_annotation(decl, "Leaf"))
                continue;

            const Node* fn_lit = shd_uint32_literal(a, get_fn_ptr(ctx, decl));

            Node* if_true_case = case_(a, shd_empty(a));
            BodyBuilder* if_builder = begin_body_with_mem(a, shd_get_abstraction_mem(if_true_case));
            if (ctx->config->printf_trace.god_function) {
                const Node* sid = gen_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupId);
                gen_debug_printf(if_builder, "trace: thread %d:%d will run fn %u with mask = %lx\n", mk_nodes(a, sid, local_id, fn_lit, next_mask));
            }
            gen_call(if_builder, fn_addr_helper(a, shd_rewrite_node(&ctx->rewriter, decl)), shd_empty(a));
            shd_set_abstraction_body(if_true_case, finish_body_with_join(if_builder, l.continue_jp, count_iterations ? shd_singleton(iteration_count_plus_one) : shd_empty(a)));

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
            const Node* j = jump_helper(a, bb_mem(loop_body_builder), fn_case, shd_empty(a));
            shd_list_append(const Node*, jumps, j);
        }
    }

    Node* default_case = case_(a, shd_nodes(a, 0, NULL));
    shd_set_abstraction_body(default_case, unreachable(a, (Unreachable) { .mem = shd_get_abstraction_mem(default_case) }));

    shd_set_abstraction_body(loop_inside_case, finish_body(loop_body_builder, br_switch(a, (Switch) {
        .mem = bb_mem(loop_body_builder),
        .switch_value = next_function,
        .case_values = shd_nodes(a, shd_list_count(literals), shd_read_list(const Node*, literals)),
        .case_jumps = shd_nodes(a, shd_list_count(jumps), shd_read_list(const Node*, jumps)),
        .default_jump = jump_helper(a, bb_mem(loop_body_builder), default_case, shd_empty(a))
    })));

    shd_destroy_list(literals);
    shd_destroy_list(jumps);

    if (ctx->config->printf_trace.god_function)
        gen_debug_printf(dispatcher_body_builder, "trace: end of top\n", shd_empty(a));

    shd_set_abstraction_body(*ctx->top_dispatcher_fn, finish_body(dispatcher_body_builder, fn_ret(a, (Return) {
        .args = shd_nodes(a, 0, NULL),
        .mem = bb_mem(dispatcher_body_builder),
    })));
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

Module* shd_pass_lower_tailcalls(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    struct Dict* ptrs = shd_new_dict(const Node*, FnPtr, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);

    Node* init_fn = function(dst, shd_nodes(a, 0, NULL), "generated_init", mk_nodes(a, annotation(a, (Annotation) { .name = "Generated" }), annotation(a, (Annotation) { .name = "Leaf" })), shd_nodes(a, 0, NULL));
    shd_set_abstraction_body(init_fn, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_get_abstraction_mem(init_fn) }));

    FnPtr next_fn_ptr = 1;

    Node* top_dispatcher_fn = NULL;

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .disable_lowering = false,
        .assigned_fn_ptrs = ptrs,
        .next_fn_ptr = &next_fn_ptr,

        .top_dispatcher_fn = &top_dispatcher_fn,
        .init_fn = init_fn,
    };

    shd_rewrite_module(&ctx.rewriter);

    // Generate the top dispatcher, but only if it is used for realsies
    if (*ctx.top_dispatcher_fn)
        generate_top_level_dispatch_fn(&ctx);

    shd_destroy_dict(ptrs);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
