#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../rewrite.h"
#include "../type.h"

#include "../transform/ir_gen_helpers.h"

#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;
    bool disable_lowering;
    struct Dict* assigned_fn_ptrs;
    FnPtr* next_fn_ptr;

    Node** top_dispatcher_fn;
    Node* init_fn;
} Context;

static const Node* process(Context* ctx, const Node* old);

static const Node* fn_ptr_as_value(IrArena* arena, FnPtr ptr) {
    return uint32_literal(arena, ptr);
}

static const Node* lower_fn_addr(Context* ctx, const Node* the_function) {
    assert(the_function->arena == ctx->rewriter.src_arena);
    assert(the_function->tag == Function_TAG);

    FnPtr* found = find_value_dict(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function);
    if (found) return fn_ptr_as_value(ctx->rewriter.dst_arena, *found);

    FnPtr ptr = (*ctx->next_fn_ptr)++;
    bool r = insert_dict_and_get_result(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function, ptr);
    assert(r);
    return fn_ptr_as_value(ctx->rewriter.dst_arena, ptr);
}

/// Turn a function into a top-level entry point, calling into the top dispatch function.
static void lift_entry_point(Context* ctx, const Node* old, const Node* fun) {
    assert(old->tag == Function_TAG && fun->tag == Function_TAG);
    Context ctx2 = *ctx;
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    // For the lifted entry point, we keep _all_ annotations
    Nodes rewritten_params = recreate_variables(&ctx2.rewriter, old->payload.fun.params);
    Node* new_entry_pt = function(ctx2.rewriter.dst_module, rewritten_params, old->payload.fun.name, rewrite_nodes(&ctx2.rewriter, old->payload.fun.annotations), nodes(dst_arena, 0, NULL));

    BodyBuilder* builder = begin_body(dst_arena);

    bind_instruction(builder, call(dst_arena, (Call) { .callee = fn_addr(dst_arena, (FnAddr) { .fn = ctx->init_fn }), .args = empty(dst_arena) }));
    bind_instruction(builder, call(dst_arena, (Call) { .callee = access_decl(&ctx->rewriter, "builtin_init_scheduler"), .args = empty(dst_arena) }));

    // shove the arguments on the stack
    for (size_t i = rewritten_params.count - 1; i < rewritten_params.count; i--) {
        gen_push_value_stack(builder, rewritten_params.nodes[i]);
    }

    // Initialise next_fn/next_mask to the entry function
    const Node* jump_fn = access_decl(&ctx->rewriter, "builtin_fork");
    bind_instruction(builder, call(dst_arena, (Call) { .callee = jump_fn, .args = singleton(lower_fn_addr(ctx, old)) }));

    if (!*ctx->top_dispatcher_fn) {
        *ctx->top_dispatcher_fn = function(ctx->rewriter.dst_module, nodes(dst_arena, 0, NULL), "top_dispatcher", singleton(annotation(dst_arena, (Annotation) { .name = "Generated" })), nodes(dst_arena, 0, NULL));
    }

    bind_instruction(builder, call(dst_arena, (Call) {
        .callee = fn_addr(dst_arena, (FnAddr) { .fn = *ctx->top_dispatcher_fn }),
        .args = nodes(dst_arena, 0, NULL)
    }));

    new_entry_pt->payload.fun.body = finish_body(builder, fn_ret(dst_arena, (Return) {
        .fn = NULL,
        .args = nodes(dst_arena, 0, NULL)
    }));
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case Function_TAG: {
            Context ctx2 = *ctx;

            const Node* entry_point_annotation = lookup_annotation_list(old->payload.fun.annotations, "EntryPoint");

            // Leave leaf-calls alone :)
            ctx2.disable_lowering = lookup_annotation(old, "Leaf") || !old->payload.fun.body;
            if (ctx2.disable_lowering) {
                Node* fun = recreate_decl_header_identity(&ctx2.rewriter, old);
                if (old->payload.fun.body) {
                    const Node* nbody = rewrite_node(&ctx->rewriter, old->payload.fun.body);
                    if (entry_point_annotation) {
                        const Node* lam = lambda(ctx->rewriter.dst_arena, empty(dst_arena), nbody);
                        nbody = let(dst_arena, call(dst_arena, (Call) { .callee = fn_addr(dst_arena, (FnAddr) { .fn = ctx->init_fn }), .args = empty(dst_arena)}), lam);
                    }
                    fun->payload.fun.body = nbody;
                }
                return fun;
            }

            assert(ctx->config->dynamic_scheduling && "Dynamic scheduling is disabled, but we encountered a non-leaf function");

            Nodes new_annotations = rewrite_nodes(&ctx->rewriter, old->payload.fun.annotations);
            new_annotations = append_nodes(dst_arena, new_annotations, annotation_value(dst_arena, (AnnotationValue) { .name = "FnId", .value = lower_fn_addr(ctx, old) }));

            String new_name = format_string(dst_arena, "%s_indirect", old->payload.fun.name);

            Node* fun = function(ctx->rewriter.dst_module, nodes(dst_arena, 0, NULL), new_name, filter_out_annotation(dst_arena, new_annotations, "EntryPoint"), nodes(dst_arena, 0, NULL));
            register_processed(&ctx->rewriter, old, fun);

            if (entry_point_annotation)
                lift_entry_point(ctx, old, fun);

            BodyBuilder* bb = begin_body(dst_arena);
            // Params become stack pops !
            for (size_t i = 0; i < old->payload.fun.params.count; i++) {
                const Node* old_param = old->payload.fun.params.nodes[i];
                const Type* new_param_type = rewrite_node(&ctx->rewriter, get_unqualified_type(old_param->type));
                const Node* popped = first(bind_instruction_named(bb, prim_op(dst_arena, (PrimOp) { .op = pop_stack_op, .type_arguments = singleton(new_param_type), .operands = empty(dst_arena) }), &old_param->payload.var.name));
                // TODO use the uniform stack instead ? or no ?
                if (is_qualified_type_uniform(old_param->type))
                    popped = first(bind_instruction_named(bb, prim_op(dst_arena, (PrimOp) { .op = subgroup_broadcast_first_op, .type_arguments = empty(dst_arena), .operands =singleton(popped) }), &old_param->payload.var.name));
                register_processed(&ctx->rewriter, old_param, popped);
            }
            fun->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, old->payload.fun.body));
            return fun;
        }
        case FnAddr_TAG: return lower_fn_addr(ctx, old->payload.fn_addr.fn);
        case Call_TAG: {
            const Node* ocallee = old->payload.call.callee;
            assert(ocallee->tag == FnAddr_TAG);
            return call(dst_arena, (Call) {
                .callee = fn_addr(dst_arena, (FnAddr) { .fn = rewrite_node(&ctx->rewriter, ocallee->payload.fn_addr.fn )}),
                .args = rewrite_nodes(&ctx->rewriter, old->payload.call.args),
            });
        }
        case JoinPointType_TAG: return type_decl_ref(dst_arena, (TypeDeclRef) {
            .decl = find_or_process_decl(&ctx->rewriter, "JoinPoint"),
        });
        case PrimOp_TAG: {
            switch (old->payload.prim_op.op) {
                case create_joint_point_op: {
                    const Node* join_destination = rewrite_node(&ctx->rewriter, first(old->payload.prim_op.operands));
                    return call(dst_arena, (Call) {
                        .callee = access_decl(&ctx->rewriter, "builtin_create_control_point"),
                        .args = mk_nodes(dst_arena, join_destination)
                    });
                }
                case default_join_point_op: {
                    return call(dst_arena, (Call) {
                        .callee = access_decl(&ctx->rewriter, "builtin_entry_join_point"),
                        .args = empty(dst_arena)
                    });
                }
                default: return recreate_node_identity(&ctx->rewriter, old);
            }
        }
        case TailCall_TAG: {
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);
            BodyBuilder* bb = begin_body(dst_arena);
            gen_push_values_stack(bb, rewrite_nodes(&ctx->rewriter, old->payload.tail_call.args));
            const Node* target = rewrite_node(&ctx->rewriter, old->payload.tail_call.target);

            const Node* fork_call = call(dst_arena, (Call) {
                .callee = access_decl(&ctx->rewriter, "builtin_fork"),
                .args = nodes(dst_arena, 1, (const Node*[]) { target })
            });
            bind_instruction(bb, fork_call);
            return finish_body(bb, fn_ret(dst_arena, (Return) { .fn = NULL, .args = nodes(dst_arena, 0, NULL) }));
        }
        case Join_TAG: {
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);

            BodyBuilder* bb = begin_body(dst_arena);
            gen_push_values_stack(bb, rewrite_nodes(&ctx->rewriter, old->payload.join.args));

            const Node* jp = rewrite_node(&ctx->rewriter, old->payload.join.join_point);
            const Node* dst = gen_primop_e(bb, extract_op, empty(dst_arena), mk_nodes(dst_arena, jp, int32_literal(dst_arena, 1)));
            const Node* tree_node = gen_primop_e(bb, extract_op, empty(dst_arena), mk_nodes(dst_arena, jp, int32_literal(dst_arena, 0)));

            const Node* join_call = call(dst_arena, (Call) {
                .callee = access_decl(&ctx->rewriter, "builtin_join"),
                .args = mk_nodes(dst_arena, dst, tree_node)
            });
            bind_instruction(bb, join_call);
            return finish_body(bb, fn_ret(dst_arena, (Return) { .fn = NULL, .args = nodes(dst_arena, 0, NULL) }));
        }
        case PtrType_TAG: {
            const Node* pointee = old->payload.ptr_type.pointed_type;
            if (pointee->tag == FnType_TAG) {
                const Type* emulated_fn_ptr_type = uint32_type(ctx->rewriter.dst_arena);
                return emulated_fn_ptr_type;
            }
            SHADY_FALLTHROUGH
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

void generate_top_level_dispatch_fn(Context* ctx) {
    assert(ctx->config->dynamic_scheduling);
    assert(*ctx->top_dispatcher_fn);
    assert((*ctx->top_dispatcher_fn)->tag == Function_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    BodyBuilder* loop_body_builder = begin_body(dst_arena);

    const Node* next_function = gen_load(loop_body_builder, access_decl(&ctx->rewriter, "next_fn"));
    const Node* get_active_branch_fn = access_decl(&ctx->rewriter, "builtin_get_active_branch");
    assert(get_active_branch_fn->tag == FnAddr_TAG);
    get_active_branch_fn = get_active_branch_fn->payload.fn_addr.fn;
    const Node* next_mask = first(bind_instruction(loop_body_builder, call(dst_arena, (Call) { .callee = get_active_branch_fn, .args = empty(dst_arena) })));
    const Node* local_id = gen_primop_e(loop_body_builder, subgroup_local_id_op, empty(dst_arena), empty(dst_arena));
    const Node* should_run = gen_primop_e(loop_body_builder, mask_is_thread_active_op, empty(dst_arena), mk_nodes(dst_arena, next_mask, local_id));

    bool count_iterations = ctx->config->shader_diagnostics.max_top_iterations > 0;
    const Node* iterations_count_param = NULL;
    if (count_iterations)
        iterations_count_param = var(dst_arena, qualified_type(dst_arena, (QualifiedType) { .type = int32_type(dst_arena), .is_uniform = true }), "iterations");

    if (ctx->config->printf_trace.god_function) {
        const Node* sid = gen_primop_e(loop_body_builder, subgroup_id_op, empty(dst_arena), empty(dst_arena));
        if (count_iterations)
            bind_instruction(loop_body_builder, prim_op(dst_arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(dst_arena, string_lit(dst_arena, (StringLiteral) { .string = "trace: top loop, thread:%d:%d iteration=%d next_fn=%d next_mask=%lx\n" }), sid, local_id, iterations_count_param, next_function, next_mask) }));
        else
            bind_instruction(loop_body_builder, prim_op(dst_arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(dst_arena, string_lit(dst_arena, (StringLiteral) { .string = "trace: top loop, thread:%d:%d next_fn=%d next_mask=%x\n" }), sid, local_id, next_function, next_mask) }));
    }

    const Node* iteration_count_plus_one = NULL;
    if (count_iterations)
        iteration_count_plus_one = gen_primop_e(loop_body_builder, add_op, empty(dst_arena), mk_nodes(dst_arena, iterations_count_param, int32_literal(dst_arena, 1)));

    const Node* break_terminator = merge_break(dst_arena, (MergeBreak) { .args = nodes(dst_arena, 0, NULL) });
    const Node* continue_terminator = merge_continue(dst_arena, (MergeContinue) {
        .args = count_iterations ? singleton(iteration_count_plus_one) : nodes(dst_arena, 0, NULL),
    });

    if (ctx->config->shader_diagnostics.max_top_iterations > 0) {
        const Node* bail_condition = gen_primop_e(loop_body_builder, gt_op, empty(dst_arena), mk_nodes(dst_arena, iterations_count_param, int32_literal(dst_arena, ctx->config->shader_diagnostics.max_top_iterations)));
        const Node* bail_true_lam = lambda(ctx->rewriter.dst_arena, empty(dst_arena), break_terminator);
        const Node* bail_if = if_instr(dst_arena, (If) {
            .condition = bail_condition,
            .if_true = bail_true_lam,
            .if_false = NULL,
            .yield_types = empty(dst_arena)
        });
        bind_instruction(loop_body_builder, bail_if);
    }

    struct List* literals = new_list(const Node*);
    struct List* cases = new_list(const Node*);

    // Build 'zero' case (exits the program)
    BodyBuilder* zero_case_builder = begin_body(dst_arena);
    BodyBuilder* zero_if_case_builder = begin_body(dst_arena);
    if (ctx->config->printf_trace.god_function) {
        const Node* sid = gen_primop_e(loop_body_builder, subgroup_id_op, empty(dst_arena), empty(dst_arena));
        bind_instruction(zero_if_case_builder, prim_op(dst_arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(dst_arena, string_lit(dst_arena, (StringLiteral) { .string = "trace: kill thread %d:%d\n" }), sid, local_id) }));
    }
    const Node* zero_if_true_lam = lambda(ctx->rewriter.dst_arena, empty(dst_arena), finish_body(zero_if_case_builder, break_terminator));
    const Node* zero_if_instruction = if_instr(dst_arena, (If) {
        .condition = should_run,
        .if_true = zero_if_true_lam,
        .if_false = NULL,
        .yield_types = empty(dst_arena),
    });
    bind_instruction(zero_case_builder, zero_if_instruction);
    if (ctx->config->printf_trace.god_function) {
        const Node* sid = gen_primop_e(loop_body_builder, subgroup_id_op, empty(dst_arena), empty(dst_arena));
        bind_instruction(zero_case_builder, prim_op(dst_arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(dst_arena, string_lit(dst_arena, (StringLiteral) { .string = "trace: thread %d:%d escaped death!\n" }), sid, local_id) }));
    }

    const Node* zero_case_lam = lambda(ctx->rewriter.dst_arena, nodes(dst_arena, 0, NULL), finish_body(zero_case_builder, continue_terminator));
    const Node* zero_lit = uint32_literal(dst_arena, 0);
    append_list(const Node*, literals, zero_lit);
    append_list(const Node*, cases, zero_case_lam);

    Nodes old_decls = get_module_declarations(ctx->rewriter.src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (decl->tag == Function_TAG) {
            if (lookup_annotation(decl, "Leaf"))
                continue;

            const Node* fn_lit = lower_fn_addr(ctx, decl);

            BodyBuilder* if_builder = begin_body(dst_arena);
            if (ctx->config->printf_trace.god_function) {
                const Node* sid = gen_primop_e(loop_body_builder, subgroup_id_op, empty(dst_arena), empty(dst_arena));
                bind_instruction(if_builder, prim_op(dst_arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(dst_arena, string_lit(dst_arena, (StringLiteral) { .string = "trace: thread %d:%d will run fn %d with mask = %x %b\n" }), sid, local_id, fn_lit, next_mask, should_run) }));
            }
            bind_instruction(if_builder, call(dst_arena, (Call) {
                .callee = find_processed(&ctx->rewriter, decl),
                .args = nodes(dst_arena, 0, NULL)
            }));
            const Node* if_true_lam = lambda(ctx->rewriter.dst_arena, empty(dst_arena), finish_body(if_builder, yield(dst_arena, (Yield) { .args = nodes(dst_arena, 0, NULL) })));
            const Node* if_instruction = if_instr(dst_arena, (If) {
                .condition = should_run,
                .if_true = if_true_lam,
                .if_false = NULL,
                .yield_types = empty(dst_arena),
            });

            BodyBuilder* case_builder = begin_body(dst_arena);
            bind_instruction(case_builder, if_instruction);
            const Node* case_lam = lambda(ctx->rewriter.dst_arena, nodes(dst_arena, 0, NULL), finish_body(case_builder, continue_terminator));

            append_list(const Node*, literals, fn_lit);
            append_list(const Node*, cases, case_lam);
        }
    }

    const Node* default_case_lam = lambda(ctx->rewriter.dst_arena, nodes(dst_arena, 0, NULL), unreachable(dst_arena));

    bind_instruction(loop_body_builder, match_instr(dst_arena, (Match) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .inspect = next_function,
        .literals = nodes(dst_arena, entries_count_list(literals), read_list(const Node*, literals)),
        .cases = nodes(dst_arena, entries_count_list(cases), read_list(const Node*, cases)),
        .default_case = default_case_lam,
    }));

    destroy_list(literals);
    destroy_list(cases);

    const Node* loop_inside_lam = lambda(ctx->rewriter.dst_arena, count_iterations ? singleton(iterations_count_param) : nodes(dst_arena, 0, NULL), finish_body(loop_body_builder, unreachable(dst_arena)));

    const Node* the_loop = loop_instr(dst_arena, (Loop) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .initial_args = count_iterations ? singleton(int32_literal(dst_arena, 0)) : nodes(dst_arena, 0, NULL),
        .body = loop_inside_lam
    });

    BodyBuilder* dispatcher_body_builder = begin_body(dst_arena);
    bind_instruction(dispatcher_body_builder, the_loop);
    if (ctx->config->printf_trace.god_function)
        bind_instruction(dispatcher_body_builder, prim_op(dst_arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(dst_arena, string_lit(dst_arena, (StringLiteral) { .string = "trace: end of top\n" })) }));

    (*ctx->top_dispatcher_fn)->payload.fun.body = finish_body(dispatcher_body_builder, fn_ret(dst_arena, (Return) {
        .args = nodes(dst_arena, 0, NULL),
        .fn = *ctx->top_dispatcher_fn,
    }));
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void lower_tailcalls(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    struct Dict* ptrs = new_dict(const Node*, FnPtr, (HashFn) hash_node, (CmpFn) compare_node);
    IrArena* dst_arena = get_module_arena(dst);

    Node* init_fn = function(dst, nodes(dst_arena, 0, NULL), "generated_init", singleton(annotation(dst_arena, (Annotation) { .name = "Generated" })), nodes(dst_arena, 0, NULL));
    init_fn->payload.fun.body = fn_ret(dst_arena, (Return) { .fn = init_fn, .args = empty(dst_arena) });

    FnPtr next_fn_ptr = 1;

    Node* top_dispatcher_fn = NULL;

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
        .disable_lowering = false,
        .assigned_fn_ptrs = ptrs,
        .next_fn_ptr = &next_fn_ptr,

        .top_dispatcher_fn = &top_dispatcher_fn,
        .init_fn = init_fn,
    };

    rewrite_module(&ctx.rewriter);

    // Generate the top dispatcher, but only if it is used for realsies
    if (*ctx.top_dispatcher_fn)
        generate_top_level_dispatch_fn(&ctx);

    destroy_dict(ptrs);
    destroy_rewriter(&ctx.rewriter);
}
