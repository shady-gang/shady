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

typedef uint64_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    const TargetConfig* target;
    const CompilerConfig* config;
    bool disable_lowering;
    struct Dict* assigned_fn_ptrs;
    FnPtr* next_fn_ptr;

    CFG* cfg;
    const UsesMap* uses;

    Node** top_dispatcher_fn;
} Context;

static const Node* process(Context* ctx, const Node* old);

static const Node* fn_ptr_as_value(Context* ctx, FnPtr ptr) {
    IrArena* a = ctx->rewriter.dst_arena;
    return int_literal(a, (IntLiteral) {
        .is_signed = false,
        .width = ctx->target->memory.fn_ptr_size,
        .value = ptr
    });
}

static FnPtr get_fn_ptr(Context* ctx, const Node* the_function) {
    assert(the_function->arena == ctx->rewriter.src_arena);
    assert(the_function->tag == Function_TAG);

    FnPtr* found = shd_dict_find_value(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function);
    if (found) return *found;

    FnPtr ptr = (*ctx->next_fn_ptr)++;
    bool r = shd_dict_insert(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function, ptr);
    assert(r);
    return ptr;
}

static const Node* lower_fn_addr(Context* ctx, const Node* the_function) {
    return fn_ptr_as_value(ctx, get_fn_ptr(ctx, the_function));
}

static const Node* get_top_dispatcher_fn(Context* ctx) {
    IrArena* a = ctx->rewriter.dst_arena;
    if (!*ctx->top_dispatcher_fn) {
        *ctx->top_dispatcher_fn = function_helper(ctx->rewriter.dst_module, shd_nodes(a, 0, NULL), shd_nodes(a, 0, NULL));
        shd_set_debug_name(*ctx->top_dispatcher_fn, "top_dispatcher");
        // shd_add_annotation_named(*ctx->top_dispatcher_fn, "Generated");
        shd_add_annotation_named(*ctx->top_dispatcher_fn, "Leaf");
    }
    return *ctx->top_dispatcher_fn;
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

            // Leave leaf-calls alone :)
            ctx2.disable_lowering = shd_lookup_annotation(old, "Leaf") || !old->payload.fun.body;
            if (ctx2.disable_lowering) {
                Node* fun = shd_recreate_node_head(&ctx2.rewriter, old);
                if (old->payload.fun.body) {
                    shd_set_abstraction_body(fun, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old)));
                }

                shd_destroy_uses_map(ctx2.uses);
                shd_destroy_cfg(ctx2.cfg);
                return fun;
            }

            assert(ctx->config->dynamic_scheduling && "Dynamic scheduling is disabled, but we encountered a non-leaf function");
            Node* fun = shd_recreate_node_head(r, old);
            shd_set_abstraction_body(fun, shd_rewrite_node(&ctx2.rewriter, get_abstraction_body(old)));
            shd_destroy_uses_map(ctx2.uses);
            shd_destroy_cfg(ctx2.cfg);
            return fun;
        }
        case FnAddr_TAG:
            if (ctx->target->capabilities.native_tailcalls)
                break;
            return lower_fn_addr(ctx, old->payload.fn_addr.fn);
        case ExtInstr_TAG: {
            ExtInstr payload = old->payload.ext_instr;
            if (strcmp(payload.set, "shady.internal") == 0 && payload.opcode == ShadyOpDispatcherEnterFn) {
                return call_helper(a, shd_rewrite_node(r, payload.mem), get_top_dispatcher_fn(ctx), shd_empty(a));
            }
            break;
        }
        case ExtTerminator_TAG: {
            ExtTerminator payload = old->payload.ext_terminator;
            if (strcmp(payload.set, "shady.internal") == 0 && payload.opcode == ShadyOpDispatcherContinue) {
                return fn_ret_helper(a, shd_rewrite_node(r, payload.mem), shd_empty(a));
            }
            break;
        }
        case IndirectTailCall_TAG: {
            IndirectTailCall payload = old->payload.indirect_tail_call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_stack_push_values(bb, shd_rewrite_nodes(&ctx->rewriter, payload.args));
            const Node* target = shd_rewrite_node(&ctx->rewriter, payload.callee);
            target = shd_bld_bitcast(bb, int_type_helper(a, ctx->target->memory.fn_ptr_size, false), target);

            if (ctx->target->capabilities.native_tailcalls)
                break;

            // fast-path
            assert(shd_get_qualified_type_scope(payload.callee->type) <= ShdScopeSubgroup && "only uniform tailcalls are allowed here");
            //shd_bld_store(bb, shd_find_or_process_decl(r, "next_fn"), target);
            shd_bld_call(bb, shd_find_or_process_decl(&ctx->rewriter, "builtin_jump"), shd_singleton(target));
            return shd_bld_finish(bb, fn_ret(a, (Return) { .args = shd_empty(a), .mem = shd_bld_mem(bb) }));
        }
        case PtrType_TAG: {
            const Node* pointee = old->payload.ptr_type.pointed_type;
            if (pointee->tag == FnType_TAG && !ctx->target->capabilities.native_tailcalls)
                return int_type_helper(a, ctx->target->memory.fn_ptr_size, false);
            break;
        }
        default: break;
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
    const Node* builtin_get_active_threads_mask_fn = shd_find_or_process_decl(r, "builtin_get_active_threads_mask");
    const Node* next_mask = shd_first(shd_bld_call(loop_body_builder, builtin_get_active_threads_mask_fn, shd_empty(a)));
    const Node* local_id = shd_bld_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupLocalInvocationId);
    const Node* should_run = prim_op_helper(a, mask_is_thread_active_op, mk_nodes(a, next_mask, local_id));

    const Node* sid = shd_bld_builtin_load(ctx->rewriter.dst_module, loop_body_builder, BuiltinSubgroupId);
    if (ctx->config->printf_trace.top_function) {
        const Node* resume_at = shd_bld_load(loop_body_builder, lea_helper(a, shd_find_or_process_decl(r, "resume_at"), shd_uint32_literal(a, 0), mk_nodes(a, local_id)));
        if (count_iterations)
            shd_bld_debug_printf(loop_body_builder, "trace: top loop, thread:%d:%d iteration=%d next_fn=%lu next_mask=%lx resume_at=%lu\n", mk_nodes(a, sid, local_id, iterations_count_param, next_function, next_mask, resume_at));
        else
            shd_bld_debug_printf(loop_body_builder, "trace: top loop, thread:%d:%d next_fn=%lu next_mask=%lx resume_at=%lu\n", mk_nodes(a, sid, local_id, next_function, next_mask, resume_at));
    }

    const Node* iteration_count_plus_one = NULL;
    if (count_iterations)
        iteration_count_plus_one = prim_op_helper(a, add_op, mk_nodes(a, iterations_count_param, shd_int32_literal(a, 1)));

    if (ctx->config->shader_diagnostics.max_top_iterations > 0) {
        begin_control_t c = shd_bld_begin_control(loop_body_builder, shd_empty(a));
        const Node* bail_condition = prim_op_helper(a, gt_op, mk_nodes(a, iterations_count_param, shd_int32_literal(a, ctx->config->shader_diagnostics.max_top_iterations)));
        Node* bail_case = basic_block_helper(a, shd_empty(a));
        const Node* break_terminator = join(a, (Join) { .args = shd_empty(a), .join_point = l.break_jp, .mem = shd_get_abstraction_mem(bail_case) });
        shd_set_abstraction_body(bail_case, break_terminator);
        Node* proceed_case = basic_block_helper(a, shd_empty(a));
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
    Node* zero_case_lam = basic_block_helper(a, shd_nodes(a, 0, NULL));
    Node* zero_if_true_lam = basic_block_helper(a, shd_empty(a));
    BodyBuilder* zero_if_case_builder = shd_bld_begin(a, shd_get_abstraction_mem(zero_if_true_lam));
    if (ctx->config->printf_trace.top_function) {
        shd_bld_debug_printf(zero_if_case_builder, "trace: kill thread %d:%d\n", mk_nodes(a, sid, local_id));
    }
    shd_set_abstraction_body(zero_if_true_lam, shd_bld_join(zero_if_case_builder, l.break_jp, shd_empty(a)));

    Node* zero_if_false = basic_block_helper(a, shd_empty(a));
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

    const Node* zero_lit = int_literal_helper(a, ctx->target->memory.fn_ptr_size, false, 0);
    shd_list_append(const Node*, literals, zero_lit);
    const Node* zero_jump = jump_helper(a, shd_bld_mem(loop_body_builder), zero_case_lam, shd_empty(a));
    shd_list_append(const Node*, jumps, zero_jump);

    Nodes ofunctions = shd_module_collect_reachable_functions(ctx->rewriter.src_module);
    for (size_t i = 0; i < ofunctions.count; i++) {
        const Node* ofunction = ofunctions.nodes[i];
        if (shd_lookup_annotation(ofunction, "Leaf"))
            continue;

        FnPtr fn_ptr = get_fn_ptr(ctx, ofunction);
        const Node* fn_lit = shd_uint32_literal(a, fn_ptr);

        Node* if_true_case = basic_block_helper(a, shd_empty(a));
        BodyBuilder* if_builder = shd_bld_begin(a, shd_get_abstraction_mem(if_true_case));
        if (ctx->config->printf_trace.top_function) {
            shd_bld_debug_printf(if_builder, "trace: thread %d:%d will run fn %u with mask = %lx\n", mk_nodes(a, sid, local_id, fn_lit, next_mask));
        }
        Nodes oparams = get_abstraction_params(ofunction);
        LARRAY(const Node*, nargs, oparams.count);
        for (size_t j = 0; j < oparams.count; j++) {
            const Node* old_param = oparams.nodes[j];
            const Type* arg_type = shd_rewrite_node(r, shd_get_unqualified_type(old_param->type));
            const Node* popped = shd_bld_stack_pop_value(if_builder, arg_type);
            // TODO use the uniform stack instead ? or no ?
            popped = scope_cast_helper(a, shd_get_qualified_type_scope(old_param->type), popped);
            nargs[j] = popped;
        }
        shd_bld_call(if_builder, shd_rewrite_node(r, ofunction), shd_nodes(a, oparams.count, nargs));
        if (ctx->config->printf_trace.top_function) {
            const Node* resume_at = shd_bld_load(if_builder, lea_helper(a, shd_find_or_process_decl(r, "resume_at"), shd_uint32_literal(a, 0),mk_nodes(a, local_id)));
            String ptrn = NULL;
            //if (shd_get_node_name_unsafe(ofunction))
            //    ptrn = shd_fmt_string_irarena(a, "trace: ran %d(%s)%s\n", fn_ptr, shd_get_node_name_unsafe(ofunction), ", thread:%d:%d next_fn=%lu next_mask=%lx resume_at=%lu");
            //else
                ptrn = shd_fmt_string_irarena(a, "trace: ran %d%s\n", fn_ptr, ", thread:%d:%d next_fn=%lu next_mask=%lx resume_at=%lu");
            const Node* next_function2 = shd_bld_load(if_builder, shd_find_or_process_decl(r, "next_fn"));
            shd_bld_debug_printf(if_builder, ptrn, mk_nodes(a, sid, local_id, next_function2, next_mask, resume_at));
        }
        shd_set_abstraction_body(if_true_case, shd_bld_join(if_builder, l.continue_jp, count_iterations ? shd_singleton(iteration_count_plus_one) : shd_empty(a)));

        Node* if_false = basic_block_helper(a, shd_empty(a));
        shd_set_abstraction_body(if_false, join(a, (Join) {
            .mem = shd_get_abstraction_mem(if_false),
            .join_point = l.continue_jp,
            .args = count_iterations ? shd_singleton(iteration_count_plus_one) : shd_empty(a)
        }));

        Node* fn_case = basic_block_helper(a, shd_nodes(a, 0, NULL));
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

    Node* default_case = basic_block_helper(a, shd_nodes(a, 0, NULL));
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

Module* shd_pass_lower_tailcalls(const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.target.capabilities.native_tailcalls = false;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    struct Dict* ptrs = shd_new_dict(const Node*, FnPtr, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);

    FnPtr next_fn_ptr = 1;

    Node* top_dispatcher_fn = NULL;

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .target = &aconfig.target,
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
