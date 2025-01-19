#include "shady/pass.h"
#include "shady/ir/function.h"
#include "shady/ir/annotation.h"
#include "shady/ir/mem.h"
#include "shady/ir/debug.h"

#include <setjmp.h>
#include <string.h>

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "log.h"

#pragma GCC diagnostic error "-Wswitch"

typedef struct {
    const Node* old;
    Node* new;
} TodoEntry;

typedef struct ControlEntry_ ControlEntry;
struct ControlEntry_ {
    ControlEntry* parent;
    const Node* old_token;
    const Node** phis;
    int depth;
};

typedef struct DFSStackEntry_ DFSStackEntry;
struct DFSStackEntry_ {
    DFSStackEntry* parent;
    const Node* old;

    ControlEntry* containing_control;

    bool loop_header;
    bool in_loop;
};

typedef void (*TmpAllocCleanupFn)(void*);
typedef struct {
    TmpAllocCleanupFn fn;
    void* payload;
} TmpAllocCleanupClosure;

static TmpAllocCleanupClosure create_delete_rewriter_closure(Rewriter* r) {
    return (TmpAllocCleanupClosure) {
        .fn = (TmpAllocCleanupFn) shd_destroy_rewriter,
        .payload = r,
    };
}

static TmpAllocCleanupClosure create_cancel_body_closure(BodyBuilder* bb) {
    return (TmpAllocCleanupClosure) {
        .fn = (TmpAllocCleanupFn) shd_bld_cancel,
        .payload = bb,
    };
}

typedef struct {
    Rewriter rewriter;
    struct List* cleanup_stack;

    struct {
        size_t stack_size;
        jmp_buf buf;
    } bail;

    bool lower;
    Node* fn;
    const Node* level_ptr;
    DFSStackEntry* dfs_stack;
    ControlEntry* control_stack;
} Context;

static DFSStackEntry* encountered_before(Context* ctx, const Node* bb, size_t* path_len) {
    DFSStackEntry* entry = ctx->dfs_stack;
    if (path_len) *path_len = 0;
    while (entry != NULL) {
        if (entry->old == bb)
            return entry;
        entry = entry->parent;
        if (path_len) (*path_len)++;
    }
    return entry;
}

static const Node* make_unreachable_case(IrArena* a) {
    Node* c = case_(a, shd_empty(a));
    shd_set_abstraction_body(c, unreachable(a, (Unreachable) { .mem = shd_get_abstraction_mem(c) }));
    return c;
}

static const Node* make_selection_merge_case(IrArena* a) {
    Node* c = case_(a, shd_empty(a));
    shd_set_abstraction_body(c, merge_selection(a, (MergeSelection) { .args = shd_empty(a), .mem = shd_get_abstraction_mem(c) }));
    return c;
}

static const Node* structure(Context* ctx, const Node* abs, const Node* exit);

static void bail(Context* ctx) {
    // if we do a longjmp, we must cleanup before we tear down the stack
    // (some of the data just lives there, specifically the rewriters)
    while (shd_list_count(ctx->cleanup_stack) > ctx->bail.stack_size) {
        TmpAllocCleanupClosure cj = shd_list_pop(TmpAllocCleanupClosure, ctx->cleanup_stack);
        cj.fn(cj.payload);
    }
    
    longjmp(ctx->bail.buf, 1);
}

static const Node* handle_bb_callsite(Context* ctx, Jump jump, const Node* mem, const Node* exit) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    const Node* old_target = jump.target;
    Nodes oargs = jump.args;

    size_t path_len;
    DFSStackEntry* prior_encounter = encountered_before(ctx, old_target, &path_len);
    if (prior_encounter) {
        // Create path
        LARRAY(const Node*, path, path_len);
        DFSStackEntry* entry2 = ctx->dfs_stack->parent;
        for (size_t i = 0; i < path_len; i++) {
            assert(entry2);
            path[path_len - 1 - i] = entry2->old;
            if (entry2->in_loop)
                bail(ctx);
            if (entry2->containing_control != ctx->control_stack)
                bail(ctx);
            entry2->in_loop = true;
            entry2 = entry2->parent;
        }
        prior_encounter->loop_header = true;
        return merge_continue(a, (MergeContinue) {
            .args = shd_rewrite_nodes(r, oargs),
            .mem = mem,
        });
    } else {
        Nodes oparams = get_abstraction_params(old_target);
        assert(oparams.count == oargs.count);
        LARRAY(const Node*, nparams, oargs.count);
        Context children_ctx = *ctx;

        // Record each step of the depth-first search on a stack so we can identify loops
        DFSStackEntry dfs_entry = { .parent = ctx->dfs_stack, .old = old_target, .containing_control = ctx->control_stack };
        children_ctx.dfs_stack = &dfs_entry;

        BodyBuilder* bb = shd_bld_begin(a, mem);
        TmpAllocCleanupClosure cj1 = create_cancel_body_closure(bb);
        shd_list_append(TmpAllocCleanupClosure, ctx->cleanup_stack, cj1);
        children_ctx.rewriter = shd_create_children_rewriter(&ctx->rewriter);
        TmpAllocCleanupClosure cj2 = create_delete_rewriter_closure(&children_ctx.rewriter);
        shd_list_append(TmpAllocCleanupClosure, ctx->cleanup_stack, cj2);
        for (size_t i = 0; i < oargs.count; i++) {
            nparams[i] = param_helper(a, shd_rewrite_node(&ctx->rewriter, oparams.nodes[i]->type));
            String name = shd_get_node_name_unsafe(oparams.nodes[i]);
            if (name)
                shd_set_debug_name(nparams[i], name);
            shd_register_processed(&children_ctx.rewriter, oparams.nodes[i], nparams[i]);
        }

        // We use a basic block for the exit ladder because we don't know what the ladder needs to do ahead of time
        Node* inner_exit_ladder_bb = basic_block_helper(a, shd_empty(a), shd_make_unique_name(a, "exit_ladder_inline_me"));

        // Just jumps to the actual ladder
        Node* structured_target = case_(a, shd_nodes(a, oargs.count, nparams));
        shd_register_processed(&children_ctx.rewriter, shd_get_abstraction_mem(old_target), shd_get_abstraction_mem(structured_target));
        const Node* structured = structure(&children_ctx, get_abstraction_body(old_target), inner_exit_ladder_bb);
        assert(is_terminator(structured));
        shd_set_abstraction_body(structured_target, structured);

        // forget we rewrote all that
        shd_destroy_rewriter(&children_ctx.rewriter);
        shd_list_pop_impl(ctx->cleanup_stack);
        shd_list_pop_impl(ctx->cleanup_stack);

        if (dfs_entry.loop_header) {
            // Use the structured target as the body of a loop
            shd_bld_loop(bb, shd_empty(a), shd_rewrite_nodes(&ctx->rewriter, oargs), structured_target);
            // The exit ladder must exit that new loop
            shd_set_abstraction_body(inner_exit_ladder_bb, merge_break(a, (MergeBreak) { .args = shd_empty(a), .mem = shd_get_abstraction_mem(inner_exit_ladder_bb) }));
            // After that we jump to the parent exit
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), exit, shd_empty(a)));
        } else {
            // Simply jmp to the exit once done
            shd_set_abstraction_body(inner_exit_ladder_bb, jump_helper(a, shd_get_abstraction_mem(inner_exit_ladder_bb), exit,
                                                                       shd_empty(a)));
            // Jump into the new structured target
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), structured_target, shd_rewrite_nodes(&ctx->rewriter, oargs)));
        }
    }
}

static ControlEntry* search_containing_control(Context* ctx, const Node* old_token) {
    ControlEntry* entry = ctx->control_stack;
    assert(entry);
    while (entry != NULL) {
        if (entry->old_token == old_token)
            return entry;
        entry = entry->parent;
    }
    return entry;
}

static const Node* structure(Context* ctx, const Node* body, const Node* exit) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    assert(body && is_terminator(body));
    switch (is_terminator(body)) {
        case NotATerminator:
        case Jump_TAG: {
            Jump payload = body->payload.jump;
            return handle_bb_callsite(ctx, payload, shd_rewrite_node(r, payload.mem), exit);
        }
        // br(cond, true_bb, false_bb, args)
        // becomes
        // let(if(cond, _ => handle_bb_callsite[true_bb, args], _ => handle_bb_callsite[false_bb, args]), _ => unreachable)
        case Branch_TAG: {
            Branch payload = body->payload.branch;
            const Node* condition = shd_rewrite_node(&ctx->rewriter, payload.condition);
            shd_rewrite_node(r, payload.mem);

            Node* true_case = case_(a, shd_empty(a));
            shd_set_abstraction_body(true_case, handle_bb_callsite(ctx, payload.true_jump->payload.jump, shd_get_abstraction_mem(true_case), make_selection_merge_case(a)));

            Node* false_case = case_(a, shd_empty(a));
            shd_set_abstraction_body(false_case, handle_bb_callsite(ctx, payload.false_jump->payload.jump, shd_get_abstraction_mem(false_case), make_selection_merge_case(a)));

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_if(bb, shd_empty(a), condition, true_case, false_case);
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), exit, shd_empty(a)));
        }
        case Switch_TAG: {
            Switch payload = body->payload.br_switch;
            const Node* switch_value = shd_rewrite_node(r, payload.switch_value);
            shd_rewrite_node(r, payload.mem);

            Node* default_case = case_(a, shd_empty(a));
            shd_set_abstraction_body(default_case, handle_bb_callsite(ctx, payload.default_jump->payload.jump, shd_get_abstraction_mem(default_case), make_selection_merge_case(a)));

            LARRAY(Node*, cases, body->payload.br_switch.case_jumps.count);
            for (size_t i = 0; i < body->payload.br_switch.case_jumps.count; i++) {
                cases[i] = case_(a, shd_empty(a));
                shd_set_abstraction_body(cases[i], handle_bb_callsite(ctx, payload.case_jumps.nodes[i]->payload.jump, shd_get_abstraction_mem(cases[i]), make_selection_merge_case(a)));
            }

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_match(bb, shd_empty(a), switch_value, shd_rewrite_nodes(&ctx->rewriter, body->payload.br_switch.case_values), shd_nodes(a, body->payload.br_switch.case_jumps.count, (const Node**) cases), default_case);
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), exit, shd_empty(a)));
        }
        // let(control(body), tail)
        // var phi = undef; level = N+1; structurize[body, if (level == N+1, _ => tail(load(phi))); structured_exit_terminator]
        case Control_TAG: {
            Control payload = body->payload.control;
            const Node* old_control_case = payload.inside;
            Nodes old_control_params = get_abstraction_params(old_control_case);
            assert(old_control_params.count == 1);

            // Create N temporary variables to hold the join point arguments
            BodyBuilder* bb_prelude = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes yield_types = shd_rewrite_nodes(&ctx->rewriter, body->payload.control.yield_types);
            LARRAY(const Node*, phis, yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                const Type* type = yield_types.nodes[i];
                assert(shd_is_data_type(type));
                phis[i] = shd_bld_local_alloc(bb_prelude, type);
            }

            // Create a new context to rewrite the body with
            // TODO: Bail if we try to re-enter the same control construct
            Context control_ctx = *ctx;
            ControlEntry control_entry = {
                .parent = ctx->control_stack,
                .old_token = shd_first(old_control_params),
                .phis = phis,
                .depth = ctx->control_stack ? ctx->control_stack->depth + 1 : 1,
            };
            control_ctx.control_stack = &control_entry;

            // Set the depth for threads entering the control body
            shd_bld_store(bb_prelude, ctx->level_ptr, shd_int32_literal(a, control_entry.depth));

            // Start building out the tail, first it needs to dereference the phi variables to recover the arguments given to join()
            Node* tail = case_(a, shd_empty(a));
            BodyBuilder* bb_tail = shd_bld_begin(a, shd_get_abstraction_mem(tail));
            LARRAY(const Node*, phi_values, yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                phi_values[i] = shd_bld_load(bb_tail, phis[i]);
                shd_register_processed(&ctx->rewriter, get_abstraction_params(get_structured_construct_tail(body)).nodes[i], phi_values[i]);
            }

            // Wrap the tail in a guarded if, to handle 'far' joins
            const Node* level_value = shd_bld_load(bb_tail, ctx->level_ptr);
            const Node* guard = prim_op(a, (PrimOp) { .op = eq_op, .operands = mk_nodes(a, level_value, shd_int32_literal(a, ctx->control_stack ? ctx->control_stack->depth : 0)) });
            Node* true_case = case_(a, shd_empty(a));
            shd_register_processed(r, shd_get_abstraction_mem(get_structured_construct_tail(body)), shd_get_abstraction_mem(true_case));
            shd_set_abstraction_body(true_case, structure(ctx, get_abstraction_body(get_structured_construct_tail(body)), make_selection_merge_case(a)));
            shd_bld_if(bb_tail, shd_empty(a), guard, true_case, NULL);
            shd_set_abstraction_body(tail, shd_bld_finish(bb_tail, jump_helper(a, shd_bld_mem(bb_tail), exit, shd_empty(a))));

            shd_register_processed(r, shd_get_abstraction_mem(old_control_case), shd_bld_mem(bb_prelude));
            return shd_bld_finish(bb_prelude, structure(&control_ctx, get_abstraction_body(old_control_case), tail));
        }
        case Join_TAG: {
            Join payload = body->payload.join;
            ControlEntry* control = search_containing_control(ctx, body->payload.join.join_point);
            if (!control)
                bail(ctx);

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            shd_bld_store(bb, ctx->level_ptr, shd_int32_literal(a, control->depth - 1));

            Nodes args = shd_rewrite_nodes(&ctx->rewriter, body->payload.join.args);
            for (size_t i = 0; i < args.count; i++) {
                shd_bld_store(bb, control->phis[i], args.nodes[i]);
            }

            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), exit, shd_empty(a)));
        }

        case Return_TAG:
        case Unreachable_TAG: return shd_recreate_node(&ctx->rewriter, body);

        case TailCall_TAG: bail(ctx);

        case If_TAG:
        case Match_TAG:
        case Loop_TAG: shd_error("not supposed to exist in IR at this stage");
        case Terminator_MergeBreak_TAG:
        case Terminator_MergeContinue_TAG:
        case Terminator_MergeSelection_TAG: shd_error("Only control nodes are tolerated here.")
    }
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    assert(a != node->arena);
    assert(node->arena == ctx->rewriter.src_arena);

    switch (node->tag) {
        case Function_TAG: {
            Node* new = shd_recreate_node_head(&ctx->rewriter, node);

            Context fn_ctx = *ctx;
            fn_ctx.dfs_stack = NULL;
            fn_ctx.control_stack = NULL;
            fn_ctx.bail.stack_size = shd_list_count(ctx->cleanup_stack);
            if (!node->payload.fun.body || shd_lookup_annotation(node, "Structured") || setjmp(fn_ctx.bail.buf)) {
                fn_ctx.lower = false;
                // make sure to reset this
                fn_ctx.rewriter = ctx->rewriter;
                if (node->payload.fun.body)
                    shd_set_abstraction_body(new, shd_rewrite_node(&fn_ctx.rewriter, node->payload.fun.body));
            } else {
                fn_ctx.lower = true;
                BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new));
                TmpAllocCleanupClosure cj1 = create_cancel_body_closure(bb);
                shd_list_append(TmpAllocCleanupClosure, ctx->cleanup_stack, cj1);
                const Node* ptr = shd_bld_local_alloc(bb, shd_int32_type(a));
                shd_set_debug_name(ptr, "cf_depth");
                shd_bld_store(bb, ptr, shd_int32_literal(a, 0));
                fn_ctx.level_ptr = ptr;
                fn_ctx.fn = new;
                fn_ctx.rewriter = shd_create_children_rewriter(&ctx->rewriter);
                TmpAllocCleanupClosure cj2 = create_delete_rewriter_closure(&fn_ctx.rewriter);
                shd_list_append(TmpAllocCleanupClosure, ctx->cleanup_stack, cj2);
                shd_register_processed(&fn_ctx.rewriter, shd_get_abstraction_mem(node), shd_bld_mem(bb));
                shd_set_abstraction_body(new, shd_bld_finish(bb, structure(&fn_ctx, get_abstraction_body(node), make_unreachable_case(a))));
                // We made it! Pop off the pending cleanup stuff and do it ourselves.
                shd_list_pop_impl(ctx->cleanup_stack);
                shd_list_pop_impl(ctx->cleanup_stack);
                shd_destroy_rewriter(&fn_ctx.rewriter);
            }

            return new;
        }
        default: break;
    }

    if (!ctx->lower)
        return shd_recreate_node(&ctx->rewriter, node);

    // These should all be manually visited by 'structure'
    // assert(!is_terminator(node) && !is_instruction(node));

    switch (node->tag) {
        case IndirectCall_TAG: {
            const Node* callee = node->payload.indirect_call.callee;
            if (callee->tag == FnAddr_TAG) {
                const Node* fn = shd_rewrite_node(&ctx->rewriter, callee->payload.fn_addr.fn);
                // leave leaf calls alone
                if (shd_lookup_annotation(fn, "Leaf")) {
                    break;
                }
            }
            // if we don't manage that, give up :(
            assert(false); // actually that should not come up.
            bail(ctx);
        }
        case BasicBlock_TAG: shd_error("All basic blocks should be processed explicitly")
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_restructurize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .cleanup_stack = shd_new_list(TmpAllocCleanupClosure),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_list(ctx.cleanup_stack);
    return dst;
}
