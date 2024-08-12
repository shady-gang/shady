#include "shady/pass.h"

#include "../type.h"
#include "../transform/ir_gen_helpers.h"

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

static TmpAllocCleanupClosure create_delete_dict_closure(struct Dict* d) {
    return (TmpAllocCleanupClosure) {
        .fn = (TmpAllocCleanupFn) destroy_dict,
        .payload = d,
    };
}

static TmpAllocCleanupClosure create_cancel_body_closure(BodyBuilder* bb) {
    return (TmpAllocCleanupClosure) {
        .fn = (TmpAllocCleanupFn) cancel_body,
        .payload = bb,
    };
}

typedef struct {
    Rewriter rewriter;
    struct List* cleanup_stack;

    jmp_buf bail;

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
    Node* c = case_(a, empty(a));
    set_abstraction_body(c, unreachable(a, (Unreachable) { .mem = get_abstraction_mem(c) }));
    return c;
}

static const Node* make_selection_merge_case(IrArena* a) {
    Node* c = case_(a, empty(a));
    set_abstraction_body(c, merge_selection(a, (MergeSelection) { .args = empty(a), .mem = get_abstraction_mem(c) }));
    return c;
}

static const Node* structure(Context* ctx, const Node* abs, const Node* exit);

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
                longjmp(ctx->bail, 1);
            if (entry2->containing_control != ctx->control_stack)
                longjmp(ctx->bail, 1);
            entry2->in_loop = true;
            entry2 = entry2->parent;
        }
        prior_encounter->loop_header = true;
        return merge_continue(a, (MergeContinue) {
            .args = rewrite_nodes(r, oargs),
            .mem = mem,
        });
    } else {
        Nodes oparams = get_abstraction_params(old_target);
        assert(oparams.count == oargs.count);
        LARRAY(const Node*, nparams, oargs.count);
        Context ctx2 = *ctx;

        // Record each step of the depth-first search on a stack so we can identify loops
        DFSStackEntry dfs_entry = { .parent = ctx->dfs_stack, .old = old_target, .containing_control = ctx->control_stack };
        ctx2.dfs_stack = &dfs_entry;

        BodyBuilder* bb = begin_body_with_mem(a, mem);
        TmpAllocCleanupClosure cj1 = create_cancel_body_closure(bb);
        append_list(TmpAllocCleanupClosure, ctx->cleanup_stack, cj1);
        struct Dict* tmp_processed = clone_dict(ctx->rewriter.map);
        TmpAllocCleanupClosure cj2 = create_delete_dict_closure(tmp_processed);
        append_list(TmpAllocCleanupClosure, ctx->cleanup_stack, cj2);
        ctx2.rewriter.map = tmp_processed;
        for (size_t i = 0; i < oargs.count; i++) {
            nparams[i] = param(a, rewrite_node(&ctx->rewriter, oparams.nodes[i]->type), "arg");
            register_processed(&ctx2.rewriter, oparams.nodes[i], nparams[i]);
        }

        // We use a basic block for the exit ladder because we don't know what the ladder needs to do ahead of time
        Node* inner_exit_ladder_bb = basic_block(a, empty(a), unique_name(a, "exit_ladder_inline_me"));

        // Just jumps to the actual ladder
        Node* structured_target = case_(a, nodes(a, oargs.count, nparams));
        register_processed(&ctx2.rewriter, get_abstraction_mem(old_target), get_abstraction_mem(structured_target));
        const Node* structured = structure(&ctx2, get_abstraction_body(old_target), inner_exit_ladder_bb);
        assert(is_terminator(structured));
        set_abstraction_body(structured_target, structured);

        // forget we rewrote all that
        destroy_dict(tmp_processed);
        pop_list_impl(ctx->cleanup_stack);
        pop_list_impl(ctx->cleanup_stack);

        if (dfs_entry.loop_header) {
            // Use the structured target as the body of a loop
            gen_loop(bb, empty(a), rewrite_nodes(&ctx->rewriter, oargs), structured_target);
            // The exit ladder must exit that new loop
            set_abstraction_body(inner_exit_ladder_bb, merge_break(a, (MergeBreak) { .args = empty(a), .mem = get_abstraction_mem(inner_exit_ladder_bb) }));
            // After that we jump to the parent exit
            return finish_body(bb, jump_helper(a, exit, empty(a), bb_mem(bb)));
        } else {
            // Simply jmp to the exit once done
            set_abstraction_body(inner_exit_ladder_bb, jump_helper(a, exit, empty(a), get_abstraction_mem(inner_exit_ladder_bb)));
            // Jump into the new structured target
            return finish_body(bb, jump_helper(a, structured_target, rewrite_nodes(&ctx->rewriter, oargs), bb_mem(bb)));
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
            return handle_bb_callsite(ctx, payload, rewrite_node(r, payload.mem), exit);
        }
        // br(cond, true_bb, false_bb, args)
        // becomes
        // let(if(cond, _ => handle_bb_callsite[true_bb, args], _ => handle_bb_callsite[false_bb, args]), _ => unreachable)
        case Branch_TAG: {
            Branch payload = body->payload.branch;
            const Node* condition = rewrite_node(&ctx->rewriter, payload.condition);

            Node* true_case = case_(a, empty(a));
            set_abstraction_body(true_case, handle_bb_callsite(ctx, payload.true_jump->payload.jump, get_abstraction_mem(true_case), make_selection_merge_case(a)));

            Node* false_case = case_(a, empty(a));
            set_abstraction_body(false_case, handle_bb_callsite(ctx, payload.false_jump->payload.jump, get_abstraction_mem(false_case), make_selection_merge_case(a)));

            BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, payload.mem));
            gen_if(bb, empty(a), condition, true_case, false_case);
            return finish_body(bb, jump_helper(a, exit, empty(a), bb_mem(bb)));
        }
        case Switch_TAG: {
            Switch payload = body->payload.br_switch;
            const Node* switch_value = rewrite_node(r, payload.switch_value);

            Node* default_case = case_(a, empty(a));
            set_abstraction_body(default_case, handle_bb_callsite(ctx, payload.default_jump->payload.jump, get_abstraction_mem(default_case), make_selection_merge_case(a)));

            LARRAY(Node*, cases, body->payload.br_switch.case_jumps.count);
            for (size_t i = 0; i < body->payload.br_switch.case_jumps.count; i++) {
                cases[i] = case_(a, empty(a));
                set_abstraction_body(cases[i], handle_bb_callsite(ctx, payload.case_jumps.nodes[i]->payload.jump, get_abstraction_mem(cases[i]), make_selection_merge_case(a)));
            }

            BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, payload.mem));
            gen_match(bb, empty(a), switch_value, rewrite_nodes(&ctx->rewriter, body->payload.br_switch.case_values), nodes(a, body->payload.br_switch.case_jumps.count, (const Node**) cases), default_case);
            return finish_body(bb, jump_helper(a, exit, empty(a), bb_mem(bb)));
        }
        // let(control(body), tail)
        // var phi = undef; level = N+1; structurize[body, if (level == N+1, _ => tail(load(phi))); structured_exit_terminator]
        case Control_TAG: {
            Control payload = body->payload.control;
            const Node* old_control_case = payload.inside;
            Nodes old_control_params = get_abstraction_params(old_control_case);
            assert(old_control_params.count == 1);

            // Create N temporary variables to hold the join point arguments
            BodyBuilder* bb_prelude = begin_body_with_mem(a, rewrite_node(r, payload.mem));
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, body->payload.control.yield_types);
            LARRAY(const Node*, phis, yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                const Type* type = yield_types.nodes[i];
                assert(is_data_type(type));
                phis[i] = gen_local_alloc(bb_prelude, type);
            }

            // Create a new context to rewrite the body with
            // TODO: Bail if we try to re-enter the same control construct
            Context control_ctx = *ctx;
            ControlEntry control_entry = {
                .parent = ctx->control_stack,
                .old_token = first(old_control_params),
                .phis = phis,
                .depth = ctx->control_stack ? ctx->control_stack->depth + 1 : 1,
            };
            control_ctx.control_stack = &control_entry;

            // Set the depth for threads entering the control body
            gen_store(bb_prelude, ctx->level_ptr, int32_literal(a, control_entry.depth));

            // Start building out the tail, first it needs to dereference the phi variables to recover the arguments given to join()
            Node* tail = case_(a, empty(a));
            BodyBuilder* bb_tail = begin_body_with_mem(a, get_abstraction_mem(tail));
            LARRAY(const Node*, phi_values, yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                phi_values[i] = gen_load(bb_tail, phis[i]);
                register_processed(&ctx->rewriter, get_abstraction_params(get_structured_construct_tail(body)).nodes[i], phi_values[i]);
            }

            // Wrap the tail in a guarded if, to handle 'far' joins
            const Node* level_value = gen_load(bb_tail, ctx->level_ptr);
            const Node* guard = first(bind_instruction(bb_tail, prim_op(a, (PrimOp) { .op = eq_op, .operands = mk_nodes(a, level_value, int32_literal(a, ctx->control_stack ? ctx->control_stack->depth : 0)) })));
            Node* true_case = case_(a, empty(a));
            register_processed(r, get_abstraction_mem(get_structured_construct_tail(body)), get_abstraction_mem(true_case));
            set_abstraction_body(true_case, structure(ctx, get_abstraction_body(get_structured_construct_tail(body)), make_selection_merge_case(a)));
            gen_if(bb_tail, empty(a), guard, true_case, NULL);
            set_abstraction_body(tail, finish_body(bb_tail, jump_helper(a, exit, empty(a), bb_mem(bb_tail))));

            register_processed(r, get_abstraction_mem(old_control_case), bb_mem(bb_prelude));
            return finish_body(bb_prelude, structure(&control_ctx, get_abstraction_body(old_control_case), tail));
        }
        case Join_TAG: {
            Join payload = body->payload.join;
            ControlEntry* control = search_containing_control(ctx, body->payload.join.join_point);
            if (!control)
                longjmp(ctx->bail, 1);

            BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, payload.mem));
            gen_store(bb, ctx->level_ptr, int32_literal(a, control->depth - 1));

            Nodes args = rewrite_nodes(&ctx->rewriter, body->payload.join.args);
            for (size_t i = 0; i < args.count; i++) {
                gen_store(bb, control->phis[i], args.nodes[i]);
            }

            return finish_body(bb, jump_helper(a, exit, empty(a), bb_mem(bb)));
        }

        case Return_TAG:
        case Unreachable_TAG: return recreate_node_identity(&ctx->rewriter, body);

        case TailCall_TAG: longjmp(ctx->bail, 1);

        case If_TAG:
        case Match_TAG:
        case Loop_TAG: error("not supposed to exist in IR at this stage");
        case Terminator_MergeBreak_TAG:
        case Terminator_MergeContinue_TAG:
        case Terminator_MergeSelection_TAG: error("Only control nodes are tolerated here.")
    }
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    if (!node) return NULL;
    assert(a != node->arena);
    assert(node->arena == ctx->rewriter.src_arena);

    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    if (is_declaration(node)) {
        String name = get_declaration_name(node);
        Nodes decls = get_module_declarations(ctx->rewriter.dst_module);
        for (size_t i = 0; i < decls.count; i++) {
            if (strcmp(get_declaration_name(decls.nodes[i]), name) == 0)
                return decls.nodes[i];
        }
    }

    if (node->tag == Function_TAG) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, node);

        size_t alloc_stack_size_now = entries_count_list(ctx->cleanup_stack);

        Context ctx2 = *ctx;
        ctx2.dfs_stack = NULL;
        ctx2.control_stack = NULL;
        bool is_builtin = lookup_annotation(node, "Builtin");
        bool is_leaf = false;
        if (is_builtin || !node->payload.fun.body || lookup_annotation(node, "Structured") || setjmp(ctx2.bail)) {
            ctx2.lower = false;
            ctx2.rewriter.map = ctx->rewriter.map;
            if (node->payload.fun.body)
                set_abstraction_body(new, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
            // builtin functions are always considered leaf functions
            is_leaf = is_builtin || !node->payload.fun.body;
        } else {
            ctx2.lower = true;
            BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(new));
            TmpAllocCleanupClosure cj1 = create_cancel_body_closure(bb);
            append_list(TmpAllocCleanupClosure, ctx->cleanup_stack, cj1);
            const Node* ptr = gen_local_alloc(bb, int32_type(a));
            set_value_name(ptr, "cf_depth");
            gen_store(bb, ptr, int32_literal(a, 0));
            ctx2.level_ptr = ptr;
            ctx2.fn = new;
            struct Dict* tmp_processed = clone_dict(ctx->rewriter.map);
            TmpAllocCleanupClosure cj2 = create_delete_dict_closure(tmp_processed);
            append_list(TmpAllocCleanupClosure, ctx->cleanup_stack, cj2);
            ctx2.rewriter.map = tmp_processed;
            register_processed(&ctx2.rewriter, get_abstraction_mem(node), bb_mem(bb));
            set_abstraction_body(new, finish_body(bb, structure(&ctx2, get_abstraction_body(node), make_unreachable_case(a))));
            is_leaf = true;
            // We made it! Pop off the pending cleanup stuff and do it ourselves.
            pop_list_impl(ctx->cleanup_stack);
            pop_list_impl(ctx->cleanup_stack);
            destroy_dict(tmp_processed);
        }

        //if (is_leaf)
        //    new->payload.fun.annotations = append_nodes(arena, new->payload.fun.annotations, annotation(arena, (Annotation) { .name = "Leaf" }));

        // if we did a longjmp, we might have orphaned a few of those
        while (alloc_stack_size_now < entries_count_list(ctx->cleanup_stack)) {
            TmpAllocCleanupClosure cj = pop_last_list(TmpAllocCleanupClosure, ctx->cleanup_stack);
            cj.fn(cj.payload);
        }

        new->payload.fun.annotations = filter_out_annotation(a, new->payload.fun.annotations, "MaybeLeaf");

        return new;
    }

    if (!ctx->lower)
        return recreate_node_identity(&ctx->rewriter, node);

    // These should all be manually visited by 'structure'
    // assert(!is_terminator(node) && !is_instruction(node));

    switch (node->tag) {
        case Instruction_Call_TAG: {
            const Node* callee = node->payload.call.callee;
            if (callee->tag == FnAddr_TAG) {
                const Node* fn = rewrite_node(&ctx->rewriter, callee->payload.fn_addr.fn);
                // leave leaf calls alone
                if (lookup_annotation(fn, "Leaf")) {
                    break;
                }
            }
            // if we don't manage that, give up :(
            assert(false); // actually that should not come up.
            longjmp(ctx->bail, 1);
        }
        case BasicBlock_TAG: error("All basic blocks should be processed explicitly")
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

Module* opt_restructurize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .cleanup_stack = new_list(TmpAllocCleanupClosure),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_list(ctx.cleanup_stack);
    return dst;
}
