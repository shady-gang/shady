#include "pass.h"

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

static const Node* structure(Context* ctx, const Node* abs, const Node* exit_ladder);

static const Node* handle_bb_callsite(Context* ctx, const Node* caller, const Node* j, const Node* exit_ladder) {
    assert(j->tag == Jump_TAG);
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* dst = j->payload.jump.target;
    Nodes oargs = j->payload.jump.args;

    size_t path_len;
    DFSStackEntry* prior_encounter = encountered_before(ctx, dst, &path_len);
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
            .args = rewrite_nodes(&ctx->rewriter, oargs)
        });
    } else {
        Nodes oparams = get_abstraction_params(dst);
        assert(oparams.count == oargs.count);
        LARRAY(const Node*, nparams, oargs.count);
        Context ctx2 = *ctx;
        
        // Record each step of the depth-first search on a stack so we can identify loops
        DFSStackEntry dfs_entry = { .parent = ctx->dfs_stack, .old = dst, .containing_control = ctx->control_stack };
        ctx2.dfs_stack = &dfs_entry;

        BodyBuilder* bb = begin_body(a);
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
        // opt_simplify_cf will later inline this
        Node* inner_exit_ladder_bb = basic_block(a, empty(a), unique_name(a, "exit_ladder_inline_me"));

        // Just jumps to the actual ladder
        const Node* exit_ladder_trampoline = case_(a, empty(a), jump(a, (Jump) {.target = inner_exit_ladder_bb, .args = empty(a)}));

        const Node* structured = structure(&ctx2, dst, let(a, quote_helper(a, empty(a)), empty(a), exit_ladder_trampoline));
        assert(is_terminator(structured));
        // forget we rewrote all that
        destroy_dict(tmp_processed);
        pop_list_impl(ctx->cleanup_stack);
        pop_list_impl(ctx->cleanup_stack);

        if (dfs_entry.loop_header) {
            const Node* body = case_(a, nodes(a, oargs.count, nparams), structured);
            gen_loop(bb, empty(a), rewrite_nodes(&ctx->rewriter, oargs), body);
            // we decide 'late' what the exit ladder should be
            inner_exit_ladder_bb->payload.basic_block.body = merge_break(a, (MergeBreak) { .args = empty(a) });
            return finish_body(bb, exit_ladder);
        } else {
            Node* bb2 = basic_block(a, nodes(a, oargs.count, nparams), NULL);
            bb2->payload.basic_block.body = structured;
            //bind_variables(bb, nodes(a, oargs.count, nparams), rewrite_nodes(&ctx->rewriter, oargs));
            inner_exit_ladder_bb->payload.basic_block.body = exit_ladder;
            //return finish_body(bb, structured);
            return finish_body(bb, jump_helper(a, bb2, rewrite_nodes(&ctx->rewriter, oargs)));
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

static const Node* rebuild_let(Context* ctx, const Node* old_let, const Node* new_instruction, const Node* exit_ladder) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* old_tail = get_let_tail(old_let);
    Nodes otail_params = get_abstraction_params(old_tail);
    assert(otail_params.count == 0);

    Nodes ovars = old_let->payload.let.variables;
    Nodes nvars = recreate_vars(a, ovars, new_instruction);
    register_processed_list(&ctx->rewriter, ovars, nvars);

    const Node* structured_lam = case_(a, empty(a), structure(ctx, old_tail, exit_ladder));
    return let(a, new_instruction, nvars, structured_lam);
}

static const Node* structure(Context* ctx, const Node* abs, const Node* exit_ladder) {
    IrArena* a = ctx->rewriter.dst_arena;

    const Node* body = get_abstraction_body(abs);
    assert(body);
    switch (is_terminator(body)) {
        case NotATerminator:
        case Let_TAG: {
            const Node* old_tail = get_let_tail(body);
            Nodes ovars = body->payload.let.variables;
            //Nodes otail_params = get_abstraction_params(old_tail);

            const Node* old_instr = get_let_instruction(body);
            switch (is_instruction(old_instr)) {
                case NotAnInstruction: assert(false);
                case Instruction_Loop_TAG:
                case Instruction_Match_TAG: error("not supposed to exist in IR at this stage");
                case Instruction_Block_TAG: error("Should be eliminated by the compiler");
                case Instruction_Call_TAG: {
                    const Node* callee = old_instr->payload.call.callee;
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
                // let(control(body), tail)
                // var phi = undef; level = N+1; structurize[body, if (level == N+1, _ => tail(load(phi))); structured_exit_terminator]
                case Instruction_Control_TAG: {
                    const Node* old_control_body = old_instr->payload.control.inside;
                    assert(old_control_body->tag == Case_TAG);
                    Nodes old_control_params = get_abstraction_params(old_control_body);
                    assert(old_control_params.count == 1);

                    // Create N temporary variables to hold the join point arguments
                    BodyBuilder* bb_outer = begin_body(a);
                    Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instr->payload.control.yield_types);
                    LARRAY(const Node*, phis, yield_types.count);
                    for (size_t i = 0; i < yield_types.count; i++) {
                        const Type* type = yield_types.nodes[i];
                        assert(is_data_type(type));
                        phis[i] = gen_local_alloc(bb_outer, type);
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
                    gen_store(bb_outer, ctx->level_ptr, int32_literal(a, control_entry.depth));

                    // Start building out the tail, first it needs to dereference the phi variables to recover the arguments given to join()
                    BodyBuilder* bb2 = begin_body(a);
                    LARRAY(const Node*, phi_values, yield_types.count);
                    for (size_t i = 0; i < yield_types.count; i++) {
                        phi_values[i] = gen_load(bb2, phis[i]);
                        register_processed(&ctx->rewriter, ovars.nodes[i], phi_values[i]);
                    }

                    // Wrap the tail in a guarded if, to handle 'far' joins
                    const Node* level_value = gen_load(bb2, ctx->level_ptr);
                    const Node* guard = first(bind_instruction(bb2, prim_op(a, (PrimOp) { .op = eq_op, .operands = mk_nodes(a, level_value, int32_literal(a, ctx->control_stack ? ctx->control_stack->depth : 0)) })));
                    const Node* true_body = structure(ctx, old_tail, merge_selection(a, (MergeSelection) { .args = empty(a) }));
                    const Node* if_true_lam = case_(a, empty(a), true_body);
                    gen_if(bb2, empty(a), guard, if_true_lam, NULL);

                    const Node* tail_lambda = case_(a, empty(a), finish_body(bb2, exit_ladder));
                    return finish_body(bb_outer, structure(&control_ctx, old_control_body, let(a, quote_helper(a, empty(a)), empty(a), tail_lambda)));
                }
                default: {
                    break;
                }
            }
            return rebuild_let(ctx, body, recreate_node_identity(&ctx->rewriter, old_instr), exit_ladder);
        }
        case Jump_TAG: {
            return handle_bb_callsite(ctx, abs, body, exit_ladder);
        }
        // br(cond, true_bb, false_bb, args)
        // becomes
        // let(if(cond, _ => handle_bb_callsite[true_bb, args], _ => handle_bb_callsite[false_bb, args]), _ => unreachable)
        case Branch_TAG: {
            const Node* condition = rewrite_node(&ctx->rewriter, body->payload.branch.condition);

            const Node* true_body = handle_bb_callsite(ctx, abs, body->payload.branch.true_jump, merge_selection(a, (MergeSelection) { .args = empty(a) }));
            const Node* if_true_lam = case_(a, empty(a), true_body);

            const Node* false_body = handle_bb_callsite(ctx, abs, body->payload.branch.false_jump, merge_selection(a, (MergeSelection) { .args = empty(a) }));
            const Node* if_false_lam = case_(a, empty(a), false_body);

            BodyBuilder* bb = begin_body(a);
            gen_if(bb, empty(a), condition, if_true_lam, if_false_lam);
            return finish_body(bb, exit_ladder);
        }
        case Switch_TAG: {
            const Node* switch_value = rewrite_node(&ctx->rewriter, body->payload.br_switch.switch_value);

            const Node* default_body = handle_bb_callsite(ctx, abs, body->payload.br_switch.default_jump, merge_selection(a, (MergeSelection) { .args = empty(a) }));
            const Node* default_case = case_(a, empty(a), default_body);

            LARRAY(const Node*, cases, body->payload.br_switch.case_jumps.count);
            for (size_t i = 0; i < body->payload.br_switch.case_jumps.count; i++) {
                cases[i] = case_(a, empty(a), handle_bb_callsite(ctx, abs, body->payload.br_switch.case_jumps.nodes[i], merge_selection(a, (MergeSelection) {.args = empty(a)})));
            }

            BodyBuilder* bb = begin_body(a);
            gen_match(bb, empty(a), switch_value, rewrite_nodes(&ctx->rewriter, body->payload.br_switch.case_values), nodes(a, body->payload.br_switch.case_jumps.count, cases), default_case);
            finish_body(bb, exit_ladder);
        }
        case Join_TAG: {
            ControlEntry* control = search_containing_control(ctx, body->payload.join.join_point);
            if (!control)
                longjmp(ctx->bail, 1);

            BodyBuilder* bb = begin_body(a);
            gen_store(bb, ctx->level_ptr, int32_literal(a, control->depth - 1));

            Nodes args = rewrite_nodes(&ctx->rewriter, body->payload.join.args);
            for (size_t i = 0; i < args.count; i++) {
                gen_store(bb, control->phis[i], args.nodes[i]);
            }

            return finish_body(bb, exit_ladder);
        }

        case Return_TAG:
        case Unreachable_TAG: return recreate_node_identity(&ctx->rewriter, body);

        case TailCall_TAG: longjmp(ctx->bail, 1);

        case If_TAG:
        case Terminator_BlockYield_TAG:
        case Terminator_MergeBreak_TAG:
        case Terminator_MergeContinue_TAG:
        case Terminator_MergeSelection_TAG: error("Only control nodes are tolerated here.")
    }
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
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
                new->payload.fun.body = rewrite_node(&ctx2.rewriter, node->payload.fun.body);
            // builtin functions are always considered leaf functions
            is_leaf = is_builtin || !node->payload.fun.body;
        } else {
            ctx2.lower = true;
            BodyBuilder* bb = begin_body(a);
            TmpAllocCleanupClosure cj1 = create_cancel_body_closure(bb);
            append_list(TmpAllocCleanupClosure, ctx->cleanup_stack, cj1);
            const Node* ptr = gen_local_alloc(bb, int32_type(a));
            set_variable_name(ptr, "cf_depth");
            gen_store(bb, ptr, int32_literal(a, 0));
            ctx2.level_ptr = ptr;
            ctx2.fn = new;
            struct Dict* tmp_processed = clone_dict(ctx->rewriter.map);
            TmpAllocCleanupClosure cj2 = create_delete_dict_closure(tmp_processed);
            append_list(TmpAllocCleanupClosure, ctx->cleanup_stack, cj2);
            ctx2.rewriter.map = tmp_processed;
            new->payload.fun.body = finish_body(bb, structure(&ctx2, node, unreachable(a)));
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
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
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
