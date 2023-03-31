#include "passes.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"

#include <setjmp.h>
#include <string.h>

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

typedef struct {
    Rewriter rewriter;
    struct List* tmp_alloc_stack;

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

static const Node* handle_bb_callsite(Context* ctx, BodyBuilder* bb, const Node* caller, const Node* dst, Nodes oargs, const Node* exit_ladder) {
    IrArena* arena = ctx->rewriter.dst_arena;

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
        return finish_body(bb, merge_continue(arena, (MergeContinue) {
            .args = rewrite_nodes(&ctx->rewriter, oargs)
        }));
    } else {
        Nodes oparams = get_abstraction_params(dst);
        assert(oparams.count == oargs.count);
        LARRAY(const Node*, nparams, oargs.count);
        Context ctx2 = *ctx;
        
        // Record each step of the depth-first search on a stack so we can identify loops
        DFSStackEntry dfs_entry = { .parent = ctx->dfs_stack, .old = dst, .containing_control = ctx->control_stack };
        ctx2.dfs_stack = &dfs_entry;
        
        struct Dict* tmp_processed = clone_dict(ctx->rewriter.processed);
        append_list(struct Dict*, ctx->tmp_alloc_stack, tmp_processed);
        ctx2.rewriter.processed = tmp_processed;
        for (size_t i = 0; i < oargs.count; i++) {
            nparams[i] = var(arena, rewrite_node(&ctx->rewriter, oparams.nodes[i]->type), "arg");
            register_processed(&ctx2.rewriter, oparams.nodes[i], nparams[i]);
        }

        // We use a basic block for the exit ladder because we don't know what the ladder needs to do ahead of time
        // opt_simplify_cf will later inline this
        Node* inner_exit_ladder_bb = basic_block(arena, ctx->fn, empty(arena), unique_name(arena, "exit_ladder_inline_me"));

        // Just jumps to the actual ladder
        const Node* exit_ladder_trampoline = lambda(ctx2.rewriter.dst_module, empty(arena), jump(arena, (Jump) { .target = inner_exit_ladder_bb, .args = empty(arena) }));

        const Node* structured = structure(&ctx2, dst, let(arena, unit(arena), exit_ladder_trampoline));
        assert(is_terminator(structured));
        // forget we rewrote all that
        destroy_dict(tmp_processed);
        pop_list_impl(ctx->tmp_alloc_stack);

        if (dfs_entry.loop_header) {
            const Node* body = lambda(ctx->rewriter.dst_module, nodes(arena, oargs.count, nparams), structured);
            bind_instruction(bb, loop_instr(arena, (Loop) {
                .body = body,
                .initial_args = rewrite_nodes(&ctx->rewriter, oargs),
                .yield_types = nodes(arena, 0, NULL),
            }));
            // we decide 'late' what the exit ladder should be
            inner_exit_ladder_bb->payload.basic_block.body = merge_break(arena, (MergeBreak) { .args = empty(arena) });
            return finish_body(bb, exit_ladder);
        } else {
            bind_variables(bb, nodes(arena, oargs.count, nparams), rewrite_nodes(&ctx->rewriter, oargs));
            inner_exit_ladder_bb->payload.basic_block.body = exit_ladder;
            return finish_body(bb, structured);
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
    IrArena* arena = ctx->rewriter.dst_arena;
    const Node* old_tail = get_let_tail(old_let);
    Nodes otail_params = get_abstraction_params(old_tail);

    Nodes rewritten_params = recreate_variables(&ctx->rewriter, otail_params);
    register_processed_list(&ctx->rewriter, otail_params, rewritten_params);
    const Node* structured_lam = lambda(ctx->rewriter.dst_module, rewritten_params, structure(ctx, old_tail, exit_ladder));
    return let(arena, new_instruction, structured_lam);
}

static const Node* structure(Context* ctx, const Node* abs, const Node* exit_ladder) {
    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* body = get_abstraction_body(abs);
    assert(body);
    switch (is_terminator(body)) {
        case NotATerminator:
        case LetMut_TAG: assert(false);
        case Terminator_Yield_TAG: error("Should be eliminated by the compiler");
        case Let_TAG: {
            const Node* old_tail = get_let_tail(body);
            Nodes otail_params = get_abstraction_params(old_tail);

            const Node* old_instr = get_let_instruction(body);
            switch (is_instruction(old_instr)) {
                case NotAnInstruction: assert(false);
                case Instruction_If_TAG:
                case Instruction_Loop_TAG:
                case Instruction_Match_TAG: error("not supposed to exist in IR at this stage");
                case Instruction_Block_TAG: error("Should be eliminated by the compiler");
                case Instruction_LeafCall_TAG:
                case Instruction_PrimOp_TAG: {
                    return rebuild_let(ctx, body, recreate_node_identity(&ctx->rewriter, old_instr), exit_ladder);
                }
                case Instruction_IndirectCall_TAG: {
                    const Node* callee = old_instr->payload.indirect_call.callee;
                    if (callee->tag == FnAddr_TAG) {
                        const Node* fn = rewrite_node(&ctx->rewriter, callee->payload.fn_addr.fn);
                        if (lookup_annotation(fn, "Leaf")) {
                            const Node* call = leaf_call(arena, (LeafCall) {
                                .callee = fn,
                                .args = rewrite_nodes(&ctx->rewriter, old_instr->payload.indirect_call.args)
                            });
                            return rebuild_let(ctx, body, call, exit_ladder);
                        }
                    }
                    // if we don't manage that, give up :(
                    longjmp(ctx->bail, 1);
                }
                // let(control(body), tail)
                // var phi = undef; level = N+1; structurize[body, if (level == N+1, _ => tail(load(phi))); structured_exit_terminator]
                case Instruction_Control_TAG: {
                    const Node* old_control_body = old_instr->payload.control.inside;
                    assert(old_control_body->tag == AnonLambda_TAG);
                    Nodes old_control_params = get_abstraction_params(old_control_body);
                    assert(old_control_params.count == 1);

                    // Create N temporary variables to hold the join point arguments
                    BodyBuilder* bb_outer = begin_body(ctx->rewriter.dst_module);
                    Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instr->payload.control.yield_types);
                    LARRAY(const Node*, phis, yield_types.count);
                    for (size_t i = 0; i < yield_types.count; i++) {
                        const Type* type = yield_types.nodes[i];
                        assert(is_data_type(type));
                        phis[i] = first(bind_instruction_named(bb_outer, prim_op(arena, (PrimOp) { .op = alloca_logical_op, .type_arguments = singleton(type) }), (String []) { "ctrl_phi" }));
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
                    bind_instruction(bb_outer, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, ctx->level_ptr, int32_literal(arena, control_entry.depth)) }));

                    // Start building out the tail, first it needs to dereference the phi variables to recover the arguments given to join()
                    BodyBuilder* bb2 = begin_body(ctx->rewriter.dst_module);
                    LARRAY(const Node*, phi_values, yield_types.count);
                    for (size_t i = 0; i < yield_types.count; i++) {
                        phi_values[i] = first(bind_instruction(bb2, prim_op(arena, (PrimOp) { .op = load_op, .operands = singleton(phis[i]) })));
                        register_processed(&ctx->rewriter, otail_params.nodes[i], phi_values[i]);
                    }

                    // Wrap the tail in a guarded if, to handle 'far' joins
                    const Node* level_value = first(bind_instruction(bb2, prim_op(arena, (PrimOp) { .op = load_op, .operands = singleton(ctx->level_ptr) })));
                    const Node* guard = first(bind_instruction(bb2, prim_op(arena, (PrimOp) { .op = eq_op, .operands = mk_nodes(arena, level_value, int32_literal(arena, ctx->control_stack ? ctx->control_stack->depth : 0)) })));
                    const Node* true_body = structure(ctx, old_tail, merge_selection(arena, (MergeSelection) { .args = empty(arena) }));
                    const Node* if_true_lam = lambda(ctx->rewriter.dst_module, empty(arena), true_body);
                    bind_instruction(bb2, if_instr(arena, (If) {
                        .condition = guard,
                        .yield_types = empty(arena),
                        .if_true = if_true_lam,
                        .if_false = NULL
                    }));

                    const Node* tail_lambda = lambda(ctx->rewriter.dst_module, empty(arena), finish_body(bb2, exit_ladder));
                    return finish_body(bb_outer, structure(&control_ctx, old_control_body, let(arena, unit(arena), tail_lambda)));
                }
            }
        }
        case Jump_TAG: {
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            return handle_bb_callsite(ctx, bb, abs, body->payload.jump.target, body->payload.jump.args, exit_ladder);
        }
        // br(cond, true_bb, false_bb, args)
        // becomes
        // let(if(cond, _ => handle_bb_callsite[true_bb, args], _ => handle_bb_callsite[false_bb, args]), _ => unreachable)
        case Branch_TAG: {
            const Node* condition = rewrite_node(&ctx->rewriter, body->payload.branch.branch_condition);

            BodyBuilder* if_true_bb = begin_body(ctx->rewriter.dst_module);
            const Node* true_body = handle_bb_callsite(ctx, if_true_bb, abs, body->payload.branch.true_target, body->payload.branch.args, merge_selection(arena, (MergeSelection) { .args = empty(arena) }));
            const Node* if_true_lam = lambda(ctx->rewriter.dst_module, empty(ctx->rewriter.dst_arena), true_body);

            BodyBuilder* if_false_bb = begin_body(ctx->rewriter.dst_module);
            const Node* false_body = handle_bb_callsite(ctx, if_false_bb, abs, body->payload.branch.false_target, body->payload.branch.args, merge_selection(arena, (MergeSelection) { .args = empty(arena) }));
            const Node* if_false_lam = lambda(ctx->rewriter.dst_module, empty(ctx->rewriter.dst_arena), false_body);

            const Node* instr = if_instr(arena, (If) {
                .condition = condition,
                .yield_types = empty(arena),
                .if_true = if_true_lam,
                .if_false = if_false_lam,
            });
            const Node* post_merge_lam = lambda(ctx->rewriter.dst_module, empty(ctx->rewriter.dst_arena), exit_ladder);
            return let(ctx->rewriter.dst_arena, instr, post_merge_lam);
        }
        case Switch_TAG: {
            error("TODO");
        }
        case Join_TAG: {
            ControlEntry* control = search_containing_control(ctx, body->payload.join.join_point);
            if (!control)
                longjmp(ctx->bail, 1);

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            bind_instruction(bb, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, ctx->level_ptr, int32_literal(arena, control->depth - 1)) }));

            Nodes args = rewrite_nodes(&ctx->rewriter, body->payload.join.args);
            for (size_t i = 0; i < args.count; i++) {
                bind_instruction(bb, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, control->phis[i], args.nodes[i]) }));
            }

            return finish_body(bb, exit_ladder);
        }

        case Return_TAG:
        case Unreachable_TAG: return recreate_node_identity(&ctx->rewriter, body);

        case TailCall_TAG: longjmp(ctx->bail, 1);

        case Terminator_MergeBreak_TAG:
        case Terminator_MergeContinue_TAG:
        case Terminator_MergeSelection_TAG: error("Only control nodes are tolerated here.")
    }
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    assert(arena != node->arena);
    assert(node->arena == ctx->rewriter.src_arena);

    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    if (is_declaration(node)) {
        String name = get_decl_name(node);
        Nodes decls = get_module_declarations(ctx->rewriter.dst_module);
        for (size_t i = 0; i < decls.count; i++) {
            if (strcmp(get_decl_name(decls.nodes[i]), name) == 0)
                return decls.nodes[i];
        }
    }

    if (node->tag == Function_TAG) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, node);

        size_t alloc_stack_size_now = entries_count_list(ctx->tmp_alloc_stack);

        Context ctx2 = *ctx;
        ctx2.dfs_stack = NULL;
        ctx2.control_stack = NULL;
        bool is_builtin = lookup_annotation(node, "Builtin");
        bool is_leaf = false;
        if (is_builtin || !node->payload.fun.body || !lookup_annotation(node, "MaybeLeaf") || setjmp(ctx2.bail)) {
            ctx2.lower = false;
            ctx2.rewriter.processed = ctx->rewriter.processed;
            if (node->payload.fun.body)
                new->payload.fun.body = rewrite_node(&ctx2.rewriter, node->payload.fun.body);
            // builtin functions are always considered leaf functions
            is_leaf = is_builtin || !node->payload.fun.body;
        } else {
            ctx2.lower = true;
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            const Node* ptr = first(bind_instruction_named(bb, prim_op(arena, (PrimOp) { .op = alloca_logical_op, .type_arguments = singleton(int32_type(arena)) }), (String []) { "cf_depth" }));
            bind_instruction(bb, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, ptr, int32_literal(arena, 0)) }));
            ctx2.level_ptr = ptr;
            ctx2.fn = new;
            struct Dict* tmp_processed = clone_dict(ctx->rewriter.processed);
            append_list(struct Dict*, ctx->tmp_alloc_stack, tmp_processed);
            ctx2.rewriter.processed = tmp_processed;
            new->payload.fun.body = finish_body(bb, structure(&ctx2, node, unreachable(arena)));
            is_leaf = true;
        }

        if (is_leaf)
            new->payload.fun.annotations = append_nodes(arena, new->payload.fun.annotations, annotation(arena, (Annotation) { .name = "Leaf" }));

        // if we did a longjmp, we might have orphaned a few of those
        while (alloc_stack_size_now < entries_count_list(ctx->tmp_alloc_stack)) {
            struct Dict* orphan = pop_last_list(struct Dict*, ctx->tmp_alloc_stack);
            destroy_dict(orphan);
        }

        new->payload.fun.annotations = filter_out_annotation(arena, new->payload.fun.annotations, "MaybeLeaf");

        return new;
    } else if (node->tag == Instruction_IndirectCall_TAG) {
        const Node* callee = node->payload.indirect_call.callee;
        if (callee->tag == FnAddr_TAG) {
            const Node* fn = rewrite_node(&ctx->rewriter, callee->payload.fn_addr.fn);
            if (lookup_annotation(fn, "Leaf")) {
                const Node* call = leaf_call(arena, (LeafCall) {
                        .callee = fn,
                        .args = rewrite_nodes(&ctx->rewriter, node->payload.indirect_call.args)
                });
                return call;
            }
        }
    }

    if (!ctx->lower)
        return recreate_node_identity(&ctx->rewriter, node);

    // These should all be manually visited by 'structure'
    assert(!is_terminator(node) && !is_instruction(node));

    switch (node->tag) {
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void opt_restructurize(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    IrArena* arena = get_module_arena(dst);
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .tmp_alloc_stack = new_list(struct Dict*),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_list(ctx.tmp_alloc_stack);
}
