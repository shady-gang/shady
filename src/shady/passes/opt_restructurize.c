#include "passes.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"

#include <setjmp.h>

#pragma GCC diagnostic error "-Wswitch"

typedef struct {
    const Node* old;
    Node* new;
} TodoEntry;

typedef struct DFSStackEntry_ DFSStackEntry;
struct DFSStackEntry_ {
    DFSStackEntry* parent;
    const Node* old;

    bool loop_header;
    bool in_loop;
};

typedef struct ControlEntry_ ControlEntry;
struct ControlEntry_ {
    ControlEntry* parent;
    const Node* old_token;
    const Node** phis;
    int depth;
};

typedef struct {
    Rewriter rewriter;
    TodoEntry* todos;
    size_t todo_count;

    bool lower;
    const Node* level_ptr;
    DFSStackEntry* dfs_stack;
    ControlEntry* control_stack;
} Context;

static DFSStackEntry* encountered_before(Context* ctx, const Node* bb, size_t* path_len) {
    DFSStackEntry* entry = ctx->dfs_stack;
    assert(entry);
    entry = entry->parent;
    if (path_len) *path_len = 1;
    while (entry != NULL) {
        if (entry->old == bb)
            return entry;
        entry = entry->parent;
        if (path_len) (*path_len)++;
    }
    return entry;
}

static const Node* structure(Context* entry_ctx, const Node* abs, const Node* exit_ladder);

static const Node* handle_bb_callsite(Context* ctx, BodyBuilder* bb, const Node* dst, Nodes args, const Node* exit_ladder) {
    IrArena* arena = ctx->rewriter.dst_arena;
    DFSStackEntry* entry = ctx->dfs_stack;
    // assert(entry->old == dst);

    size_t path_len;
    DFSStackEntry* prior_encounter = encountered_before(ctx, dst, &path_len);
    if (prior_encounter) {
        // Create path
        LARRAY(const Node*, path, path_len);
        DFSStackEntry* entry2 = ctx->dfs_stack->parent;
        for (size_t i = 0; i < path_len; i++) {
            assert(entry2);
            path[path_len - 1 - i] = entry2->old;
            assert(!entry2->in_loop); // TODO: BAIL
            entry2->in_loop = true;
            entry2 = entry2->parent;
        }
        prior_encounter->loop_header = true;
        return finish_body(bb, merge_continue(ctx->rewriter.dst_arena, (MergeContinue) {
            .args = rewrite_nodes(&ctx->rewriter, args)
        }));
    } else {
        LARRAY(const Node*, vars, args.count);
        struct Dict* old_processed = clone_dict(ctx->rewriter.processed);
        for (size_t i = 0; i < args.count; i++) {
            vars[i] = var(ctx->rewriter.dst_arena, args.nodes[i]->type, "arg");
            register_processed(&ctx->rewriter, args.nodes[i], vars[i]);
        }
        Nodes varss = nodes(ctx->rewriter.dst_arena, args.count, vars);
        const Node* structured = structure(ctx, dst, merge_break(arena, (MergeBreak) { .args = empty(arena) }));
        assert(is_terminator(structured));
        // forget we rewrote all that
        destroy_dict(ctx->rewriter.processed);
        ctx->rewriter.processed = old_processed;

        Node* body = lambda(ctx->rewriter.dst_module, varss);
        body->payload.anon_lam.body = structured;
        bind_instruction(bb, loop_instr(ctx->rewriter.dst_arena, (Loop) {
            .body = body,
            .initial_args = args,
            .yield_types = nodes(ctx->rewriter.dst_arena, 0, NULL),
        }));
        return finish_body(bb, exit_ladder);
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

static const Node* structure(Context* entry_ctx, const Node* abs, const Node* exit_ladder) {
    IrArena* arena = entry_ctx->rewriter.dst_arena;

    // Record each step of the depth-first search on a stack so we can identify loops
    DFSStackEntry dfs_entry = { .parent = entry_ctx->dfs_stack, .old = abs };
    Context ctx2 = *entry_ctx;
    ctx2.dfs_stack = &dfs_entry;

    const Node* body = get_abstraction_body(abs);
    switch (is_terminator(body)) {
        case NotATerminator:
        case LetMut_TAG: assert(false);
        case Let_TAG: {
            const Node* old_tail = get_let_tail(body);
            Nodes otail_params = get_abstraction_params(old_tail);

            const Node* old_instr = get_let_instruction(body);
            switch (is_instruction(old_instr)) {
                case NotAnInstruction: assert(false);
                case Instruction_If_TAG:
                case Instruction_Loop_TAG:
                case Instruction_Match_TAG: error("not supposed to exist in IR at this stage");
                case Instruction_LeafCall_TAG:
                case Instruction_PrimOp_TAG: {
                    Nodes rewritten_params = recreate_variables(&ctx2.rewriter, otail_params);
                    register_processed_list(&ctx2.rewriter, otail_params, rewritten_params);
                    Node* structured_lam = lambda(ctx2.rewriter.dst_module, rewritten_params);
                    structured_lam->payload.anon_lam.body = structure(&ctx2, old_tail, exit_ladder);
                    return let(arena, recreate_node_identity(&ctx2.rewriter, old_instr), structured_lam);
                }
                case Instruction_IndirectCall_TAG: error("TODO: bail");
                // let(control(body), tail)
                // var phi = undef; level = N+1; structurize[body, if (level == N+1, _ => tail(load(phi))); structured_exit_terminator]
                case Instruction_Control_TAG: {
                    const Node* old_control_body = old_instr->payload.control.inside;
                    assert(old_control_body->tag == AnonLambda_TAG);
                    Nodes old_control_params = get_abstraction_params(old_control_body);
                    assert(old_control_params.count == 1);

                    // Create N temporary variables to hold the join point arguments
                    BodyBuilder* bb_outer = begin_body(ctx2.rewriter.dst_module);
                    Nodes yield_types = rewrite_nodes(&ctx2.rewriter, old_instr->payload.control.yield_types);
                    LARRAY(const Node*, phis, yield_types.count);
                    for (size_t i = 0; i < yield_types.count; i++) {
                        const Type* type = extract_operand_type(yield_types.nodes[i]);
                        phis[i] = first(bind_instruction(bb_outer, prim_op(arena, (PrimOp) { .op = alloca_logical_op, .type_arguments = singleton(type) })));
                    }

                    // Create a new context to rewrite the body with
                    // TODO: Bail if we try to re-enter the same control construct
                    Context control_ctx = ctx2;
                    control_ctx.dfs_stack = NULL;
                    ControlEntry control_entry = {
                        .parent = ctx2.control_stack,
                        .old_token = first(old_control_params),
                        .phis = phis,
                        .depth = ctx2.control_stack ? ctx2.control_stack->depth + 1 : 1,
                    };
                    control_ctx.control_stack = &control_entry;

                    // Set the depth for threads entering the control body
                    bind_instruction(bb_outer, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, ctx2.level_ptr, int32_literal(arena, control_entry.depth)) }));

                    // Start building out the tail, first it needs to dereference the phi variables to recover the arguments given to join()
                    Node* tail_lambda = lambda(ctx2.rewriter.dst_module, empty(arena));
                    BodyBuilder* bb2 = begin_body(ctx2.rewriter.dst_module);
                    LARRAY(const Node*, phi_values, yield_types.count);
                    for (size_t i = 0; i < yield_types.count; i++) {
                        phi_values[i] = first(bind_instruction(bb2, prim_op(arena, (PrimOp) { .op = load_op, .operands = singleton(phis[i]) })));
                        register_processed(&ctx2.rewriter, otail_params.nodes[i], phi_values[i]);
                    }

                    // Wrap the tail in a guarded if, to handle 'far' joins
                    const Node* level_value = first(bind_instruction(bb2, prim_op(arena, (PrimOp) { .op = load_op, .operands = singleton(ctx2.level_ptr) })));
                    const Node* guard = first(bind_instruction(bb2, prim_op(arena, (PrimOp) { .op = eq_op, .operands = mk_nodes(arena, level_value, int32_literal(arena, ctx2.control_stack ? ctx2.control_stack->depth : 0)) })));
                    Node* if_true_lam = lambda(ctx2.rewriter.dst_module, empty(arena));
                    if_true_lam->payload.anon_lam.body = structure(&ctx2, old_tail, merge_selection(arena, (MergeSelection) { .args = empty(arena) }));
                    bind_instruction(bb2, if_instr(arena, (If) {
                        .condition = guard,
                        .yield_types = empty(arena),
                        .if_true = if_true_lam,
                        .if_false = NULL
                    }));

                    tail_lambda->payload.anon_lam.body = finish_body(bb2, exit_ladder);
                    return finish_body(bb_outer, structure(&control_ctx, old_control_body, let(arena, unit(arena), tail_lambda)));
                }
            }
        }
        case Jump_TAG: {
            BodyBuilder* bb = begin_body(ctx2.rewriter.dst_module);
            return handle_bb_callsite(&ctx2, bb, body->payload.jump.target, body->payload.jump.args, exit_ladder);
        }
        // br(cond, true_bb, false_bb, args)
        // becomes
        // let(if(cond, _ => handle_bb_callsite[true_bb, args], _ => handle_bb_callsite[false_bb, args]), _ => unreachable)
        case Branch_TAG: {
            const Node* condition = rewrite_node(&ctx2.rewriter, body->payload.branch.branch_condition);

            Node* if_true_lam = lambda(ctx2.rewriter.dst_module, empty(ctx2.rewriter.dst_arena));
            BodyBuilder* if_true_bb = begin_body(ctx2.rewriter.dst_module);
            if_true_lam->payload.anon_lam.body = handle_bb_callsite(&ctx2, if_true_bb, body->payload.branch.true_target, body->payload.branch.args, merge_selection(arena, (MergeSelection) { .args = empty(arena) }));

            Node* if_false_lam = lambda(ctx2.rewriter.dst_module, empty(ctx2.rewriter.dst_arena));
            BodyBuilder* if_false_bb = begin_body(ctx2.rewriter.dst_module);
            if_false_lam->payload.anon_lam.body = handle_bb_callsite(&ctx2, if_false_bb, body->payload.branch.false_target, body->payload.branch.args, merge_selection(arena, (MergeSelection) { .args = empty(arena) }));

            const Node* instr = if_instr(ctx2.rewriter.dst_arena, (If) {
                .condition = condition,
                .yield_types = empty(ctx2.rewriter.dst_arena),
                .if_true = if_true_lam,
                .if_false = if_false_lam,
            });
            Node* post_merge_lam = lambda(ctx2.rewriter.dst_module, empty(ctx2.rewriter.dst_arena));
            post_merge_lam->payload.anon_lam.body = exit_ladder;
            return let(ctx2.rewriter.dst_arena, instr, post_merge_lam);
        }
        case Switch_TAG: {
            error("TODO");
        }
        case Join_TAG: {
            ControlEntry* control = search_containing_control(&ctx2, body->payload.join.join_point);
            assert(control); // TODO bail out if we can't find control instead

            BodyBuilder* bb = begin_body(ctx2.rewriter.dst_module);
            bind_instruction(bb, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, ctx2.level_ptr, int32_literal(arena, control->depth - 1)) }));

            Nodes args = rewrite_nodes(&ctx2.rewriter, body->payload.join.args);
            for (size_t i = 0; i < args.count; i++) {
                bind_instruction(bb, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, control->phis[i], args.nodes[i]) }));
            }

            return finish_body(bb, exit_ladder);
        }

        case Return_TAG:
        case Unreachable_TAG: return recreate_node_identity(&ctx2.rewriter, body);

        case TailCall_TAG: error("TODO: bail")

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

    if (node->tag == Function_TAG) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
        ctx->todos[ctx->todo_count++] = (TodoEntry) { .old = node, .new = new };
        return new;
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
    LARRAY(TodoEntry, todos, get_module_declarations(src).count);
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .todos = todos,
    };
    rewrite_module(&ctx.rewriter);

    for (size_t i = 0; i < ctx.todo_count; i++) {
        Context ctx2 = ctx;
        ctx2.lower = lookup_annotation(ctx.todos[i].old, "RestructureMe");
        ctx2.dfs_stack = NULL;
        ctx2.control_stack = NULL;
        ctx2.todos = NULL;
        ctx2.todo_count = 0;
        if (ctx2.lower) {
            BodyBuilder* bb = begin_body(ctx.rewriter.dst_module);
            const Node* ptr = first(bind_instruction(bb, prim_op(arena, (PrimOp) { .op = alloca_logical_op, .type_arguments = singleton(int32_type(arena)) })));
            bind_instruction(bb, prim_op(arena, (PrimOp) { .op = store_op, .operands = mk_nodes(arena, ptr, int32_literal(arena, 0)) }));
            ctx2.level_ptr = ptr;
            ctx.todos[i].new->payload.fun.body = finish_body(bb, structure(&ctx2, ctx.todos[i].old, unreachable(ctx.rewriter.dst_arena)));
        } else
            ctx.todos[i].new->payload.fun.body = rewrite_node(&ctx2.rewriter, ctx.todos[i].old->payload.fun.body);
    }

    destroy_rewriter(&ctx.rewriter);
}
