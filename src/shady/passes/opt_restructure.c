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

static const Node* structure(Context* ctx, const Node* body, const Node* exit_ladder);

static const Node* handle_bb_callsite(Context* ctx, BodyBuilder* bb, const Node* j, const Node* exit_ladder) {
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
        return finish_body(bb, merge_continue(a, (MergeContinue) {
            .args = rewrite_nodes(&ctx->rewriter, oargs)
        }));
    } else {
        Nodes oparams = get_abstraction_params(dst);
        assert(oparams.count == oargs.count);
        Context ctx2 = *ctx;

        struct Dict* tmp_processed = clone_dict(ctx->rewriter.map);
        append_list(struct Dict*, ctx->tmp_alloc_stack, tmp_processed);
        ctx2.rewriter.map = tmp_processed;
        
        // Record each step of the depth-first search on a stack so we can identify loops
        DFSStackEntry dfs_entry = { .parent = ctx->dfs_stack, .old = dst, .containing_control = ctx->control_stack };
        ctx2.dfs_stack = &dfs_entry;

        // We use a basic block for the exit ladder because we don't know what the ladder needs to do ahead of time
        // opt_simplify_cf will later inline this
        Node* inner_exit_ladder_bb = basic_block(a, ctx->fn, empty(a), unique_name(a, "exit_ladder_inline_me"));

        const Node* new;

        // Just jumps to the actual ladder
        if (dfs_entry.loop_header) {
            LARRAY(const Node*, nparams, oargs.count);
            for (size_t i = 0; i < oargs.count; i++) {
                nparams[i] = var(a, rewrite_node(&ctx->rewriter, oparams.nodes[i]->type), "arg");
                register_processed(&ctx2.rewriter, oparams.nodes[i], nparams[i]);
            }

            const Node* structured = structure(&ctx2, get_abstraction_body(dst), jump(a, (Jump) {.target = inner_exit_ladder_bb, .args = empty(a)}));
            assert(is_terminator(structured));

            const Node* body = case_(a, nodes(a, oargs.count, nparams), structured);
            create_structured_loop(bb, empty(a), rewrite_nodes(&ctx->rewriter, oargs), body);
            // we decide 'late' what the exit ladder should be
            inner_exit_ladder_bb->payload.basic_block.body = merge_break(a, (MergeBreak) { .args = empty(a) });
            new = finish_body(bb, exit_ladder);
        } else {
            register_processed_list(&ctx2.rewriter, oparams, rewrite_nodes(&ctx->rewriter, oargs));

            const Node* structured = structure(&ctx2, get_abstraction_body(dst), jump(a, (Jump) {.target = inner_exit_ladder_bb, .args = empty(a)}));
            assert(is_terminator(structured));

            inner_exit_ladder_bb->payload.basic_block.body = exit_ladder;
            new = finish_body(bb, structured);
        }

        // forget we rewrote all that
        destroy_dict(tmp_processed);
        pop_list_impl(ctx->tmp_alloc_stack);

        return new;
    }
    SHADY_UNREACHABLE;
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

static const Node* structure(Context* ctx, const Node* terminator, const Node* exit_ladder) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (is_terminator(terminator)) {
        case InsertHelperEnd_TAG: assert(false);
        case NotATerminator:
        case Body_TAG: {
            return body(a, (Body) {
                .instructions = rewrite_nodes(&ctx->rewriter, terminator->payload.body.instructions),
                .terminator = structure(ctx, terminator->payload.body.terminator, exit_ladder)
            });
        }
        case If_TAG:
        case Loop_TAG:
        case Match_TAG: error("not supposed to exist in IR at this stage");
        case Control_TAG: {
            const Node* old_control_body = terminator->payload.control.inside;
            assert(old_control_body->tag == Case_TAG);
            Nodes old_control_params = get_abstraction_params(old_control_body);
            assert(old_control_params.count == 1);

            const Node* old_tail = terminator->payload.control.tail;
            assert(old_tail->tag == Case_TAG);

            // Create N temporary variables to hold the join point arguments
            BodyBuilder* bb_outer = begin_body(a);
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, terminator->payload.control.yield_types);
            LARRAY(const Node*, phis, yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                const Type* type = yield_types.nodes[i];
                assert(is_data_type(type));
                phis[i] = first(bind_instruction_named(bb_outer, prim_op(a, (PrimOp) { .op = alloca_logical_op, .type_arguments = singleton(type) }), (String []) {"ctrl_phi" }));
            }

            // Create a new context to rewrite the terminator with
            // TODO: Bail if we try to re-enter the same control construct
            Context control_ctx = *ctx;
            ControlEntry control_entry = {
                .parent = ctx->control_stack,
                .old_token = first(old_control_params),
                .phis = phis,
                .depth = ctx->control_stack ? ctx->control_stack->depth + 1 : 1,
            };
            control_ctx.control_stack = &control_entry;

            // Set the depth for threads entering the control terminator
            bind_instruction(bb_outer, prim_op(a, (PrimOp) { .op = store_op, .operands = mk_nodes(a, ctx->level_ptr, int32_literal(a, control_entry.depth)) }));

            // Start building out the tail, first it needs to dereference the phi variables to recover the arguments given to join()
            BodyBuilder* bb2 = begin_body(a);
            LARRAY(const Node*, phi_values, yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                phi_values[i] = first(bind_instruction(bb2, prim_op(a, (PrimOp) { .op = load_op, .operands = singleton(phis[i]) })));
                // register_processed(&ctx->rewriter, otail_params.nodes[i], phi_values[i]);
                register_processed(&ctx->rewriter, get_abstraction_params(old_tail).nodes[i], phi_values[i]);
            }

            // Wrap the tail in a guarded if, to handle 'far' joins
            const Node* level_value = first(bind_instruction(bb2, prim_op(a, (PrimOp) { .op = load_op, .operands = singleton(ctx->level_ptr) })));
            const Node* guard = first(bind_instruction(bb2, prim_op(a, (PrimOp) { .op = eq_op, .operands = mk_nodes(a, level_value, int32_literal(a, ctx->control_stack ? ctx->control_stack->depth : 0)) })));
            const Node* true_body = structure(ctx, get_abstraction_body(old_tail), yield(a, (Yield) { .args = empty(a) }));
            const Node* if_true_lam = case_(a, empty(a), true_body);
            create_structured_if(bb2, empty(a), guard, if_true_lam, NULL);

            return finish_body(bb_outer, structure(&control_ctx, old_control_body, finish_body(bb2, exit_ladder)));
        }
        case Jump_TAG: {
            BodyBuilder* bb = begin_body(a);
            return handle_bb_callsite(ctx, bb, terminator, exit_ladder);
        }
        // br(cond, true_bb, false_bb, args)
        // becomes
        // let(if(cond, _ => handle_bb_callsite[true_bb, args], _ => handle_bb_callsite[false_bb, args]), _ => unreachable)
        case Branch_TAG: {
            const Node* condition = rewrite_node(&ctx->rewriter, terminator->payload.branch.branch_condition);

            BodyBuilder* true_bb = begin_body(a);
            const Node* true_body = handle_bb_callsite(ctx, true_bb, terminator->payload.branch.true_destination, yield(a, (Yield) { .args = empty(a) }));
            const Node* true_case = case_(a, empty(a), true_body);

            BodyBuilder* false_bb = begin_body(a);
            const Node* false_body = handle_bb_callsite(ctx, false_bb, terminator->payload.branch.false_destination, yield(a, (Yield) { .args = empty(a) }));
            const Node* false_case = case_(a, empty(a), false_body);

            BodyBuilder* bb = begin_body(a);
            create_structured_if(bb, empty(a), condition, false_case, false_case);
            return finish_body(bb, exit_ladder);
        }
        case Switch_TAG: {
            const Node* switch_value = rewrite_node(&ctx->rewriter, terminator->payload.br_switch.inspect);

            BodyBuilder* default_bb = begin_body(a);
            const Node* default_body = handle_bb_callsite(ctx, default_bb, terminator->payload.br_switch.default_destination, yield(a, (Yield) { .args = empty(a) }));
            const Node* default_case = case_(a, empty(a), default_body);

            LARRAY(const Node*, cases, terminator->payload.br_switch.destinations.count);
            for (size_t i = 0; i < terminator->payload.br_switch.destinations.count; i++) {
                BodyBuilder* bb = begin_body(a);
                cases[i] = case_(a, empty(a), handle_bb_callsite(ctx, bb, terminator->payload.br_switch.destinations.nodes[i], yield(a, (Yield) {.args = empty(a)})));
            }

            BodyBuilder* bb = begin_body(a);
            create_structured_match(bb, empty(a), switch_value, rewrite_nodes(&ctx->rewriter, terminator->payload.br_switch.literals), nodes(a, terminator->payload.br_switch.destinations.count, cases), default_case);
            return finish_body(bb, exit_ladder);
        }
        case Join_TAG: {
            ControlEntry* control = search_containing_control(ctx, terminator->payload.join.join_point);
            if (!control)
                longjmp(ctx->bail, 1);

            BodyBuilder* bb = begin_body(a);
            bind_instruction(bb, prim_op(a, (PrimOp) { .op = store_op, .operands = mk_nodes(a, ctx->level_ptr, int32_literal(a, control->depth - 1)) }));

            Nodes args = rewrite_nodes(&ctx->rewriter, terminator->payload.join.args);
            for (size_t i = 0; i < args.count; i++) {
                bind_instruction(bb, prim_op(a, (PrimOp) { .op = store_op, .operands = mk_nodes(a, control->phis[i], args.nodes[i]) }));
            }

            return finish_body(bb, exit_ladder);
        }

        case Return_TAG:
        case Unreachable_TAG: return recreate_node_identity(&ctx->rewriter, terminator);

        case TailCall_TAG: longjmp(ctx->bail, 1);

        case Terminator_MergeBreak_TAG:
        case Terminator_MergeContinue_TAG:
        case Yield_TAG: error("Only control nodes are tolerated here.")
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

        size_t alloc_stack_size_now = entries_count_list(ctx->tmp_alloc_stack);

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
            const Node* ptr = first(bind_instruction_named(bb, prim_op(a, (PrimOp) { .op = alloca_logical_op, .type_arguments = singleton(int32_type(a)) }), (String []) {"cf_depth" }));
            bind_instruction(bb, prim_op(a, (PrimOp) { .op = store_op, .operands = mk_nodes(a, ptr, int32_literal(a, 0)) }));
            ctx2.level_ptr = ptr;
            ctx2.fn = new;
            struct Dict* tmp_processed = clone_dict(ctx->rewriter.map);
            append_list(struct Dict*, ctx->tmp_alloc_stack, tmp_processed);
            ctx2.rewriter.map = tmp_processed;
            new->payload.fun.body = finish_body(bb, structure(&ctx2, get_abstraction_body(node), unreachable(a)));
            is_leaf = true;
        }

        //if (is_leaf)
        //    new->payload.fun.annotations = append_nodes(arena, new->payload.fun.annotations, annotation(arena, (Annotation) { .name = "Leaf" }));

        // if we did a longjmp, we might have orphaned a few of those
        while (alloc_stack_size_now < entries_count_list(ctx->tmp_alloc_stack)) {
            struct Dict* orphan = pop_last_list(struct Dict*, ctx->tmp_alloc_stack);
            destroy_dict(orphan);
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
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

Module* opt_restructurize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .tmp_alloc_stack = new_list(struct Dict*),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_list(ctx.tmp_alloc_stack);
    return dst;
}
