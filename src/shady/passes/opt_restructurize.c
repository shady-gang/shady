#include "passes.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"

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
    int depth;
};

typedef struct {
    Rewriter rewriter;
    TodoEntry* todos;
    size_t todo_count;

    bool lower;
    DFSStackEntry* dfs_stack;
    ControlEntry* control_stack;
} Context;

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
        //Context fn_ctx = *ctx;
        //fn_ctx.lower = lookup_annotation(node, "Leaf");
        //recreate_decl_body_identity(&fn_ctx.rewriter, node, new);
        return new;
    }

    if (!ctx->lower)
        return recreate_node_identity(&ctx->rewriter, node);

    // These should all be manuall visited by 'structure'
    assert(!is_terminator(node) && !is_instruction(node));

    switch (node->tag) {
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

static DFSStackEntry* encountered_before(Context* ctx, const Node* bb, size_t* path_len) {
    DFSStackEntry* entry = ctx->dfs_stack;
    assert(entry);
    entry = entry->parent;
    if (path_len) *path_len = 1;
    while (entry != NULL) {
        // if (entry->control_depth < ctx->depth)
        //     return NULL;
        if (entry->old == bb)
            return entry;
        entry = entry->parent;
        if (path_len) (*path_len)++;
    }
    return entry;
}

// static bool compare_path

static const Node* structure(Context* ctx, const Node* abs);

static const Node* handle_bb_callsite(Context* ctx, BodyBuilder* bb, const Node* dst, Nodes args) {
    DFSStackEntry* entry = ctx->dfs_stack;
    assert(entry->old == dst);

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
        const Node* structured = structure(ctx, dst);
        assert(is_terminator(structured));
        // forget we rewrote all that
        destroy_dict(ctx->rewriter.processed);
        ctx->rewriter.processed = old_processed;

        if (entry->loop_header) {
            Node* body = lambda(ctx->rewriter.dst_module, varss);
            body->payload.anon_lam.body = structured;
            bind_instruction(bb, loop_instr(ctx->rewriter.dst_arena, (Loop) {
                .body = body,
                .initial_args = args,
                .yield_types = nodes(ctx->rewriter.dst_arena, 0, NULL),
            }));
            return finish_body(bb, unreachable(ctx->rewriter.dst_arena));
        } else {
            bind_variables(bb, varss, args);
            return finish_body(bb, structured);
        }
    }
}

static const Node* structure(Context* ctx, const Node* abs) {
    DFSStackEntry dfs_entry = { .parent = ctx->dfs_stack, .old = abs };
    Context ctx2 = *ctx;
    if (abs->tag == BasicBlock_TAG) {
        ctx2.dfs_stack = &dfs_entry;
    } else {
        // entry.num_backedges = 666;
    }

    const Node* body = get_abstraction_body(abs);
    switch (is_terminator(body)) {
        case NotATerminator:
        case LetMut_TAG: assert(false);
        case Let_TAG: {
            const Node* old_instr = get_let_instruction(body);
            const Node* new_instr = NULL;
            switch (is_instruction(old_instr)) {
                case NotAnInstruction: assert(false);
                case Instruction_If_TAG:
                case Instruction_Loop_TAG:
                case Instruction_Match_TAG: error("not supposed to exist in IR at this stage");
                case Instruction_LeafCall_TAG:
                case Instruction_PrimOp_TAG: {
                    new_instr = recreate_node_identity(&ctx2.rewriter, old_instr);
                    break;
                }
                case Instruction_IndirectCall_TAG: error("TODO: bail");
                // let(control(body), tail)
                // var phi = undef; level = N+1; structurize[body]; if (level == N+1, _ => tail(phi));
                case Instruction_Control_TAG: {
                    const Node* old_control_body = old_instr->payload.control.inside;
                    assert(old_control_body->tag == AnonLambda_TAG);
                    Nodes old_control_params = get_abstraction_params(old_control_body);
                    assert(old_control_params.count == 1);

                    Context control_ctx = *ctx;
                    control_ctx.dfs_stack = NULL;
                    ControlEntry control_entry = {
                        .parent = control_ctx.control_stack,
                        .old_token = first(old_control_params),
                        .depth = 0,
                    };
                    control_ctx.control_stack = &control_entry;
                    if (control_entry.parent)
                        control_entry.depth = control_entry.parent->depth + 1;

                    
                }
            }
        }
        case Jump_TAG: {
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            return handle_bb_callsite(&ctx2, bb, body->payload.jump.target, body->payload.jump.args);
        }
        // br(cond, true_bb, false_bb, args)
        // becomes
        // let(if(cond, _ => handle_bb_callsite[true_bb, args], _ => handle_bb_callsite[false_bb, args]), _ => unreachable)
        case Branch_TAG: {
            const Node* condition = rewrite_node(&ctx2.rewriter, body->payload.branch.branch_condition);

            Node* if_true_lam = lambda(ctx2.rewriter.dst_module, empty(ctx->rewriter.dst_arena));
            BodyBuilder* if_true_bb = begin_body(ctx2.rewriter.dst_module);
            if_true_lam->payload.anon_lam.body = handle_bb_callsite(&ctx2, if_true_bb, body->payload.branch.true_target, body->payload.branch.args);

            Node* if_false_lam = lambda(ctx2.rewriter.dst_module, empty(ctx->rewriter.dst_arena));
            BodyBuilder* if_false_bb = begin_body(ctx2.rewriter.dst_module);
            if_false_lam->payload.anon_lam.body = handle_bb_callsite(&ctx2, if_false_bb, body->payload.branch.false_target, body->payload.branch.args);

            const Node* instr = if_instr(ctx2.rewriter.dst_arena, (If) {
                .condition = condition,
                .yield_types = empty(ctx2.rewriter.dst_arena),
                .if_true = if_true_lam,
                .if_false = if_false_lam,
            });
            Node* unreach_lam = lambda(ctx2.rewriter.dst_module, empty(ctx2.rewriter.dst_arena));
            unreach_lam->payload.anon_lam.body = unreachable(ctx2.rewriter.dst_arena);
            return let(ctx2.rewriter.dst_arena, instr, unreach_lam);
        }
        case Switch_TAG: {
            error("TODO");
        }
        case Join_TAG: {

        }

        case Return_TAG:
        case Unreachable_TAG: return recreate_node_identity(&ctx2.rewriter, body);

        case TailCall_TAG: error("TODO: bail")

        case Terminator_MergeBreak_TAG:
        case Terminator_MergeContinue_TAG:
        case Terminator_MergeSelection_TAG: error("Only control nodes are tolerated here.")
    }
}

void opt_restructurize(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    LARRAY(TodoEntry, todo, get_module_declarations(src).count);
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
    };
    rewrite_module(&ctx.rewriter);

    for (size_t i = 0; i < ctx.todo_count; i++) {
        Context ctx2 = ctx;
        ctx2.lower = true;
        ctx2.dfs_stack = NULL;
        ctx2.control_stack = NULL;
        ctx2.todos = NULL;
        ctx2.todo_count = 0;
        ctx.todos[i].new->payload.fun.body = structure(&ctx2, ctx.todos[i].old);
    }

    destroy_rewriter(&ctx.rewriter);
}
