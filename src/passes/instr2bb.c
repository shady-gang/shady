#include "shady/ir.h"

#include "../analysis/scope.h"
#include "../analysis/free_variables.h"
#include "../log.h"
#include "../type.h"
#include "../local_array.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    struct Dict* done;
    struct List* new_fns;
} Context;

static const Node* instr2bb_process(Context* rewriter, const Node* node);

static FnAttributes cont_attr = {
    .is_continuation = true,
    .entry_point_type = NotAnEntryPoint
};

static const Node* handle_block(Context* ctx, const Node* node, size_t start, Node** outer_join) {
    assert(node->tag == Block_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    assert(dst_arena == ctx->rewriter.src_arena);
    const Block* old_block = &node->payload.block;
    struct List* accumulator = new_list(const Node*);
    assert(start <= old_block->instructions.count);
    for (size_t i = start; i < old_block->instructions.count; i++) {
        const Node* let_node = old_block->instructions.nodes[i];
        const Node* instr = let_node->payload.let.instruction;
        switch (instr->tag) {
            case If_TAG: {
                // TODO handle yield types !
                bool has_false_branch = instr->payload.if_instr.if_false;
                Nodes yield_types = instr->payload.if_instr.yield_types;

                LARRAY(const Node*, rest_params, yield_types.count);
                for (size_t j = 0; j < yield_types.count; j++) {
                    rest_params[j] = let_node->payload.let.variables.nodes[j];
                }

                Node* join_cont = fn(dst_arena, cont_attr, unique_name(dst_arena, "if_join"), nodes(dst_arena, yield_types.count, rest_params), nodes(dst_arena, 0, NULL));
                Node* true_branch = fn(dst_arena, cont_attr, unique_name(dst_arena, "if_true"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL));
                Node* false_branch = has_false_branch ? fn(dst_arena, cont_attr, unique_name(dst_arena, "if_false"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL)) : NULL;

                true_branch->payload.fn.block = handle_block(ctx,  instr->payload.if_instr.if_true, 0, &join_cont);
                if (has_false_branch)
                    false_branch->payload.fn.block = handle_block(ctx,  instr->payload.if_instr.if_false, 0, &join_cont);
                join_cont->payload.fn.block = handle_block(ctx, node, i + 1, outer_join);

                Nodes instructions = nodes(dst_arena, entries_count_list(accumulator), read_list(const Node*, accumulator));
                destroy_list(accumulator);
                const Node* branch_t = branch(dst_arena, (Branch) {
                    .condition = instr->payload.if_instr.condition,
                    .true_target = true_branch,
                    .false_target = has_false_branch ? false_branch : join_cont,
                });
                return block(dst_arena, (Block) {
                    .instructions = instructions,
                    .terminator = branch_t
                });
            }
            case Call_TAG: {
                const Node* callee = instr->payload.call_instr.callee;
                assert(get_qualifier(callee->type) == Uniform);
                const Type* callee_type = without_qualifier(callee->type);
                assert(callee_type->tag == FnType_TAG);

                size_t args_count = instr->payload.call_instr.args.count;

                FnAttributes rest_attrs = {
                    .is_continuation = false,
                    .entry_point_type = NotAnEntryPoint,
                };

                Node* rest = fn(dst_arena, rest_attrs, unique_name(dst_arena, "call_ret"), let_node->payload.let.variables, nodes(dst_arena, 0, NULL));
                append_list(const Node*, ctx->new_fns, rest);
                rest->payload.fn.block = handle_block(ctx, node, i + 1, outer_join);

                // analyse the live stuff in rest and push so we can recover it
                struct List* recover_context = compute_free_variables(rest);
                size_t recover_context_size = entries_count_list(recover_context);

                // prepare to rewrite the rest block to include the context recovery instructions
                const Block* orest_block = &rest->payload.fn.block->payload.block;
                struct List* prepended_rest_instrs = new_list(const Node*);

                // add save instructions to the origin BB
                for (size_t j = 0; j < recover_context_size; j++) {
                    const Variable* var = &read_list(const Node*, recover_context)[j]->payload.var;
                    const Node* args[] = {without_qualifier(var->type), read_list(const Node*, recover_context)[j] };
                    const Node* save_instr = let(dst_arena, (Let) {
                        .variables = nodes(dst_arena, 0, NULL),
                        .instruction = prim_op(dst_arena, (PrimOp) {
                            .op = push_stack_op,
                            .operands = nodes(dst_arena, 2, args)
                        })
                    });
                    append_list(const Node*, accumulator, save_instr);
                }

                // prepend load instructions to the dest BB
                for (size_t j = recover_context_size - 1; j < recover_context_size; j--) {
                    const Variable* var = &read_list(const Node*, recover_context)[j]->payload.var;
                    const Node* vars[] = {read_list(const Node*, recover_context)[j] };
                    const Node* args[] = {without_qualifier(var->type) };
                    const Node* load_instr = let(dst_arena, (Let) {
                        .variables = nodes(dst_arena, 1, vars),
                        .instruction = prim_op(dst_arena, (PrimOp) {
                            .op = pop_stack_op,
                            .operands = nodes(dst_arena, 1, args)
                        })
                    });
                    append_list(const Node*, prepended_rest_instrs, load_instr);
                }

                for (size_t j = 0; j < orest_block->instructions.count; j++) {
                    const Node* oinstr = orest_block->instructions.nodes[j];
                    append_list(const Node*, prepended_rest_instrs, oinstr);
                }

                // Update the rest block accordingly
                rest->payload.fn.block = block(dst_arena, (Block) {
                    .instructions = nodes(dst_arena, entries_count_list(prepended_rest_instrs), read_list(const Node*, prepended_rest_instrs)),
                    .terminator = orest_block->terminator
                });

                destroy_list(recover_context);
                destroy_list(prepended_rest_instrs);

                Nodes instructions = nodes(dst_arena, entries_count_list(accumulator), read_list(const Node*, accumulator));
                destroy_list(accumulator);

                // TODO we probably want to emit a callc here and lower that later to a separate function in an optional pass
                return block(dst_arena, (Block) {
                    .instructions = instructions,
                    .terminator = callf(dst_arena, (Callf) {
                        .ret_fn = rest,
                        .callee = instr2bb_process(ctx, callee),
                        .args = nodes(dst_arena, args_count, instr->payload.call_instr.args.nodes)
                    })
                });
            }
            default: {
                const Node* imported = recreate_node_identity(&ctx->rewriter, let_node);
                append_list(const Node*, accumulator, imported);
                break;
            }
        }
    }

    const Node* old_terminator = old_block->terminator;
    const Node* new_terminator = NULL;
    switch (old_terminator->tag) {
        case Merge_TAG: {
            switch (old_terminator->payload.merge.what) {
                case Join: {
                    assert(old_terminator->payload.merge.what == Join);
                    assert(outer_join);
                    new_terminator = jump(dst_arena, (Jump) {
                        .target = *outer_join,
                        .args = nodes(dst_arena, old_terminator->payload.merge.args.count, old_terminator->payload.merge.args.nodes)
                    });
                    break;
                }
                // TODO handle other kind of merges
                case Continue:
                case Break: error("TODO")
                default: SHADY_UNREACHABLE;
            }
            break;
        }
        default: new_terminator = recreate_node_identity(&ctx->rewriter, old_terminator); break;
    }

    assert(new_terminator);
    Nodes instructions = nodes(dst_arena, entries_count_list(accumulator), read_list(const Node*, accumulator));
    destroy_list(accumulator);
    return block(dst_arena, (Block) {
        .instructions = instructions,
        .terminator = new_terminator
    });
}

static const Node* instr2bb_process(Context* rewriter, const Node* node) {
    IrArena* dst_arena = rewriter->rewriter.dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Node** already_done = find_value_dict(const Node*, Node*, rewriter->done, node);
            if (already_done)
                return *already_done;

            Node* fun = fn(dst_arena, node->payload.fn.atttributes, string(dst_arena, node->payload.fn.name), rewrite_nodes(&rewriter->rewriter, node->payload.fn.params), rewrite_nodes(&rewriter->rewriter, node->payload.fn.return_types));
            bool r = insert_dict_and_get_result(const Node*, Node*, rewriter->done, node, fun);
            assert(r && "insertion of fun failed - the dict isn't working as it should");

            fun->payload.fn.block = instr2bb_process(rewriter, node->payload.fn.block);
            return fun;
        }
        case Block_TAG: return handle_block(rewriter, node, 0, NULL);
        case Constant_TAG: return node;
        case Root_TAG: error("illegal node");
        default: return recreate_node_identity(&rewriter->rewriter, node);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* instr2bb(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* decls = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) instr2bb_process,
        },
        .done = done,
        .new_fns = decls,
    };

    assert(src_program->tag == Root_TAG);
    const Root* oroot = &src_program->payload.root;
    for (size_t i = 0; i < oroot->declarations.count; i++) {
        const Node* new_decl = instr2bb_process(&ctx, oroot->declarations.nodes[i]);
        append_list(const Node*, decls, new_decl);
    }

    const Node* rewritten = root(dst_arena, (Root) {
       .declarations = nodes(dst_arena, entries_count_list(decls), read_list(const Node*, decls))
    });

    destroy_list(decls);
    destroy_dict(done);
    return rewritten;
}
