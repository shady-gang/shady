#include "shady/ir.h"

#include "../analysis/scope.h"
#include "../analysis/free_variables.h"
#include "../log.h"
#include "../type.h"

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

static const Node* handle_block(Context* rewriter, const Node* node, size_t start, Node** join) {
    assert(node->tag == Block_TAG);
    IrArena* dst_arena = rewriter->rewriter.dst_arena;
    const Block* old_block = &node->payload.block;
    struct List* accumulator = new_list(const Node*);
    assert(start <= old_block->instructions.count);
    for (size_t i = start; i < old_block->instructions.count; i++) {
        const Node* instruction = old_block->instructions.nodes[i];
        switch (instruction->tag) {
            case Let_TAG: {
                if (instruction->payload.let.op == call_op) {
                    const Node* callee = instruction->payload.let.args.nodes[0];
                    assert(get_qualifier(callee->type) == Uniform);
                    const Type* callee_type = without_qualifier(callee->type);
                    assert(callee_type->tag == FnType_TAG);

                    size_t args_count = instruction->payload.let.args.count - 1;

                    FnAttributes rest_attrs = {
                        .is_continuation = false,
                        .entry_point_type = NotAnEntryPoint,
                    };
                    Node* rest = fn(dst_arena, rest_attrs, unique_name(dst_arena, "call_ret"), instruction->payload.let.variables, nodes(dst_arena, 0, NULL));
                    append_list(const Node*, rewriter->new_fns, rest);
                    rest->payload.fn.block = handle_block(rewriter, node, i + 1, join);

                    // analyse the live stuff in rest and push so we can recover it
                    Scope join_scope = build_scope(rest);
                    struct List* recover_context = compute_free_variables(&join_scope);
                    size_t recover_context_size = entries_count_list(recover_context);

                    // prepare to rewrite the rest block to include the context recovery instructions
                    const Block* orest_block = &rest->payload.fn.block->payload.block;
                    struct List* prepended_rest_instrs = new_list(const Node*);

                    // add save instructions to the origin BB
                    for (size_t j = 0; j < recover_context_size; j++) {
                        const Variable* var = &read_list(const Node*, recover_context)[j]->payload.var;
                        // const Node* vars[] = {read_list(const Node*, recover_context)[j] };
                        const Node* args[] = {without_qualifier(var->type), read_list(const Node*, recover_context)[j] };
                        const Node* save_instr = let(dst_arena, (Let) {
                            .variables = nodes(dst_arena, 0, NULL),
                            .op = push_stack_op,
                            .args = nodes(dst_arena, 2, args)
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
                            .op = pop_stack_op,
                            .args = nodes(dst_arena, 1, args)
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
                    dispose_scope(&join_scope);

                    Nodes instructions = nodes(dst_arena, entries_count_list(accumulator), read_list(const Node*, accumulator));
                    destroy_list(accumulator);
                    return block(dst_arena, (Block) {
                        .instructions = instructions,
                        .terminator = callf(dst_arena, (Callf) {
                            .ret_cont = rest,
                            .target = instr2bb_process(rewriter, callee),
                            .args = nodes(dst_arena, args_count, &instruction->payload.let.args.nodes[1])
                        })
                    });
                } else {
                    const Node* imported = recreate_node_identity(&rewriter->rewriter, instruction);
                    append_list(const Node*, accumulator, imported);
                    break;
                }
            }
            case IfInstr_TAG: {
                bool has_false_branch = instruction->payload.if_instr.if_false;

                Node* rest = fn(dst_arena, cont_attr, unique_name(dst_arena, "if_join"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL));
                Node* true_branch = fn(dst_arena, cont_attr, unique_name(dst_arena, "if_true"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL));
                Node* false_branch = has_false_branch ? fn(dst_arena, cont_attr, unique_name(dst_arena, "if_false"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL)) : NULL;

                true_branch->payload.fn.block = handle_block(rewriter,  instruction->payload.if_instr.if_true, 0, &rest);
                if (has_false_branch)
                    false_branch->payload.fn.block = handle_block(rewriter,  instruction->payload.if_instr.if_false, 0, &rest);
                rest->payload.fn.block = handle_block(rewriter, node, i + 1, join);

                Nodes instructions = nodes(dst_arena, entries_count_list(accumulator), read_list(const Node*, accumulator));
                destroy_list(accumulator);
                const Node* branch_t = branch(dst_arena, (Branch) {
                    .condition = instruction->payload.if_instr.condition,
                    .true_target = true_branch,
                    .false_target = has_false_branch ? false_branch : rest,
                });
                return block(dst_arena, (Block) {
                    .instructions = instructions,
                    .terminator = branch_t
                });
            }
            default: error("not an instruction");
        }
    }

    const Node* old_terminator = old_block->terminator;
    const Node* new_terminator = NULL;
    switch (old_terminator->tag) {
        case Join_TAG: {
            assert(join);
            new_terminator = jump(dst_arena, (Jump) {
                .target = *join,
                .args = nodes(dst_arena, 0, NULL)
            });
            break;
        }
        default: new_terminator = recreate_node_identity(&rewriter->rewriter, old_terminator); break;
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
        case Root_TAG: error("");
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
