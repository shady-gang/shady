#include "../implem.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

struct Instr2BBRewriter {
    Rewriter rewriter;
    struct Dict* done;
};

static const Node* handle_block(struct Instr2BBRewriter* rewriter, const Node* node, size_t start, Node** join) {
    assert(node->tag == Block_TAG);
    IrArena* dst_arena = rewriter->rewriter.dst_arena;
    const Block* old_block = &node->payload.block;
    struct List* accumulator = new_list(const Node*);
    assert(start <= old_block->instructions.count);
    for (size_t i = start; i < old_block->instructions.count; i++) {
        const Node* instruction = old_block->instructions.nodes[i];
        switch (instruction->tag) {
            case Let_TAG: {
                const Node* imported = recreate_node_identity(&rewriter->rewriter, instruction);
                append_list(const Node*, accumulator, imported);
                break;
            }
            case IfInstr_TAG: {
                bool has_false_branch = instruction->payload.if_instr.if_false;

                Node* rest = fn(dst_arena, true, unique_name(dst_arena, "if_join"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL));
                Node* true_branch = fn(dst_arena, true, unique_name(dst_arena, "if_true"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL));
                Node* false_branch = has_false_branch ? fn(dst_arena, true, unique_name(dst_arena, "if_false"), nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL)) : NULL;

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

static const Node* instr2bb_process(struct Instr2BBRewriter* rewriter, const Node* node) {
    IrArena* dst_arena = rewriter->rewriter.dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Node** already_done = find_value_dict(const Node*, Node*, rewriter->done, node);
            if (already_done)
                return *already_done;

            Node* fun = fn(dst_arena, node->payload.fn.is_continuation, string(dst_arena, node->payload.fn.name), rewrite_nodes(&rewriter->rewriter, node->payload.fn.params), rewrite_nodes(&rewriter->rewriter, node->payload.fn.return_types));
            bool r = insert_dict_and_get_result(const Node*, Node*, rewriter->done, node, fun);
            assert(r && "insertion of fun failed - the dict isn't working as it should");

            fun->payload.fn.block = instr2bb_process(rewriter, node->payload.fn.block);
            return fun;
        }
        case Block_TAG: return handle_block(rewriter, node, 0, NULL);
        default: return recreate_node_identity(&rewriter->rewriter, node);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* instr2bb(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct Instr2BBRewriter ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) instr2bb_process,
        },
        .done = done
    };

    const Node* rewritten = instr2bb_process(&ctx, src_program);

    destroy_dict(done);
    return rewritten;
}
