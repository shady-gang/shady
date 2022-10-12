#include "body_builder.h"
#include "rewrite.h"
#include "fold.h"
#include "log.h"
#include "ir_private.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

BodyBuilder* begin_body(IrArena* arena) {
    BodyBuilder* builder = malloc(sizeof(BodyBuilder));
    *builder = (BodyBuilder) {
        .arena = arena,
        .list = new_list(const Node*)
    };
    return builder;
}

void append_body(BodyBuilder* builder, const Node* instruction) {
    append_list(const Node*, builder->list, instruction);
}

void copy_instrs(BodyBuilder* builder, Nodes instructions) {
    for (size_t i = 0; i < instructions.count; i++)
        append_body(builder, instructions.nodes[i]);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct {
    Rewriter rewriter;
    struct Dict* in_use;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    if (is_instruction(node) || is_terminator(node))
        return recreate_node_identity(&ctx->rewriter, node);

    if (is_declaration(node->tag))
        return node;

    if (node->tag == Variable_TAG) {
        const Node* found = search_processed(&ctx->rewriter, node);
        if (found) node = found;
    } else {
        node = recreate_node_identity(&ctx->rewriter, node);
    }

    if (ctx->rewriter.dst_arena->config.allow_fold)
        node = resolve_known_vars(node, true);
    if (node->tag == Variable_TAG && node->payload.var.instruction)
        insert_set_get_result(const Node*, ctx->in_use, node->payload.var.instruction);

    return node;
    //return recreate_node_identity(&ctx->rewriter, node);
}

static bool has_side_effects(const Node* instruction) {
    if (is_value(instruction))
        return false;

    switch (instruction->tag) {
        case If_TAG:
        case Match_TAG:
        case Loop_TAG: return true; // TODO: check contents !
        case Call_TAG: return true; // TODO: maybe one day track side effects...
        case PrimOp_TAG: return has_primop_got_side_effects(instruction->payload.prim_op.op);
        default: error("Not an instruction")
    }
    SHADY_UNREACHABLE;
}

const Node* finish_body(BodyBuilder* builder, const Node* terminator) {
    struct List* folded_list = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    Context ctx = {
        .rewriter = {
            .dst_arena = builder->arena,
            .src_arena = builder->arena,
            .rewrite_fn = (RewriteFn) process_node,
            .processed = done,
        },
        .in_use = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };

    for (size_t i = 0; i < entries_count_list(builder->list); i++) {
        const Node* instruction = read_list(const Node*, builder->list)[i];
        instruction = process_node(&ctx, instruction);
        append_list(const Node*, folded_list, instruction);
    }

    const Node* nterminator = process_node(&ctx, terminator);

    destroy_list(builder->list);

    struct List* final_list = new_list(const Node*);

    for (size_t i = 0; i < entries_count_list(folded_list); i++) {
        const Node* instruction = read_list(const Node*, folded_list)[i];

        const Node* actual_instruction = instruction;
        if (instruction->tag == Let_TAG)
            actual_instruction = actual_instruction->payload.let.instruction;

        // we keep instructions that have useful results, calls and primops tagged as having side effects
        if (find_key_dict(const Node*, ctx.in_use, actual_instruction) || has_side_effects(actual_instruction))
            append_list(const Node*, final_list, instruction);
    }

    const Node* finished = body(builder->arena, (Body) {
        .instructions = nodes(builder->arena, entries_count_list(final_list), read_list(const Node*, final_list)),
        .terminator = nterminator
    });

    destroy_dict(done);
    destroy_dict(ctx.in_use);
    destroy_list(folded_list);
    destroy_list(final_list);
    free(builder);
    return finished;
}
