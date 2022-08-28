#include "shady/ir.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include "../transform/ir_gen_helpers.h"
#include "../analysis/free_variables.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    struct Dict* spilled;
    struct List* new_fns;
    struct List* todo;
} Context;

typedef struct {
    const Node* old_block;
    const Node** new_block;
} Todo;

static const Node* lift_continuation_into_function(Context* ctx, const Node* cont, Nodes* callsite_instructions) {
    assert(cont->tag == Function_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    // TODO we assume cont is only called once - otherwise we'll have duplication at every callsite
    //const Node* already_done = search_processed(&ctx->rewriter, cont);
    //if (already_done)
    //    return already_done;

    // Create a new context
    // TODO: ensure this context has the top-level decls but NOT the continuations we might have previously encountered !
    Context new_ctx = *ctx;
    struct Dict* new_dict = clone_dict(ctx->rewriter.processed);
    new_ctx.rewriter.processed = new_dict;

    // Create and register new parameters for the lifted continuation
    Nodes new_params = recreate_variables(&ctx->rewriter, cont->payload.fn.params);
    for (size_t i = 0; i < new_params.count; i++)
        register_processed(&new_ctx.rewriter, cont->payload.fn.params.nodes[i], new_params.nodes[i]);

    // Compute the live stuff we'll need
    struct List* recover_context = compute_free_variables(cont);
    size_t recover_context_size = entries_count_list(recover_context);

    // Save what we'll need later
    for (size_t i = 0; i < recover_context_size; i++) {
        const Variable* var = &read_list(const Node*, recover_context)[i]->payload.var;
        // TODO: shouldn't we recreate everything (ie the type too) here ?
        // We should at least be consistent
        const Node* args[] = { extract_operand_type(var->type), read_list(const Node*, recover_context)[i] };
        const Node* save_instruction = prim_op(dst_arena, (PrimOp) {
            .op = push_stack_op,
            .operands = nodes(dst_arena, 2, args)
        });
        *callsite_instructions = append_nodes(dst_arena, *callsite_instructions, save_instruction);
    }

    // Recover that stuff inside the new block
    Nodes new_block_instructions = nodes(dst_arena, 0, NULL);
    for (size_t i = recover_context_size - 1; i < recover_context_size; i--) {
        const Node* ovar = read_list(const Node*, recover_context)[i];
        const char* output_names[] = {ovar->payload.var.name };

        const Type* type = rewrite_node(&ctx->rewriter, extract_operand_type(ovar->payload.var.type));

        const Node* let_load = let(dst_arena, prim_op(dst_arena, (PrimOp) {
            .op = pop_stack_op,
            .operands = nodes(dst_arena, 1, (const Node* []) {type})
        }), 1, output_names);
        insert_dict(const Node*, const Node*, ctx->spilled, ovar, let_load->payload.let.variables.nodes[0]);
        // register_processed(&new_ctx.rewriter, ovar, let_load->payload.let.variables.nodes[0]);
        new_block_instructions = append_nodes(dst_arena, new_block_instructions, let_load);
    }

    // Write out the rest of the new block using this fresh context
    for (size_t i = 0; i < cont->payload.fn.block->payload.block.instructions.count; i++) {
        const Node* new_instruction = rewrite_node(&new_ctx.rewriter, cont->payload.fn.block->payload.block.instructions.nodes[i]);
        new_block_instructions = append_nodes(dst_arena, new_block_instructions, new_instruction);
    }
    const Node* new_terminator = rewrite_node(&new_ctx.rewriter, cont->payload.fn.block->payload.block.terminator);

    clear_dict(ctx->spilled);

    // Keep annotations the same
    Nodes annotations = rewrite_nodes(&ctx->rewriter, cont->payload.fn.annotations);

    Node* new_fn = fn(dst_arena, annotations, cont->payload.fn.name, false, new_params, nodes(dst_arena, 0, NULL));
    new_fn->payload.fn.block = block(dst_arena, (Block) {
        .instructions = new_block_instructions,
        .terminator = new_terminator,
    });
    append_list(const Node*, ctx->new_fns, new_fn);

    destroy_list(recover_context);
    destroy_dict(new_dict);
    return new_fn;
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node** spilled = find_value_dict(const Node*, const Node*, ctx->spilled, node);
    if (spilled) return *spilled;

    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);

            const Node* old_block = node->payload.fn.block;
            // If the block has a callc, delay
            if (old_block->payload.block.terminator->tag == Callc_TAG) {
                assert(!old_block->payload.block.terminator->payload.callc.is_return_indirect && "Return continuations should be function pointers at this stage.");
                Todo t = { old_block, &new->payload.fn.block };
                debug_print("Found a callc - adding to todo list\n");
                append_list(Todo, ctx->todo, t);
                return new;
            }

            recreate_decl_body_identity(&ctx->rewriter, node, new);
            return new;
        }
        // leave other declarations alone
        case GlobalVariable_TAG:
        case Constant_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            recreate_decl_body_identity(&ctx->rewriter, node, new);
            return new;
        }
        case Root_TAG: error("illegal node");
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

static void handle_todo_entry(Context* ctx, Todo todo) {
    assert(todo.old_block->payload.block.terminator->tag == Callc_TAG);
    const Callc* old_callc = &todo.old_block->payload.block.terminator->payload.callc;

    debug_print("Processing callc ret_cont: ");
    debug_node(old_callc->join_at);
    debug_print("\n");

    Nodes instructions = todo.old_block->payload.block.instructions;
    const Node* lifted_fn = lift_continuation_into_function(ctx, old_callc->join_at, &instructions);
    *todo.new_block = block(ctx->rewriter.dst_arena, (Block) {
        .instructions = rewrite_nodes(&ctx->rewriter, instructions),
        .terminator = callc(ctx->rewriter.dst_arena, (Callc) {
            .is_return_indirect = true,
            .callee = process_node(ctx, old_callc->callee),
            .args = rewrite_nodes(&ctx->rewriter, old_callc->args),
            .join_at = fn_addr(ctx->rewriter.dst_arena, (FnAddr) {.fn = lifted_fn}),
        })
    });
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* lower_callc(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* new_decls_list = new_list(const Node*);
    struct List* todos = new_list(Todo);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct Dict* spilled = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) process_node,
            .processed = done,
        },
        .new_fns = new_decls_list,
        .todo = todos,
        .spilled = spilled,
    };

    assert(src_program->tag == Root_TAG);

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);
    debug_print("Size of processed after initial rewrite: %d\n", entries_count_dict(done));

    while (entries_count_list(todos) > 0) {
        Todo entry = pop_last_list(Todo, todos);
        handle_todo_entry(&ctx, entry);
    }

    Nodes new_decls = rewritten->payload.root.declarations;
    for (size_t i = 0; i < entries_count_list(new_decls_list); i++) {
        new_decls = append_nodes(dst_arena, new_decls, read_list(const Node*, new_decls_list)[i]);
    }
    rewritten = root(dst_arena, (Root) {
        .declarations = new_decls
    });

    destroy_list(new_decls_list);
    destroy_list(todos);
    destroy_dict(spilled);
    destroy_dict(done);
    return rewritten;
}
