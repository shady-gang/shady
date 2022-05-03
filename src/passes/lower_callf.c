#include "shady/ir.h"

#include "../rewrite.h"
#include "../type.h"
#include "../log.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    struct Dict* assigned_fn_ptrs;
    FnPtr next_fn_ptr;
} Context;

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static void gen_push_values_stack(IrArena* arena, Nodes* instructions, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        *instructions = append_nodes(arena, *instructions, let(arena, (Let) {
            .variables = nodes(arena, 0, NULL),
            .instruction = prim_op(arena, (PrimOp) {
                .op = push_stack_op,
                .operands = nodes(arena, 2, (const Node*[]) { without_qualifier(value->type), value })
            })
        }));
    }
}

static void gen_push_fn_stack(IrArena* arena, Nodes* instructions, const Node* fn_ptr) {
    const Type* ret_param_type = int_type(arena);

    *instructions = append_nodes(arena, *instructions, let(arena, (Let) {
        .variables = nodes(arena, 0, NULL),
        .instruction = prim_op(arena, (PrimOp) {
            .op = push_stack_uniform_op,
            .operands = nodes(arena, 2, (const Node*[]) { ret_param_type, fn_ptr })
        })
    }));
}

static const Node* gen_pop_fn_stack(IrArena* arena, Nodes* instructions) {
    const Type* ret_param_type = int_type(arena);
    const Type* q_ret_param_type = qualified_type(arena, (QualifiedType) {.type = ret_param_type, .is_uniform = true});

    const Node* ret_tmp_vars[] = { var(arena, q_ret_param_type, "ret_tmp")};
    *instructions = append_nodes(arena, *instructions, let(arena, (Let) {
        .variables = nodes(arena, 1, ret_tmp_vars),
        .instruction = prim_op(arena, (PrimOp) {
            .op = pop_stack_uniform_op,
            .operands = nodes(arena, 1, (const Node*[]) { ret_param_type })
        })
    }));
    return ret_tmp_vars[0];
}

static const Node* fn_ptr_as_value(IrArena* arena, FnPtr ptr) {
    return int_literal(arena, (IntLiteral) {
        .value = ptr
    });
}

static const Node* callee_to_ptr(Context* ctx, const Node* callee) {
    const Type* ret_param_type = int_type(ctx->rewriter.dst_arena);

    if (callee->tag != Function_TAG) {
        assert(is_subtype(ret_param_type, without_qualifier(callee)));
        return callee;
    }

    FnPtr* found = find_value_dict(const Node*, FnPtr, ctx->assigned_fn_ptrs, callee);
    if (found) return fn_ptr_as_value(ctx->rewriter.dst_arena, *found);

    FnPtr ptr = ctx->next_fn_ptr++;
    bool r = insert_dict_and_get_result(const Node*, FnPtr, ctx->assigned_fn_ptrs, callee, ptr);
    assert(r);
    return fn_ptr_as_value(ctx->rewriter.dst_arena, ptr);
}

static const Node* lower_callf_process(Context* ctx, const Node* old) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case GlobalVariable_TAG:
        case Constant_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            recreate_decl_body_identity(&ctx->rewriter, old, new);
            return new;
        }
        case Function_TAG: {
             //Node* fun = fn(dst_arena, old->payload.fn.atttributes, old->payload.fn.name, recreate_variables(&ctx->rewriter, old->payload.fn.params), nodes(dst_arena, 0, NULL));
             Node* fun = fn(dst_arena, old->payload.fn.atttributes, old->payload.fn.name, old->payload.fn.params, nodes(dst_arena, 0, NULL));
             register_processed(&ctx->rewriter, old, fun);
             for (size_t i = 0; i < fun->payload.fn.params.count; i++)
                 register_processed(&ctx->rewriter, old->payload.fn.params.nodes[i], fun->payload.fn.params.nodes[i]);
             fun->payload.fn.block = lower_callf_process(ctx, old->payload.fn.block);
             return fun;
        }
        case Block_TAG: {
            // this may miss call instructions...
            //Nodes instructions = rewrite_nodes(&ctx->rewriter, old->payload.block.instructions);
            Nodes instructions = old->payload.block.instructions;

            const Node* terminator = old->payload.block.terminator;

            switch (terminator->tag) {
                case Return_TAG: {
                    Nodes ret_values = terminator->payload.fn_ret.values;

                    if (ret_values.count > 0) {
                        // Pop the old return address off the stack
                        const Node* ret_tmp_var = gen_pop_fn_stack(dst_arena, &instructions);
                        // Push the return values as arguments to the return function
                        gen_push_values_stack(dst_arena, &instructions, ret_values);
                        // Push back the return address on the now-top of the stack
                        gen_push_fn_stack(dst_arena, &instructions, ret_tmp_var);
                    }
                    // Kill the function
                    terminator = fn_ret(dst_arena, (Return) {
                        .fn = NULL,
                        .values = nodes(dst_arena, 0, NULL)
                    });
                    break;
                }
                case Callf_TAG: {
                    // put the return address at the bottom
                    gen_push_fn_stack(dst_arena, &instructions, callee_to_ptr(ctx, terminator->payload.callf.ret_fn));
                    // push the arguments to the next call, then the target ptr
                    gen_push_values_stack(dst_arena, &instructions, terminator->payload.callf.args);
                    gen_push_fn_stack(dst_arena, &instructions, callee_to_ptr(ctx, terminator->payload.callf.callee));
                    // Kill the function
                    terminator = fn_ret(dst_arena, (Return) {
                        .fn = NULL,
                        .values = nodes(dst_arena, 0, NULL)
                    });
                    break;
                }
                default: terminator = lower_callf_process(ctx, terminator); break;
            }
            return block(dst_arena, (Block) {
                .instructions = instructions,
                .terminator = terminator
            });
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

/*static void lower_callf_decl_body(Context* ctx, const Node* old, Node* new) {
    switch (new->tag) {
        case Function_TAG: {
            for (size_t i = 0; i < new->payload.fn.params.count; i++)
                register_processed(&ctx->rewriter, old->payload.fn.params.nodes[i], new->payload.fn.params.nodes[i]);
            new->payload.fn.block = lower_callf_process(ctx, old->payload.fn.block);
            break;
        }
        case GlobalVariable_TAG:
        case Constant_TAG: {
            recreate_decl_body_identity(&ctx->rewriter, old, new);
            break;
        }
        default: error("not a decl");
    }
}*/

const Node* lower_callf(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct Dict* ptrs = new_dict(const Node*, FnPtr, (HashFn) hash_node, (CmpFn) compare_node);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) lower_callf_process,
            //.rewrite_decl_body = (RewriteFnMut) recreate_decl_body_identity,
            .rewrite_decl_body = NULL,
            .processed = done,
        },
        .assigned_fn_ptrs = ptrs,
        .next_fn_ptr = 0,
    };

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);

    destroy_dict(done);
    destroy_dict(ptrs);
    return rewritten;
}
