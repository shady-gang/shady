#include "shady/ir.h"

#include "../rewrite.h"
#include "../type.h"
#include "../log.h"
#include "../portability.h"

#include "../transform/ir_gen_helpers.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    struct Dict* assigned_fn_ptrs;
    FnPtr next_fn_ptr;

    struct List* new_decls;
} Context;

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

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
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case GlobalVariable_TAG:
        case Constant_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            recreate_decl_body_identity(&ctx->rewriter, old, new);
            return new;
        }
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, old);

            fun->payload.fn.block = lower_callf_process(ctx, old->payload.fn.block);
            return fun;
        }
        case Block_TAG: {
            // this may miss call instructions...
            BlockBuilder* instructions = begin_block(dst_arena);
            for (size_t i = 0; i < old->payload.block.instructions.count; i++)
                append_block(instructions, rewrite_node(&ctx->rewriter, old->payload.block.instructions.nodes[i]));

            const Node* terminator = old->payload.block.terminator;

            switch (terminator->tag) {
                case Return_TAG: {
                    Nodes nargs = rewrite_nodes(&ctx->rewriter, terminator->payload.fn_ret.values);
                    const Type* return_type = fn_type(dst_arena, (FnType) {
                        .is_continuation = false,
                        .param_types = extract_types(dst_arena, nargs),
                        .return_types = nodes(dst_arena, 0, NULL)
                    });
                    const Type* return_address_type = ptr_type(dst_arena, (PtrType) {
                        .address_space = AsProgramCode,
                        .pointed_type = return_type
                    });

                    // Pop the return address and the convergence token, and join on that
                    const Node* return_address = gen_pop_value_stack(instructions, "return_addr", return_address_type);
                    const Node* return_convtok = gen_pop_value_stack(instructions, "return_convtok", mask_type(dst_arena));

                    // Join up at the return address
                    terminator = join(dst_arena, (Join) {
                        .is_indirect = true,
                        .join_at = return_address,
                        .args = nargs,
                        .desired_mask = return_convtok,
                    });
                    break;
                }
                case Callc_TAG: {
                    assert(terminator->payload.callc.is_return_indirect && "make sure lower_callc runs first !");
                    // put the return address and a convergence token in the stack
                    const Node* conv_token = gen_primop(instructions, (PrimOp) { .op = get_mask_op, .operands = nodes(dst_arena, 0, NULL)}).nodes[0];
                    gen_push_value_stack(instructions, conv_token);
                    gen_push_value_stack(instructions, terminator->payload.callc.ret_cont);
                    // Branch to the callee
                    terminator = branch(dst_arena, (Branch) {
                        .branch_mode = BrTailcall,
                        .yield = false,
                        .target = rewrite_node(&ctx->rewriter, terminator->payload.callc.callee),
                        .args = rewrite_nodes(&ctx->rewriter, terminator->payload.callc.args),
                    });
                    break;
                }
                default: terminator = lower_callf_process(ctx, terminator); break;
            }
            return finish_block(instructions, terminator);
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

const Node* lower_callf(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* new_decls_list = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct Dict* ptrs = new_dict(const Node*, FnPtr, (HashFn) hash_node, (CmpFn) compare_node);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) lower_callf_process,
            .rewrite_decl_body = NULL,
            .processed = done,
        },
        .assigned_fn_ptrs = ptrs,
        .next_fn_ptr = 1,

        .new_decls = new_decls_list,
    };

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);
    Nodes new_decls = rewritten->payload.root.declarations;
    for (size_t i = 0; i < entries_count_list(new_decls_list); i++) {
        new_decls = append_nodes(dst_arena, new_decls, read_list(const Node*, new_decls_list)[i]);
    }
    rewritten = root(dst_arena, (Root) {
        .declarations = new_decls
    });

    destroy_list(new_decls_list);

    destroy_dict(done);
    destroy_dict(ptrs);
    return rewritten;
}
