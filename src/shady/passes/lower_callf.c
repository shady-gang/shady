#include "passes.h"

#include "../rewrite.h"
#include "../type.h"
#include "log.h"
#include "portability.h"

#include "../transform/ir_gen_helpers.h"

#include <assert.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;
    const Node* return_tok;
} Context;

static const Node* lower_callf_process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    if (old->tag == Function_TAG) {
        Node* fun = recreate_decl_header_identity(&ctx->rewriter, old);
        Context ctx2 = *ctx;
        ctx2.disable_lowering = lookup_annotation_with_string_payload(old, "DisablePass", "lower_callf");
        BodyBuilder* bb = begin_body(dst_arena);
        // Pop the convergence token
        ctx2.return_tok = bind_instruction(bb, gen_pop_value_stack(bb, join_point_type(dst_arena, (JoinPointType) { .yield_types = fun->payload.fun.return_types }))).nodes[0];
        // This effectively asserts uniformity
        ctx2.return_tok = gen_primop_ce(bb, subgroup_broadcast_first_op, 1, (const Node* []) { ctx2.return_tok });
        fun->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, old->payload.fun.body));
        return fun;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, old);

    switch (old->tag) {
        case Return_TAG: {
            Nodes nargs = rewrite_nodes(&ctx->rewriter, old->payload.fn_ret.args);
            // Join up at the return address instead of returning
            return join(dst_arena, (Join) {
                .join_point = ctx->return_tok,
                .args = nargs,
            });
        }
        case LetIndirect_TAG: {
            const Node* old_instruction = old->payload.let.instruction;
            const Node* new_instruction = NULL;
            const Node* old_tail = old->payload.let.tail;
            // we convert calls to tail-calls within a control
            // let_indirect(call(...), ret_fn) to let_indirect(control(jp => save(jp); tailcall(...)), ret_fn)
            if (old_instruction->tag == Call_TAG) {
                const Node* tail_type = extract_operand_type(old_tail->type);
                assert(tail_type->tag == FnType_TAG);
                Nodes returned_types = rewrite_nodes(&ctx->rewriter, tail_type->payload.fn_type.return_types);

                const Type* jpt = join_point_type(dst_arena, (JoinPointType) { .yield_types = returned_types });
                const Node* jp = var(dst_arena, jpt, "fn_return_point");
                Node* control_insides = lambda(dst_arena, nodes(dst_arena, 1, (const Node*[]) { jp }));
                BodyBuilder* instructions = begin_body(dst_arena);
                // yeet the join point on the stack
                gen_push_value_stack(instructions, jp);
                // tail-call to the target immediately afterwards
                control_insides->payload.anon_lam.body = finish_body(instructions, tail_call(dst_arena, (TailCall) {
                    .target = rewrite_node(&ctx->rewriter, old_instruction->payload.call_instr.callee),
                    .args = rewrite_nodes(&ctx->rewriter, old_instruction->payload.call_instr.args),
                }));

                new_instruction = control(dst_arena, (Control) { .yield_types = returned_types, .inside = control_insides });
            }

            if (!new_instruction)
                new_instruction = rewrite_node(&ctx->rewriter, old_instruction);

            return let(dst_arena, new_instruction, rewrite_node(&ctx->rewriter, old_tail));
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

void lower_callf(SHADY_UNUSED CompilerConfig* config,  Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) lower_callf_process),
        .disable_lowering = false,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
