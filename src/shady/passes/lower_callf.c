#include "shady/ir.h"

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
} Context;

static const Node* lower_callf_process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case Lambda_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, old);
            Context ctx2 = *ctx;
            if (fun->payload.lam.tier == FnTier_Function)
                ctx2.disable_lowering = lookup_annotation_with_string_payload(old, "DisablePass", "lower_callf");
            fun->payload.lam.body = lower_callf_process(&ctx2, old->payload.lam.body);
            return fun;
        }
        case Return_TAG: {
            Nodes nargs = rewrite_nodes(&ctx->rewriter, old->payload.fn_ret.values);
            BodyBuilder* bb = begin_body(dst_arena);
            // Pop the convergence token, and join on that
            const Node* return_convtok = gen_pop_value_stack(bb, "return_convtok", join_point_type(dst_arena, (JoinPointType) { .yield_types = extract_types(dst_arena, nargs) }));
            // This effectively asserts uniformity
            return_convtok = gen_primop_ce(bb, subgroup_broadcast_first_op, 1, (const Node* []) { return_convtok });
            // Join up at the return address
            return finish_body(bb, join(dst_arena, (Join) {
                .join_point = return_convtok,
                .args = nargs,
            }));
        }
        case Let_TAG: {
            if (ctx->disable_lowering)
                return recreate_node_identity(&ctx->rewriter, old);

            const Node* old_instruction = old->payload.let.instruction;
            const Node* new_instruction = NULL;
            const Node* old_tail = old->payload.let.tail;
            // we convert calls to tail-calls within a control
            // let(call(...), ret_fn) to let(control(jp => save(jp); tailcall(...)), ret_fn)
            if (old_instruction->tag == Call_TAG && old_instruction->payload.call_instr.is_indirect) {
                assert(old_tail->payload.lam.tier == FnTier_Function);

                Nodes returned_types = rewrite_nodes(&ctx->rewriter, extract_variable_types(dst_arena, &old_tail->payload.lam.params));
                const Type* jpt = join_point_type(dst_arena, (JoinPointType) { .yield_types = returned_types });
                const Node* jp = var(dst_arena, jpt, "fn_return_point");
                Node* control_insides = lambda(dst_arena, nodes(dst_arena, 1, (const Node*[]) { jp }));
                BodyBuilder* instructions = begin_body(dst_arena);
                // yeet the join point on the stack
                gen_push_value_stack(instructions, jp);
                // tail-call to the target immediately afterwards
                control_insides->payload.lam.body = finish_body(instructions, tail_call(dst_arena, (TailCall) {
                    .target = rewrite_node(&ctx->rewriter, old_instruction->payload.call_instr.callee),
                    .args = rewrite_nodes(&ctx->rewriter, old_instruction->payload.call_instr.args),
                }));

                new_instruction = control(dst_arena, (Control) { .yield_types = returned_types, .inside = control_insides });
            }

            if (!new_instruction)
                new_instruction = rewrite_node(&ctx->rewriter, old_instruction);

            return let(dst_arena, false, new_instruction, rewrite_node(&ctx->rewriter, old_tail));
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

const Node* lower_callf(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    Context ctx = {
        .rewriter = create_rewriter(src_arena, dst_arena, (RewriteFn) lower_callf_process),
        .disable_lowering = false,
    };
    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);
    destroy_rewriter(&ctx.rewriter);
    return rewritten;
}
