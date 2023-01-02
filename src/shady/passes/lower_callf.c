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
} Context;

static const Node* lower_callf_process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    if (old->tag == Function_TAG) {
        Context ctx2 = *ctx;
        ctx2.disable_lowering = lookup_annotation(old, "Leaf");
        Node* fun = NULL;
        if (!ctx2.disable_lowering) {
            Nodes oparams = get_abstraction_params(old);
            Nodes nparams = recreate_variables(&ctx->rewriter, oparams);
            register_processed_list(&ctx->rewriter, oparams, nparams);
            Nodes nannots = rewrite_nodes(&ctx->rewriter, old->payload.fun.annotations);
            fun = function(ctx->rewriter.dst_module, nparams, get_abstraction_name(old), nannots, empty(dst_arena));
            register_processed(&ctx->rewriter, old, fun);
        } else
            fun = recreate_decl_header_identity(&ctx->rewriter, old);
        fun->payload.fun.body = rewrite_node(&ctx2.rewriter, old->payload.fun.body);
        return fun;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, old);

    switch (old->tag) {
        case Return_TAG: {
            Nodes nargs = rewrite_nodes(&ctx->rewriter, old->payload.fn_ret.args);

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            const Node* return_jp = first(bind_instruction(bb, gen_pop_value_stack(bb, join_point_type(dst_arena, (JoinPointType) { .yield_types = strip_qualifiers(dst_arena, get_values_types(dst_arena, nargs)) }))));
            return_jp = gen_primop_ce(bb, subgroup_broadcast_first_op, 1, (const Node* []) { return_jp });
            // Join up at the return address instead of returning
            return finish_body(bb, join(dst_arena, (Join) {
                .join_point = return_jp,
                .args = nargs,
            }));
        }
        case Let_TAG: {
            const Node* old_instruction = get_let_instruction(old);
            const Node* new_instruction = NULL;
            const Node* old_tail = get_let_tail(old);
            // we convert calls to tail-calls within a control
            // let(call_indirect(...), ret_fn) to let(control(jp => save(jp); tailcall(...)), ret_fn)
            if (old_instruction->tag == IndirectCall_TAG) {
                const Node* tail_type = (old_tail->type);
                Nodes oparam_types;
                switch (tail_type->tag) {
                    case LamType_TAG: oparam_types = tail_type->payload.lam_type.param_types; break;
                    case BBType_TAG: oparam_types = tail_type->payload.bb_type.param_types; break;
                    default: error("tail must be a bb or lambda");
                }
                Nodes returned_types = rewrite_nodes(&ctx->rewriter, oparam_types);

                const Type* jpt = qualified_type(dst_arena, (QualifiedType) {
                    .type = join_point_type(dst_arena, (JoinPointType) { .yield_types = strip_qualifiers(dst_arena, returned_types) }),
                    .is_uniform = true
                });
                const Node* jp = var(dst_arena, jpt, "fn_return_point");
                BodyBuilder* instructions = begin_body(ctx->rewriter.dst_module);
                // yeet the join point on the stack
                gen_push_value_stack(instructions, jp);
                // tail-call to the target immediately afterwards
                const Node* control_body = finish_body(instructions, tail_call(dst_arena, (TailCall) {
                    .target = rewrite_node(&ctx->rewriter, old_instruction->payload.indirect_call.callee),
                    .args = rewrite_nodes(&ctx->rewriter, old_instruction->payload.indirect_call.args),
                }));

                const Node* control_lam = lambda(ctx->rewriter.dst_module, nodes(dst_arena, 1, (const Node*[]) { jp }), control_body);
                new_instruction = control(dst_arena, (Control) { .yield_types = strip_qualifiers(dst_arena, returned_types), .inside = control_lam });
            }

            if (!new_instruction)
                new_instruction = rewrite_node(&ctx->rewriter, old_instruction);

            const Node* new_tail = rewrite_node(&ctx->rewriter, old_tail);
            return let(dst_arena, new_instruction, new_tail);
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
