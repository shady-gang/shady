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

    Node* self;
    const Node* return_jp;
} Context;

static const Node* lower_callf_process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;

    if (old->tag == Function_TAG) {
        Context ctx2 = *ctx;
        ctx2.disable_lowering = lookup_annotation(old, "Leaf");
        ctx2.return_jp = NULL;
        Node* fun = NULL;

        BodyBuilder* bb = begin_body(dst_arena);
        if (!ctx2.disable_lowering) {
            Nodes oparams = get_abstraction_params(old);
            Nodes nparams = recreate_variables(&ctx->rewriter, oparams);
            register_processed_list(&ctx->rewriter, oparams, nparams);

            // Supplement an additional parameter for the join point
            const Type* jp_type = join_point_type(dst_arena, (JoinPointType) {
                .yield_types = strip_qualifiers(dst_arena, rewrite_nodes(&ctx->rewriter, old->payload.fun.return_types))
            });

            if (lookup_annotation_list(old->payload.fun.annotations, "EntryPoint")) {
                ctx2.return_jp = gen_primop_e(bb, default_join_point_op, empty(dst_arena), empty(dst_arena));
            } else {
                const Node* jp_variable = var(dst_arena, qualified_type_helper(jp_type, true), "return_jp");
                nparams = append_nodes(dst_arena, nparams, jp_variable);
                ctx2.return_jp = jp_variable;
            }

            Nodes nannots = rewrite_nodes(&ctx->rewriter, old->payload.fun.annotations);
            fun = function(ctx->rewriter.dst_module, nparams, get_abstraction_name(old), nannots, empty(dst_arena));
            ctx2.self = fun;
            register_processed(&ctx->rewriter, old, fun);
        } else
            fun = recreate_decl_header_identity(&ctx->rewriter, old);
        if (old->payload.fun.body)
            fun->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, old->payload.fun.body));
        else
            cancel_body(bb);
        return fun;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, old);

    switch (old->tag) {
        case Return_TAG: {
            Nodes nargs = rewrite_nodes(&ctx->rewriter, old->payload.fn_ret.args);

            const Node* return_jp = ctx->return_jp;
            if (return_jp) {
                BodyBuilder* bb = begin_body(dst_arena);
                return_jp = gen_primop_ce(bb, subgroup_broadcast_first_op, 1, (const Node* []) {return_jp});
                // Join up at the return address instead of returning
                return finish_body(bb, join(dst_arena, (Join) {
                        .join_point = return_jp,
                        .args = nargs,
                }));
            } else {
                assert(false);
            }
        }
        case Let_TAG: {
            const Node* old_instruction = get_let_instruction(old);
            const Node* new_instruction = NULL;
            const Node* old_tail = get_let_tail(old);
            // we convert calls to tail-calls within a control
            // let(call_indirect(...), ret_fn) to let(control(jp => save(jp); tailcall(...)), ret_fn)
            if (old_instruction->tag == IndirectCall_TAG) {
                // Get the return types from the old callee
                const Node* ocallee = old_instruction->payload.indirect_call.callee;
                const Type* ocallee_type = ocallee->type;
                bool callee_uniform = deconstruct_qualified_type(&ocallee_type);
                ocallee_type = get_pointee_type(dst_arena, ocallee_type);
                assert(ocallee_type->tag == FnType_TAG);
                Nodes returned_types = rewrite_nodes(&ctx->rewriter, ocallee_type->payload.fn_type.return_types);

                // Rewrite the callee and its arguments
                const Node* ncallee = rewrite_node(&ctx->rewriter, ocallee);
                Nodes nargs = rewrite_nodes(&ctx->rewriter, old_instruction->payload.indirect_call.args);

                // Create the body of the control that receives the appropriately typed join point
                const Type* jp_type = qualified_type(dst_arena, (QualifiedType) {
                    .type = join_point_type(dst_arena, (JoinPointType) { .yield_types = strip_qualifiers(dst_arena, returned_types) }),
                    .is_uniform = true
                });
                const Node* jp = var(dst_arena, jp_type, "fn_return_point");

                // Add that join point as the last argument to the newly made function
                nargs = append_nodes(dst_arena, nargs, jp);

                // the body of the control is just an immediate tail-call
                const Node* control_body = tail_call(dst_arena, (TailCall) {
                    .target = ncallee,
                    .args = nargs,
                });
                const Node* control_lam = lambda(ctx->rewriter.dst_arena, nodes(dst_arena, 1, (const Node*[]) { jp }), control_body);
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
