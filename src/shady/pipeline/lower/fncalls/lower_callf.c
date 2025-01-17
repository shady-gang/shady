#include "join_point_ops.h"

#include "shady/pass.h"
#include "shady/ir/ext.h"
#include "shady/ir/annotation.h"
#include "shady/ir/function.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const Node* return_jp;
} Context;

static const Node* lower_callf_process(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    Rewriter* r = &ctx->rewriter;

    if (old->tag == Function_TAG) {
        Context ctx2 = *ctx;
        ctx2.disable_lowering = shd_lookup_annotation(old, "Leaf");
        ctx2.return_jp = NULL;

        if (!ctx2.disable_lowering && get_abstraction_body(old)) {
            Nodes oparams = get_abstraction_params(old);
            Nodes nparams = shd_recreate_params(&ctx->rewriter, oparams);
            shd_register_processed_list(&ctx->rewriter, oparams, nparams);

            Node* prelude = case_(a, shd_empty(a));
            BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(prelude));

            // Supplement an additional parameter for the join point
            const Type* jp_type = join_point_type(a, (JoinPointType) {
                .yield_types = shd_strip_qualifiers(a, shd_rewrite_nodes(&ctx->rewriter, old->payload.fun.return_types))
            });

            if (shd_lookup_annotation(old, "EntryPoint")) {
                ctx2.return_jp = shd_bld_ext_instruction(bb, "shady.internal", ShadyOpDefaultJoinPoint,
                                                         shd_as_qualified_type(jp_type, true), shd_empty(a));
            } else {
                const Node* jp_variable = param_helper(a, shd_as_qualified_type(jp_type, false), "return_jp");
                nparams = shd_nodes_append(a, nparams, jp_variable);
                ctx2.return_jp = jp_variable;
            }

            Node* fun = function_helper(ctx->rewriter.dst_module, nparams, shd_get_abstraction_name(old), shd_empty(a));
            shd_rewrite_annotations(r, old, fun);
            shd_register_processed(&ctx->rewriter, old, fun);

            shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(old), shd_bld_mem(bb));
            shd_set_abstraction_body(prelude, shd_bld_finish(bb, shd_rewrite_node(&ctx2.rewriter, old->payload.fun.body)));
            shd_set_abstraction_body(fun, jump_helper(a, shd_get_abstraction_mem(fun), prelude, shd_empty(a)));
            return fun;
        }

        Node* fun = shd_recreate_node_head(&ctx->rewriter, old);
        if (old->payload.fun.body)
            shd_set_abstraction_body(fun, shd_rewrite_node(&ctx2.rewriter, old->payload.fun.body));
        return fun;
    }

    if (ctx->disable_lowering)
        return shd_recreate_node(&ctx->rewriter, old);

    switch (old->tag) {
        case FnType_TAG: {
            Nodes param_types = shd_rewrite_nodes(r, old->payload.fn_type.param_types);
            Nodes returned_types = shd_rewrite_nodes(&ctx->rewriter, old->payload.fn_type.return_types);
            const Type* jp_type = qualified_type(a, (QualifiedType) {
                    .type = join_point_type(a, (JoinPointType) { .yield_types = shd_strip_qualifiers(a, returned_types) }),
                    .is_uniform = false
            });
            param_types = shd_nodes_append(a, param_types, jp_type);
            return fn_type(a, (FnType) {
                .param_types = param_types,
                .return_types = shd_empty(a),
            });
        }
        case Return_TAG: {
            Nodes nargs = shd_rewrite_nodes(&ctx->rewriter, old->payload.fn_ret.args);

            const Node* return_jp = ctx->return_jp;
            if (return_jp) {
                BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, old->payload.fn_ret.mem));
                return_jp = prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(return_jp));
                // Join up at the return address instead of returning
                return shd_bld_finish(bb, join(a, (Join) {
                    .join_point = return_jp,
                    .args = nargs,
                    .mem = shd_bld_mem(bb),
                }));
            } else {
                assert(false);
            }
        }
        // we convert calls to tail-calls within a control - only if the
        // call_indirect(...) to control(jp => save(jp); tailcall(...))
        case IndirectCall_TAG: {
            IndirectCall payload = old->payload.indirect_call;
            const Node* ocallee = payload.callee;
            // if we know the callee and it's a leaf - then we don't change the call
            if (ocallee->tag == FnAddr_TAG && shd_lookup_annotation(ocallee->payload.fn_addr.fn, "Leaf"))
                break;

            const Type* ocallee_type = ocallee->type;
            bool callee_uniform = shd_deconstruct_qualified_type(&ocallee_type);
            ocallee_type = shd_get_pointee_type(a, ocallee_type);
            assert(ocallee_type->tag == FnType_TAG);
            Nodes returned_types = shd_rewrite_nodes(&ctx->rewriter, ocallee_type->payload.fn_type.return_types);

            // Rewrite the callee and its arguments
            const Node* ncallee = shd_rewrite_node(&ctx->rewriter, ocallee);
            Nodes nargs = shd_rewrite_nodes(&ctx->rewriter, payload.args);

            // Create the body of the control that receives the appropriately typed join point
            const Type* jp_type = qualified_type(a, (QualifiedType) {
                    .type = join_point_type(a, (JoinPointType) { .yield_types = shd_strip_qualifiers(a, returned_types) }),
                    .is_uniform = false
            });
            const Node* jp = param_helper(a, jp_type, "fn_return_point");

            // Add that join point as the last argument to the newly made function
            nargs = shd_nodes_append(a, nargs, jp);

            // the body of the control is just an immediate tail-call
            Node* control_case = case_(a, shd_singleton(jp));
            const Node* control_body = tail_call(a, (TailCall) {
                .callee = ncallee,
                .args = nargs,
                .mem = shd_get_abstraction_mem(control_case),
            });
            shd_set_abstraction_body(control_case, control_body);
            BodyBuilder* bb = shd_bld_begin_pseudo_instr(a, shd_rewrite_node(r, payload.mem));
            return shd_bld_to_instr_yield_values(bb, shd_bld_control(bb, shd_strip_qualifiers(a, returned_types), control_case));
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, old);
}

Module* shd_pass_lower_callf(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) lower_callf_process),
        .disable_lowering = false,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
