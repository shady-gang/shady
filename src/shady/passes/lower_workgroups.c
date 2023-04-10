#include "passes.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
    Node* globals[PRIMOPS_COUNT];
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;

    switch (node->tag) {
        case PrimOp_TAG: {
            switch (node->payload.prim_op.op) {
                case subgroup_id_op:
                case workgroup_id_op:
                case workgroup_local_id_op:
                case workgroup_num_op:
                case workgroup_size_op:
                case global_id_op: {
                    String name = primop_names[node->payload.prim_op.op];
                    const Type* t = rewrite_node(&ctx->rewriter, node->type);
                    deconstruct_qualified_type(&t);
                    Node* decl = ctx->globals[node->payload.prim_op.op];
                    if (!decl) {
                        decl = ctx->globals[node->payload.prim_op.op] = global_var(m, empty(a), t, name, AsGlobalPhysical);
                    }
                    const Node* ref = ref_decl(a, (RefDecl) { .decl = decl });
                    BodyBuilder* bb = begin_body(a);
                    const Node* rslt = yield_values_and_wrap_in_block(bb, singleton(gen_load(bb, ref)));
                    // register_processed(&ctx->rewriter, node, rslt);
                    return rslt;
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_workgroups(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
