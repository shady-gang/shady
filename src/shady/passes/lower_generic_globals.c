#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case GlobalVariable_TAG: {
            if (node->payload.global_variable.address_space == AsGeneric) {
                AddressSpace dst_as = AsGlobal;
                const Type* t = rewrite_node(&ctx->rewriter, node->payload.global_variable.type);
                Node* new_global = global_var(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations), t, node->payload.global_variable.name, dst_as);

                const Type* dst_t = ptr_type(a, (PtrType) { .pointed_type = t, .address_space = AsGeneric });
                Nodes decl_annotations = singleton(annotation(a, (Annotation) { .name = "Generated" }));
                Node* constant_decl = constant(ctx->rewriter.dst_module, decl_annotations, dst_t,
                                            format_string_interned(a, "%s_generic", get_declaration_name(node)));
                const Node* result = constant_decl;
                constant_decl->payload.constant.value = prim_op_helper(a, convert_op, singleton(dst_t), singleton(ref_decl_helper(a, new_global)));
                register_processed(&ctx->rewriter, node, result);
                new_global->payload.global_variable.init = rewrite_node(&ctx->rewriter, node->payload.global_variable.init);
                return result;
            }
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_generic_globals(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
