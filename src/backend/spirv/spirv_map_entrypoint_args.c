#include "shady/pass.h"
#include "shady/ir/memory_layout.h"

#include "../shady/transform/ir_gen_helpers.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* rewrite_args_type(Rewriter* rewriter, const Node* old_type) {
    IrArena* a = rewriter->dst_arena;

    if (old_type->tag != RecordType_TAG || old_type->payload.record_type.special != NotSpecial)
        shd_error("EntryPointArgs type must be a plain record type");

    const Node* new_type = record_type(a, (RecordType) {
        .members = shd_rewrite_nodes(rewriter, old_type->payload.record_type.members),
        .names = old_type->payload.record_type.names,
        .special = DecorateBlock
    });

    shd_register_processed(rewriter, old_type, new_type);

    return new_type;
}

static const Node* process(Context* ctx, const Node* node) {
    switch (node->tag) {
        case GlobalVariable_TAG:
            if (shd_lookup_annotation(node, "EntryPointArgs")) {
                if (node->payload.global_variable.address_space != AsExternal)
                    shd_error("EntryPointArgs address space must be extern");

                Nodes annotations = shd_rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations);
                const Node* type = rewrite_args_type(&ctx->rewriter, node->payload.global_variable.type);

                const Node* new_var = global_var(ctx->rewriter.dst_module,
                    annotations,
                    type,
                    node->payload.global_variable.name,
                    AsPushConstant
                );

                shd_register_processed(&ctx->rewriter, node, new_var);

                return new_var;
            }
            break;
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_spvbe_pass_map_entrypoint_args(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
