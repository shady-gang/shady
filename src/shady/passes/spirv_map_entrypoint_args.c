#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static const Node* rewrite_args_type(Rewriter* rewriter, const Node* old_type) {
    IrArena* a = rewriter->dst_arena;

    if (old_type->tag != RecordType_TAG || old_type->payload.record_type.special != NotSpecial)
        error("EntryPointArgs type must be a plain record type");

    const Node* new_type = record_type(a, (RecordType) {
        .members = rewrite_nodes(rewriter, old_type->payload.record_type.members),
        .names = old_type->payload.record_type.names,
        .special = DecorateBlock
    });

    register_processed(rewriter, old_type, new_type);

    return new_type;
}

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case GlobalVariable_TAG:
            if (lookup_annotation(node, "EntryPointArgs")) {
                if (node->payload.global_variable.address_space != AsExternal)
                    error("EntryPointArgs address space must be extern");

                Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations);
                const Node* type = rewrite_args_type(&ctx->rewriter, node->payload.global_variable.type);

                const Node* new_var = global_var(ctx->rewriter.dst_module,
                    annotations,
                    type,
                    node->payload.global_variable.name,
                    AsVKPushConstant
                );

                register_processed(&ctx->rewriter, node, new_var);

                return new_var;
            }
            break;
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void spirv_map_entrypoint_args(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
