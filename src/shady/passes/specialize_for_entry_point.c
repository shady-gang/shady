#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
    const Node* old_entry_point_decl;
    const Node* old_wg_size_annotation;
    uint32_t sg_size;
    Nodes old_wg_size;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PrimOp_TAG: {
            switch (node->payload.prim_op.op) {
                case workgroup_size_op: {
                    const Type* t = pack_type(a, (PackType) { .element_type = uint32_type(a), .width = 3 });
                    uint32_t wg_size[3];
                    wg_size[0] = get_int_literal_value(rewrite_node(&ctx->rewriter, ctx->old_wg_size.nodes[0]), false);
                    wg_size[1] = get_int_literal_value(rewrite_node(&ctx->rewriter, ctx->old_wg_size.nodes[1]), false);
                    wg_size[2] = get_int_literal_value(rewrite_node(&ctx->rewriter, ctx->old_wg_size.nodes[2]), false);
                    return quote(a, singleton(composite(a, t,mk_nodes(a, uint32_literal(a, wg_size[0]), uint32_literal(a, wg_size[1]), uint32_literal(a, wg_size[2]) ))));
                }
                default: break;
            }
            break;
        }
        case Constant_TAG: {
            Node* ncnst = (Node*) recreate_node_identity(&ctx->rewriter, node);
            if (strcmp(get_decl_name(ncnst), "SUBGROUP_SIZE") == 0) {
                ncnst->payload.constant.value = uint32_literal(a, ctx->sg_size);
            } else if (strcmp(get_decl_name(ncnst), "SUBGROUPS_PER_WG") == 0) {
                if (ctx->old_wg_size_annotation) {
                    // TODO: this isn't necessarily correct ...
                    uint32_t wg_size[3];
                    wg_size[0] = get_int_literal_value(rewrite_node(&ctx->rewriter, ctx->old_wg_size.nodes[0]), false);
                    wg_size[1] = get_int_literal_value(rewrite_node(&ctx->rewriter, ctx->old_wg_size.nodes[1]), false);
                    wg_size[2] = get_int_literal_value(rewrite_node(&ctx->rewriter, ctx->old_wg_size.nodes[2]), false);
                    ncnst->payload.constant.value = uint32_literal(a, (wg_size[0] * wg_size[1] * wg_size[2]) / ctx->sg_size);
                }
            }
            return ncnst;
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

void specialize_for_entry_point(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
        .sg_size = config->specialization.subgroup_size
    };

    if (config->specialization.entry_point) {
        Nodes old_decls = get_module_declarations(src);
        for (size_t i = 0; i < old_decls.count; i++) {
            if (strcmp(get_decl_name(old_decls.nodes[i]), config->specialization.entry_point) == 0) {
                ctx.old_entry_point_decl = old_decls.nodes[i];
            }
        }
        if (!ctx.old_entry_point_decl) {
            error("Asked to specialize on %s but no such declaration was found", config->specialization.entry_point);
        }
        if (ctx.old_entry_point_decl->tag != Function_TAG) {
            error("%s is not a function", config->specialization.entry_point);
        }
        const Node* ep = lookup_annotation(ctx.old_entry_point_decl, "EntryPoint");
        if (!ep) {
            error("%s is not annotated with @EntryPoint", config->specialization.entry_point);
        }
        if (ep->tag != AnnotationValue_TAG || get_annotation_value(ep)->tag != StringLiteral_TAG){
            error("%s's @EntryPoint annotation does not contain a string literal", config->specialization.entry_point);
        }

        String entry_point_type = get_string_literal(ctx.rewriter.dst_arena, get_annotation_value(ep));

        if (strcmp(entry_point_type, "compute") == 0) {
            ctx.old_wg_size_annotation = lookup_annotation(ctx.old_entry_point_decl, "WorkgroupSize");
            assert(ctx.old_wg_size_annotation && ctx.old_wg_size_annotation->tag == AnnotationValues_TAG && get_annotation_values(ctx.old_wg_size_annotation).count == 3);
            ctx.old_wg_size = get_annotation_values(ctx.old_wg_size_annotation);
        } else {
            error("Unknown entry point type: %s", entry_point_type);
        }
    }

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
