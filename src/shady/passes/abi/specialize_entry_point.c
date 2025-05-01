#include "pipeline/pipeline_private.h"

#include "shady/ir/builtin.h"

#include "ir_private.h"

#include "portability.h"
#include "log.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    const Node* old_entry_point_decl;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PrimOp_TAG: {
            Builtin b;
            if (shd_is_builtin_load_op(node, &b) && b == BuiltinWorkgroupSize) {
                const Type* t = pack_type(a, (PackType) { .element_type = shd_uint32_type(a), .width = 3 });
                uint32_t wg_size[3];
                wg_size[0] = a->config.specializations.workgroup_size[0];
                wg_size[1] = a->config.specializations.workgroup_size[1];
                wg_size[2] = a->config.specializations.workgroup_size[2];
                return composite_helper(a, t, mk_nodes(a, shd_uint32_literal(a, wg_size[0]), shd_uint32_literal(a, wg_size[1]), shd_uint32_literal(a, wg_size[2]) ));
            }
            break;
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static void specialize_arena_config(String entry_point, Module* src, ArenaConfig* target) {
    const Node* old_entry_point_decl = shd_module_get_exported(src, entry_point);
    if (!old_entry_point_decl)
        shd_error("Entry point not found")
    if (old_entry_point_decl->tag != Function_TAG)
        shd_error("%s is not a function", entry_point);
    target->target.entry_point = entry_point;
    const Node* ep = shd_lookup_annotation(old_entry_point_decl, "EntryPoint");
    if (!ep)
        shd_error("%s is not annotated with @EntryPoint", entry_point);
    switch (shd_execution_model_from_entry_point(old_entry_point_decl)) {
        case EmNone: shd_error("Unknown entry point type: %s", shd_get_annotation_string_payload(ep))
        case EmCompute: {
            const Node* old_wg_size_annotation = shd_lookup_annotation(old_entry_point_decl, "WorkgroupSize");
            assert(old_wg_size_annotation && old_wg_size_annotation->tag == AnnotationValues_TAG && shd_get_annotation_values(old_wg_size_annotation).count == 3);
            Nodes wg_size_nodes = shd_get_annotation_values(old_wg_size_annotation);
            target->specializations.workgroup_size[0] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[0]), false);
            target->specializations.workgroup_size[1] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[1]), false);
            target->specializations.workgroup_size[2] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[2]), false);
            assert(target->specializations.workgroup_size[0] * target->specializations.workgroup_size[1] * target->specializations.workgroup_size[2] > 0);
            break;
        }
        default: break;
    }
}

static Module* specialize_entry_point(const CompilerConfig* config, String entry_point_name, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    specialize_arena_config(entry_point_name, src, &aconfig);
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };

    const Node* entry_pt = shd_module_get_exported(src, entry_point_name);
    assert(entry_pt);
    shd_module_add_export(dst, shd_get_exported_name(entry_pt), shd_rewrite_node(&ctx.rewriter, entry_pt));

    Nodes old_decls = shd_module_get_all_exported(src);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* old_decl = old_decls.nodes[i];
        if (shd_lookup_annotation(old_decl, "Internal")) {
            const Node* new = shd_rewrite_node(&ctx.rewriter, old_decl);
            String export_name = shd_get_exported_name(new);
            if (export_name)
                shd_module_add_export(dst, export_name, new);
        }
    }

    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

static void specialize_entry_point_f(String* entry_point, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(specialize_entry_point, entry_point)
}

void shd_pipeline_add_specialize_entry_point(ShdPipeline pipeline, String entry_point) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) specialize_entry_point_f, entry_point, strlen(entry_point) + 1);
}