#include "shady/pass.h"

#include "../ir_private.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
} Context;

static void replace_or_compare(bool weak, const Node** dst, const Node* with) {
    if (!*dst)
        *dst = with;
    else if (!weak) {
        assert(*dst == with && "conflicting definitions");
    } else {
        // keep the source
    }
}

static const Node* import_node(Rewriter* r, const Node* node) {
    const Node* ea = shd_lookup_annotation(node, "Exported");
    if (ea) {
        assert(ea->tag == AnnotationValue_TAG);
        AnnotationValue payload = ea->payload.annotation_value;
        String name = shd_get_string_literal(ea->arena, payload.value);
        Node* existing = shd_module_get_exported(r->dst_module, name);
        if (existing) {
            const Node* imported_t = shd_rewrite_node(r, node->type);
            if (imported_t != existing->type) {
                shd_error_print("Incompatible types for to-be-merged declaration: %s ", name);
                shd_log_node(ERROR, existing->type);
                shd_error_print(" vs ");
                shd_log_node(ERROR, imported_t);
                shd_error_print(".\n");
                shd_error_die();
            }
            if (node->tag != existing->tag) {
                shd_error_print("Incompatible node tags for to-be-merged declaration: %s ", name);
                shd_error_print("%s", shd_get_node_tag_string(existing->tag));
                shd_error_print(" vs ");
                shd_error_print("%s", shd_get_node_tag_string(node->tag));
                shd_error_print(".\n");
                shd_error_die();
            }
            shd_register_processed(shd_get_top_rewriter(r), node, existing);
            bool weak = shd_lookup_annotation(existing, "Weak");
            switch (node->tag) {
                default: shd_error("TODO");
                case Function_TAG:
                    replace_or_compare(weak, &existing->payload.fun.body, shd_rewrite_node(r, node->payload.fun.body));
                    break;
                case Constant_TAG:
                    replace_or_compare(weak, &existing->payload.constant.value, shd_rewrite_node(r, node->payload.constant.value));
                    break;
                case GlobalVariable_TAG:
                    replace_or_compare(weak, &existing->payload.global_variable.init, shd_rewrite_node(r, node->payload.global_variable.init));
                    break;
                case NominalType_TAG:
                    replace_or_compare(weak, &existing->payload.nom_type.body, shd_rewrite_node(r, node->payload.nom_type.body));
                    break;
            }
            return existing;
        }
    }

    return shd_recreate_node(r, node);
}

Module* shd_import(SHADY_UNUSED void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) shd_recreate_node),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

void shd_module_link(Module* dst, Module* src) {
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) import_node),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
}
