#include "shady/pass.h"

#include "../ir_private.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
} Context;

static void replace_or_compare(const Node** dst, const Node* with) {
    if (!*dst)
        *dst = with;
    else {
        assert(*dst == with && "conflicting definitions");
    }
}

const Node* import_node(Rewriter* r, const Node* node) {
    if (is_declaration(node)) {
        Node* existing = get_declaration(r->dst_module, get_declaration_name(node));
        if (existing) {
            const Node* imported_t = rewrite_node(r, node->type);
            if (imported_t != existing->type) {
                shd_error_print("Incompatible types for to-be-merged declaration: %s ", get_declaration_name(node));
                shd_log_node(ERROR, existing->type);
                shd_error_print(" vs ");
                shd_log_node(ERROR, imported_t);
                shd_error_print(".\n");
                shd_error_die();
            }
            if (node->tag != existing->tag) {
                shd_error_print("Incompatible node types for to-be-merged declaration: %s ", get_declaration_name(node));
                shd_error_print("%s", node_tags[existing->tag]);
                shd_error_print(" vs ");
                shd_error_print("%s", node_tags[node->tag]);
                shd_error_print(".\n");
                shd_error_die();
            }
            switch (is_declaration(node)) {
                case NotADeclaration: assert(false);
                case Declaration_Function_TAG:
                    replace_or_compare(&existing->payload.fun.body, rewrite_node(r, node->payload.fun.body));
                    break;
                case Declaration_Constant_TAG:
                    replace_or_compare(&existing->payload.constant.value, rewrite_node(r, node->payload.constant.value));
                    break;
                case Declaration_GlobalVariable_TAG:
                    replace_or_compare(&existing->payload.global_variable.init, rewrite_node(r, node->payload.global_variable.init));
                    break;
                case Declaration_NominalType_TAG:
                    replace_or_compare(&existing->payload.nom_type.body, rewrite_node(r, node->payload.nom_type.body));
                    break;
            }
            return existing;
        }
    }

    return recreate_node_identity(r, node);
}

Module* import(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) recreate_node_identity),
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}

void link_module(Module* dst, Module* src) {
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) import_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
