#include "shady/pass.h"
#include "shady/ir/builtin.h"

#include "../ir_private.h"

#include "portability.h"
#include "log.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    const Node* old_entry_point_decl;
    const CompilerConfig* config;
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
        case GlobalVariable_TAG: {
            const Node* ba = shd_lookup_annotation(node, "Builtin");
            if (ba) {
                Builtin b = shd_get_builtin_by_name(shd_get_annotation_string_payload(ba));
                switch (b) {
                    case BuiltinWorkgroupSize:
                        return NULL;
                    default:
                        break;
                }
            }
            break;
        }
        case Constant_TAG: {
            Node* ncnst = (Node*) shd_recreate_node(&ctx->rewriter, node);
            if (strcmp(get_declaration_name(ncnst), "SUBGROUPS_PER_WG") == 0) {
                // SUBGROUPS_PER_WG = (NUMBER OF INVOCATIONS IN SUBGROUP / SUBGROUP SIZE)
                // Note: this computations assumes only full subgroups are launched, if subgroups can launch partially filled then this relationship does not hold.
                uint32_t wg_size[3];
                wg_size[0] = a->config.specializations.workgroup_size[0];
                wg_size[1] = a->config.specializations.workgroup_size[1];
                wg_size[2] = a->config.specializations.workgroup_size[2];
                uint32_t subgroups_per_wg = (wg_size[0] * wg_size[1] * wg_size[2]) / ctx->config->specialization.subgroup_size;
                if (subgroups_per_wg == 0)
                    subgroups_per_wg = 1; // uh-oh
                ncnst->payload.constant.value = shd_uint32_literal(a, subgroups_per_wg);
            }
            return ncnst;
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static const Node* find_entry_point(Module* m, const CompilerConfig* config) {
    if (!config->specialization.entry_point)
        return NULL;
    const Node* found = NULL;
    Nodes old_decls = shd_module_get_declarations(m);
    for (size_t i = 0; i < old_decls.count; i++) {
        if (strcmp(get_declaration_name(old_decls.nodes[i]), config->specialization.entry_point) == 0) {
            assert(!found);
            found = old_decls.nodes[i];
        }
    }
    assert(found);
    return found;
}

static void specialize_arena_config(const CompilerConfig* config, Module* src, ArenaConfig* target) {
    const Node* old_entry_point_decl = find_entry_point(src, config);
    if (!old_entry_point_decl)
        shd_error("Entry point not found")
    if (old_entry_point_decl->tag != Function_TAG)
        shd_error("%s is not a function", config->specialization.entry_point);
    const Node* ep = shd_lookup_annotation(old_entry_point_decl, "EntryPoint");
    if (!ep)
        shd_error("%s is not annotated with @EntryPoint", config->specialization.entry_point);
    switch (shd_execution_model_from_string(shd_get_annotation_string_payload(ep))) {
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

Module* shd_pass_specialize_entry_point(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    specialize_arena_config(config, src, &aconfig);
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };

    const Node* old_entry_point_decl = find_entry_point(src, config);
    shd_rewrite_node(&ctx.rewriter, old_entry_point_decl);

    Nodes old_decls = shd_module_get_declarations(src);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* old_decl = old_decls.nodes[i];
        if (shd_lookup_annotation(old_decl, "RetainAfterSpecialization"))
            shd_rewrite_node(&ctx.rewriter, old_decl);
    }

    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
