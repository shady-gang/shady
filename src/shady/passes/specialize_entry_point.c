#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    const Node* old_entry_point_decl;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PrimOp_TAG: {
            Builtin b;
            if (is_builtin_load_op(node, &b) && b == BuiltinWorkgroupSize) {
                const Type* t = pack_type(a, (PackType) { .element_type = uint32_type(a), .width = 3 });
                uint32_t wg_size[3];
                wg_size[0] = a->config.specializations.workgroup_size[0];
                wg_size[1] = a->config.specializations.workgroup_size[1];
                wg_size[2] = a->config.specializations.workgroup_size[2];
                return quote_helper(a, singleton(composite_helper(a, t, mk_nodes(a, uint32_literal(a, wg_size[0]), uint32_literal(a, wg_size[1]), uint32_literal(a, wg_size[2]) ))));
            }
            break;
        }
        case GlobalVariable_TAG: {
            const Node* ba = lookup_annotation(node, "Builtin");
            if (ba) {
                Builtin b = get_builtin_by_name(get_annotation_string_payload(ba));
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
            Node* ncnst = (Node*) recreate_node_identity(&ctx->rewriter, node);
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
                ncnst->payload.constant.instruction = quote_helper(a, singleton(uint32_literal(a, subgroups_per_wg)));
            }
            return ncnst;
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* find_entry_point(Module* m, const CompilerConfig* config) {
    if (!config->specialization.entry_point)
        return NULL;
    const Node* found = NULL;
    Nodes old_decls = get_module_declarations(m);
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
    if (old_entry_point_decl->tag != Function_TAG)
        error("%s is not a function", config->specialization.entry_point);
    const Node* ep = lookup_annotation(old_entry_point_decl, "EntryPoint");
    if (!ep)
        error("%s is not annotated with @EntryPoint", config->specialization.entry_point);
    switch (execution_model_from_string(get_annotation_string_payload(ep))) {
        case EmNone: error("Unknown entry point type: %s", get_annotation_string_payload(ep))
        case EmCompute: {
            const Node* old_wg_size_annotation = lookup_annotation(old_entry_point_decl, "WorkgroupSize");
            assert(old_wg_size_annotation && old_wg_size_annotation->tag == AnnotationValues_TAG && get_annotation_values(old_wg_size_annotation).count == 3);
            Nodes wg_size_nodes = get_annotation_values(old_wg_size_annotation);
            target->specializations.workgroup_size[0] = get_int_literal_value(*resolve_to_int_literal(wg_size_nodes.nodes[0]), false);
            target->specializations.workgroup_size[1] = get_int_literal_value(*resolve_to_int_literal(wg_size_nodes.nodes[1]), false);
            target->specializations.workgroup_size[2] = get_int_literal_value(*resolve_to_int_literal(wg_size_nodes.nodes[2]), false);
            assert(target->specializations.workgroup_size[0] * target->specializations.workgroup_size[1] * target->specializations.workgroup_size[2]);
            break;
        }
        default: break;
    }
}

Module* specialize_entry_point(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    specialize_arena_config(config, src, &aconfig);
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };

    const Node* old_entry_point_decl = find_entry_point(src, config);
    rewrite_node(&ctx.rewriter, old_entry_point_decl);

    Nodes old_decls = get_module_declarations(src);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* old_decl = old_decls.nodes[i];
        if (lookup_annotation(old_decl, "RetainAfterSpecialization"))
            rewrite_node(&ctx.rewriter, old_decl);
    }

    destroy_rewriter(&ctx.rewriter);
    return dst;
}
