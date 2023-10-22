#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
    const Node* old_entry_point_decl;
    const Node* old_wg_size_annotation;
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
                return quote_helper(a, singleton(composite(a, t,mk_nodes(a, uint32_literal(a, wg_size[0]), uint32_literal(a, wg_size[1]), uint32_literal(a, wg_size[2]) ))));
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
            if (strcmp(get_decl_name(ncnst), "SUBGROUP_SIZE") == 0) {
                ncnst->payload.constant.value = uint32_literal(a, a->config.specializations.subgroup_size);
            } else if (strcmp(get_decl_name(ncnst), "SUBGROUPS_PER_WG") == 0) {
                if (ctx->old_wg_size_annotation) {
                    // SUBGROUPS_PER_WG = (NUMBER OF INVOCATIONS IN SUBGROUP / SUBGROUP SIZE)
                    // Note: this computations assumes only full subgroups are launched, if subgroups can launch partially filled then this relationship does not hold.
                    uint32_t wg_size[3];
                    wg_size[0] = a->config.specializations.workgroup_size[0];
                    wg_size[1] = a->config.specializations.workgroup_size[1];
                    wg_size[2] = a->config.specializations.workgroup_size[2];
                    uint32_t subgroups_per_wg = (wg_size[0] * wg_size[1] * wg_size[2]) / a->config.specializations.subgroup_size;
                    if (subgroups_per_wg == 0)
                        subgroups_per_wg = 1; // uh-oh
                    ncnst->payload.constant.value = uint32_literal(a, subgroups_per_wg);
                }
            }
            return ncnst;
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* find_entry_point(Module* m, CompilerConfig* config) {
    const Node* old_entry_point_decl = NULL;

    Nodes old_decls = get_module_declarations(m);
    for (size_t i = 0; i < old_decls.count; i++) {
        if (strcmp(get_decl_name(old_decls.nodes[i]), config->specialization.entry_point) == 0) {
            old_entry_point_decl = old_decls.nodes[i];
        }
    }
    if (!old_entry_point_decl) {
        error("Asked to specialize on %s but no such declaration was found", config->specialization.entry_point);
    }
    return old_entry_point_decl;
}

void specialize_arena_config(ArenaConfig* target, Module* m, CompilerConfig* config) {
    const Node* old_entry_point_decl = find_entry_point(m, config);
    if (old_entry_point_decl->tag != Function_TAG) {
        error("%s is not a function", config->specialization.entry_point);
    }
    const Node* ep = lookup_annotation(old_entry_point_decl, "EntryPoint");
    if (!ep) {
        error("%s is not annotated with @EntryPoint", config->specialization.entry_point);
    }
    if (ep->tag != AnnotationValue_TAG || get_annotation_value(ep)->tag != StringLiteral_TAG){
        error("%s's @EntryPoint annotation does not contain a string literal", config->specialization.entry_point);
    }

    String entry_point_type = get_string_literal(get_module_arena(m), get_annotation_value(ep));

    if (strcmp(entry_point_type, "compute") == 0) {
        const Node* old_wg_size_annotation = lookup_annotation(old_entry_point_decl, "WorkgroupSize");
        assert(old_wg_size_annotation && old_wg_size_annotation->tag == AnnotationValues_TAG && get_annotation_values(old_wg_size_annotation).count == 3);
        Nodes wg_size_nodes = get_annotation_values(old_wg_size_annotation);
        target->specializations.workgroup_size[0] = get_int_literal_value(wg_size_nodes.nodes[0], false);
        target->specializations.workgroup_size[1] = get_int_literal_value(wg_size_nodes.nodes[1], false);
        target->specializations.workgroup_size[2] = get_int_literal_value(wg_size_nodes.nodes[2], false);
        assert(target->specializations.workgroup_size[0] * target->specializations.workgroup_size[1] * target->specializations.workgroup_size[2]);
    } else if (strcmp(entry_point_type, "fragment") == 0) {
        // TODO
    } else {
        error("Unknown entry point type: %s", entry_point_type);
    }
}

static void specialize_for_entry_point_(CompilerConfig* config, Module* src, Module* dst, bool early) {
    IrArena* a = get_module_arena(dst);
    assert(a->config.specializations.subgroup_size);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
    };

    const Node* old_entry_point_decl = find_entry_point(src, config);
    rewrite_node(&ctx.rewriter, old_entry_point_decl);

    if (early) {
        // keep internal decls around the first time
        Nodes odecls = get_module_declarations(src);
        for (size_t i = 0; i < odecls.count; i++) {
            const Node* odecl = odecls.nodes[i];
            if (lookup_annotation(odecl, "Internal"))
                rewrite_node(&ctx.rewriter, odecl);
        }
    }

    destroy_rewriter(&ctx.rewriter);
}

void specialize_for_entry_point(CompilerConfig* config, Module* src, Module* dst) {
    specialize_for_entry_point_(config, src, dst, false);
}

void specialize_for_entry_point_early(CompilerConfig* config, Module* src, Module* dst) {
    specialize_for_entry_point_(config, src, dst, true);
}
