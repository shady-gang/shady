#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/function.h"
#include "shady/ir/debug.h"
#include "shady/ir/decl.h"
#include "shady/ir/annotation.h"
#include "shady/ir/mem.h"
#include "shady/ir/composite.h"

#include "portability.h"
#include "log.h"
#include "util.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* generate_arg_struct(Rewriter* rewriter, const Node* old_entry_point, const Node* new_entry_point) {
    IrArena* a = rewriter->dst_arena;

    Nodes params = old_entry_point->payload.fun.params;
    LARRAY(const Node*, types, params.count);
    LARRAY(String, names, params.count);

    Nodes annotations = shd_empty(a);

    //size_t offset = 0;

    bool finished_with_synethic_args = false;
    int synthetic_args_count = 0;
    for (int i = 0; i < params.count; ++i) {
        const Node* param = params.nodes[i];
        const Type* type = shd_rewrite_node(rewriter, param->type);

        if (shd_deconstruct_qualified_type(&type) != shd_get_arena_config(a)->target.scopes.constants)
            shd_error("EntryPoint parameters must be uniform");

            const Node* provide_mem = shd_lookup_annotation(param, "RuntimeProvideMem");
            const Node* provide_scratch = shd_lookup_annotation(param, "RuntimeProvideScratch");
            if (provide_mem) {
                Nodes arr = provide_mem->payload.annotation_values.values;
                const Node* contents = shd_rewrite_node(rewriter, shd_first(arr));

                annotations = shd_nodes_append(a, annotations, annotation_values(a, (AnnotationValues) {
                    .name = "RuntimeProvideMemInPushConstant",
                    .values = mk_nodes(a, shd_int32_literal(a, i), contents)
                }));
                synthetic_args_count++;
                assert(!finished_with_synethic_args);
            } else if (provide_scratch) {
                Nodes arr = provide_scratch->payload.annotation_values.values;
                const Node* contents = shd_rewrite_node(rewriter, shd_first(arr));

                annotations = shd_nodes_append(a, annotations, annotation_values(a, (AnnotationValues) {
                    .name = "RuntimeProvideScratchInPushConstant",
                    .values = mk_nodes(a, shd_int32_literal(a, i), contents)
                }));
                synthetic_args_count++;
                assert(!finished_with_synethic_args);
            } else {
                finished_with_synethic_args = true;
                annotations = shd_nodes_append(a, annotations, annotation_values(a, (AnnotationValues) {
                    .name = "RuntimeParamInPushConstant",
                    .values = mk_nodes(a, shd_int32_literal(a, i), shd_int32_literal(a, i - synthetic_args_count))
                }));
            }

        types[i] = type;
        names[i] = shd_get_node_name_unsafe(params.nodes[i]);
    }

    const Type* type = shd_struct_type_with_members_named(a, ShdStructFlagBlock, shd_nodes(a, params.count, types), shd_strings(a, params.count, names));

    String name = shd_fmt_string_irarena(a, "__%s_args", shd_get_node_name_safe(old_entry_point));
    Node* var = global_variable_helper(rewriter->dst_module, type, AsPushConstant);
    var->annotations = annotations;
    shd_set_debug_name(var, name);
    shd_module_add_export(rewriter->dst_module, name, var);

    shd_add_annotation(var, annotation_value(a, (AnnotationValue) { .name = "EntryPointPushConstants", .value = fn_addr_helper(a, new_entry_point) }));

    return var;
}

static const Node* rewrite_body(Context* ctx, const Node* old_entry_point, const Node* new, const Node* arg_struct) {
    IrArena* a = ctx->rewriter.dst_arena;

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new));

    Nodes params = old_entry_point->payload.fun.params;

    for (int i = 0; i < params.count; ++i) {
        const Node* addr = lea_helper(a, arg_struct, shd_int32_literal(a, 0), shd_singleton(shd_int32_literal(a, i)));
        const Node* val = shd_bld_load(bb, addr);
        shd_register_processed(&ctx->rewriter, params.nodes[i], val);
    }

    shd_register_processed(&ctx->rewriter, shd_get_abstraction_mem(old_entry_point), shd_bld_mem(bb));
    return shd_bld_finish(bb, shd_rewrite_node(&ctx->rewriter, old_entry_point->payload.fun.body));
}

static const Node* process(Context* ctx, const Node* node) {
    switch (node->tag) {
        case Function_TAG:
            if (shd_lookup_annotation(node, "EntryPoint") && node->payload.fun.params.count > 0) {
                Rewriter* r = &ctx->rewriter;
                IrArena* a = r->dst_arena;
                Node* fun = function_helper(r->dst_module, shd_empty(a), shd_empty(a));
                shd_rewrite_annotations(r, node, fun);
                shd_register_processed(r, node, fun);
                Node* new_entry_point = fun;
                const Node* arg_struct = generate_arg_struct(&ctx->rewriter, node, new_entry_point);
                shd_set_abstraction_body(new_entry_point, rewrite_body(ctx, node, new_entry_point, arg_struct));
                return new_entry_point;
            }
            break;
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_spv_lower_entrypoint_args(const CompilerConfig* config, SHADY_UNUSED void* unused, Module* src) {
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
