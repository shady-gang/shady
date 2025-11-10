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
    enum {
        PUSH_CONSTANT,
        DESCRIPTOR_OPAQUE,
    } to;
    union {
        int pc_idx;
        const Node* descriptor;
    };
} ParamLowered;

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* generate_arg_struct(Rewriter* rewriter, const Node* old_entry_point, const Node* new_entry_point, ParamLowered lowered[]) {
    IrArena* a = rewriter->dst_arena;

    Nodes params = old_entry_point->payload.fun.params;

    LARRAY(const Node*, pc_types, params.count);
    LARRAY(String, pc_names, params.count);
    size_t pc_struct_elements_count = 0;

    Nodes annotations = shd_empty(a);

    size_t set = 0;
    size_t binding = 0;

    bool finished_with_synethic_args = false;
    int synthetic_args_count = 0;
    for (int i = 0; i < params.count; ++i) {
        const Node* param = params.nodes[i];
        const Type* type = shd_rewrite_node(rewriter, param->type);

        if (shd_deconstruct_qualified_type(&type) != shd_get_arena_config(a)->target.scopes.constants)
            shd_error("EntryPoint parameters must be uniform");

        const Node* iface_annotation = annotation_helper(a, "EntryPointInterface");
        shd_add_annotation(new_entry_point, iface_annotation);
        const Node* src_annotation = NULL;
        const Node* dst_annotation = NULL;

        if (!shd_is_physical_data_type(type)) {
            lowered[i].to = DESCRIPTOR_OPAQUE;
            Node* descriptor = global_variable_helper(rewriter->dst_module, type, AsUniformConstant);
            shd_add_annotation(descriptor, annotation_value_helper(a, "DescriptorSet", shd_uint32_literal(a, set)));
            shd_add_annotation(descriptor, annotation_value_helper(a, "DescriptorBinding", shd_uint32_literal(a, binding)));
            lowered[i].descriptor = descriptor;
            dst_annotation = annotation_values(a, (AnnotationValues) {
                .name = "DstDescriptor",
                .values = mk_nodes(a, shd_int32_literal(a, set), shd_int32_literal(a, binding))
            });
            binding++;
        } else {
            lowered[i].to = PUSH_CONSTANT;
            TypeMemLayout pc_layout = shd_get_record_layout_from_member_types(a, shd_nodes(a, pc_struct_elements_count, pc_types), NULL);
            TypeMemLayout type_layout = shd_get_mem_layout(a, type);
            lowered[i].pc_idx = pc_struct_elements_count;
            pc_types[pc_struct_elements_count] = type;
            pc_names[pc_struct_elements_count] = shd_get_node_name_unsafe(params.nodes[i]);
            pc_struct_elements_count++;
            dst_annotation = annotation_values(a, (AnnotationValues) {
                .name = "DstPushConstant",
                .values = mk_nodes(a, shd_int32_literal(a, pc_layout.size_in_bytes), shd_int32_literal(a, type_layout.size_in_bytes))
            });
        }

        const Node* provide_tmp_alloc = shd_lookup_annotation(param, "RuntimeProvideTmpAllocation");
        const Node* provide_constant = shd_lookup_annotation(param, "RuntimeProvideConstant");
        const Node* provide_scratch = shd_lookup_annotation(param, "RuntimeProvideScratch");
        if (provide_tmp_alloc) {
            Nodes arr = provide_tmp_alloc->payload.annotation_values.values;
            const Node* contents = shd_rewrite_node(rewriter, shd_first(arr));

            src_annotation = annotation_value(a, (AnnotationValue) {
                .name = "SrcTmp",
                .value = contents
            });
            synthetic_args_count++;
            assert(!finished_with_synethic_args);
        } else if (provide_constant) {
            Nodes arr = provide_constant->payload.annotation_values.values;
            const Node* contents = shd_rewrite_node(rewriter, shd_first(arr));

            src_annotation = annotation_value(a, (AnnotationValue) {
                .name = "SrcConstant",
                .value = contents
            });
            synthetic_args_count++;
            assert(!finished_with_synethic_args);
        } else if (provide_scratch) {
            Nodes arr = provide_scratch->payload.annotation_values.values;
            const Node* contents = shd_rewrite_node(rewriter, shd_first(arr));

            src_annotation = annotation_value(a, (AnnotationValue) {
                .name = "SrcScratch",
                .value = contents
            });
            synthetic_args_count++;
            assert(!finished_with_synethic_args);
        } else {
            finished_with_synethic_args = true;
            src_annotation = annotation_value(a, (AnnotationValue) {
                .name = "SrcParam",
                .value = shd_int32_literal(a, i - synthetic_args_count)
            });
        }

        assert(src_annotation);
        assert(dst_annotation);
        shd_add_annotation(iface_annotation, src_annotation);
        shd_add_annotation(iface_annotation, dst_annotation);
    }

    const Type* type = shd_struct_type_with_members_named(a,
        ShdStructFlagBlock, shd_nodes(a, pc_struct_elements_count, pc_types),
        shd_strings(a, pc_struct_elements_count, pc_names));

    String name = shd_fmt_string_irarena(a, "__%s_args", shd_get_node_name_safe(old_entry_point));
    Node* var = global_variable_helper(rewriter->dst_module, type, AsPushConstant);
    var->annotations = annotations;
    shd_set_debug_name(var, name);
    shd_module_add_export(rewriter->dst_module, name, var);

    shd_add_annotation(var, annotation_value(a, (AnnotationValue) { .name = "EntryPointPushConstants", .value = fn_addr_helper(a, new_entry_point) }));

    return var;
}

static const Node* rewrite_body(Context* ctx, const Node* old_entry_point, const Node* new, const Node* arg_struct, ParamLowered lowered[]) {
    IrArena* a = ctx->rewriter.dst_arena;

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new));

    Nodes params = old_entry_point->payload.fun.params;

    for (int i = 0; i < params.count; ++i) {
        switch (lowered[i].to) {
            case PUSH_CONSTANT: {
                const Node* addr = lea_helper(a, arg_struct, shd_int32_literal(a, 0), shd_singleton(shd_int32_literal(a, lowered[i].pc_idx)));
                const Node* val = shd_bld_load(bb, addr);
                shd_register_processed(&ctx->rewriter, params.nodes[i], val);
                break;
            }
            case DESCRIPTOR_OPAQUE: {
                const Node* val = shd_bld_load(bb, lowered[i].descriptor);
                shd_register_processed(&ctx->rewriter, params.nodes[i], val);
                break;
            }
        }
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
                ParamLowered* lowered = calloc(get_abstraction_params(node).count, sizeof(ParamLowered));
                const Node* arg_struct = generate_arg_struct(&ctx->rewriter, node, new_entry_point, lowered);
                shd_set_abstraction_body(new_entry_point, rewrite_body(ctx, node, new_entry_point, arg_struct, lowered));
                free(lowered);
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
