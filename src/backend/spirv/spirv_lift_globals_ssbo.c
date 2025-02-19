#include "shady/pass.h"
#include "shady/ir/annotation.h"
#include "shady/ir/debug.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/function.h"
#include "shady/ir/mem.h"
#include "shady/ir/decl.h"
#include "shady/dict.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
    Node* lifted_globals_decl;
    Node2Node global2id;
} Context;

static OpRewriteResult* process(Context* ctx, NodeClass use, String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Node* newfun = shd_recreate_node_head(r, node);
            if (get_abstraction_body(node)) {
                Context fn_ctx = *ctx;
                fn_ctx.rewriter = shd_create_children_rewriter(r);
                shd_register_processed_list(&fn_ctx.rewriter, get_abstraction_params(node), get_abstraction_params(newfun));
                fn_ctx.bb = shd_bld_begin(a, shd_get_abstraction_mem(newfun));
                Node* post_prelude = basic_block_helper(a, shd_empty(a));
                shd_set_debug_name(post_prelude, "post-prelude");
                shd_register_processed(&fn_ctx.rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(post_prelude));
                shd_set_abstraction_body(post_prelude, shd_rewrite_op(&fn_ctx.rewriter, NcTerminator, "body", get_abstraction_body(node)));
                shd_set_abstraction_body(newfun, shd_bld_finish(fn_ctx.bb, jump_helper(a, shd_bld_mem(fn_ctx.bb), post_prelude,
                                                                                       shd_empty(a))));
                shd_destroy_rewriter(&fn_ctx.rewriter);
            }
            return shd_new_rewrite_result(r, newfun);
        }
        case GlobalVariable_TAG: {
            if (node->payload.global_variable.address_space != AsGlobal)
                break;
            assert(ctx->bb && "this Global isn't appearing in an abstraction - we cannot replace it with a load!");
            const Node* index = shd_node2node_find(ctx->global2id, node);
            const Node* ptr_addr = lea_helper(a, ctx->lifted_globals_decl, shd_int32_literal(a, 0), shd_singleton(index));
            const Node* ptr = shd_bld_load(ctx->bb, ptr_addr);
            OpRewriteResult* result = shd_new_rewrite_result_none(r);
            shd_rewrite_result_add_mask_rule(result, NcValue, ptr);
            return result;
        }
        default: break;
    }

    return shd_new_rewrite_result(r, shd_recreate_node(r, node));
}

static Rewriter* rewrite_globals_in_local_ctx(Rewriter* r, const Node* n) {
    if (n->tag == GlobalVariable_TAG && n->payload.global_variable.address_space == AsGlobal)
        return r;
    return shd_default_rewriter_selector(r, n);
}

Module* shd_spvbe_pass_lift_globals_ssbo(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process),
        .config = config,
        .global2id = shd_new_node2node(),
    };
    ctx.rewriter.select_rewriter_fn = rewrite_globals_in_local_ctx;

    Nodes oglobals = shd_module_collect_reachable_globals(src);
    LARRAY(const Type*, member_tys, oglobals.count);
    LARRAY(String, member_names, oglobals.count);

    size_t lifted_globals_count = 0;
    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* odecl = oglobals.nodes[i];
        if (odecl->payload.global_variable.address_space != AsGlobal)
            continue;

        member_tys[lifted_globals_count] = shd_get_unqualified_type(shd_rewrite_op(&ctx.rewriter, NcType, "type", odecl->type));
        member_names[lifted_globals_count] = shd_get_node_name_safe(odecl);

        const Node* index = shd_int32_literal(a, lifted_globals_count);
        shd_node2node_insert(ctx.global2id, odecl, index);

        lifted_globals_count++;
    }

    if (lifted_globals_count > 0) {
        const Type* lifted_globals_struct_t = record_type(a, (RecordType) {
            .members = shd_nodes(a, lifted_globals_count, member_tys),
            .names = shd_strings(a, lifted_globals_count, member_names),
            .special = DecorateBlock
        });
        ctx.lifted_globals_decl = global_variable_helper(dst, lifted_globals_struct_t, AsShaderStorageBufferObject);
        shd_set_debug_name(ctx.lifted_globals_decl, "lifted_globals");

        shd_add_annotation(ctx.lifted_globals_decl, annotation_value(a, (AnnotationValue) { .name = "DescriptorSet", .value = shd_int32_literal(a, 0) }));
        shd_add_annotation(ctx.lifted_globals_decl, annotation_value(a, (AnnotationValue) { .name = "DescriptorBinding", .value = shd_int32_literal(a, 0) }));
        shd_add_annotation_named(ctx.lifted_globals_decl, "Constants");
    }

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_node2node(ctx.global2id);

    lifted_globals_count = 0;
    for (size_t i = 0; i < oglobals.count; i++) {
        const Node* odecl = oglobals.nodes[i];
        if (odecl->payload.global_variable.address_space != AsGlobal)
            continue;
        if (odecl->payload.global_variable.init)
            shd_add_annotation(ctx.lifted_globals_decl, annotation_values(a, (AnnotationValues) {
                    .name = "InitialValue",
                    .values = mk_nodes(a, shd_int32_literal(a, lifted_globals_count), shd_rewrite_op(&ctx.rewriter, NcValue, "init", odecl->payload.global_variable.init))
            }));

        lifted_globals_count++;
    }

    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
