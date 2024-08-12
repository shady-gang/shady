#include "shady/pass.h"

#include "../shady/type.h"
#include "../shady/transform/ir_gen_helpers.h"
#include "../shady/transform/memory_layout.h"

#include "dict.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    BodyBuilder* bb;
    Node* lifted_globals_decl;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Node* newfun = recreate_decl_header_identity(r, node);
            Context functx = *ctx;
            functx.rewriter.map = clone_dict(functx.rewriter.map);
            clear_dict(functx.rewriter.map);
            register_processed_list(&functx.rewriter, get_abstraction_params(node), get_abstraction_params(newfun));
            functx.bb = begin_body_with_mem(a, get_abstraction_mem(newfun));
            Node* post_prelude = basic_block(a, empty(a), "post-prelude");
            register_processed(&functx.rewriter, get_abstraction_mem(node), get_abstraction_mem(post_prelude));
            set_abstraction_body(post_prelude, rewrite_node(&functx.rewriter, get_abstraction_body(node)));
            set_abstraction_body(newfun, finish_body(functx.bb, jump_helper(a, post_prelude, empty(a), bb_mem(functx.bb))));
            destroy_dict(functx.rewriter.map);
            return newfun;
        }
        case RefDecl_TAG: {
            const Node* odecl = node->payload.ref_decl.decl;
            if (odecl->tag != GlobalVariable_TAG || odecl->payload.global_variable.address_space != AsGlobal)
                break;
            assert(ctx->bb && "this RefDecl node isn't appearing in an abstraction - we cannot replace it with a load!");
            const Node* ptr_addr = gen_lea(ctx->bb, ref_decl_helper(a, ctx->lifted_globals_decl), int32_literal(a, 0), singleton(rewrite_node(&ctx->rewriter, odecl)));
            const Node* ptr = gen_load(ctx->bb, ptr_addr);
            return ptr;
        }
        case GlobalVariable_TAG:
            if (node->payload.global_variable.address_space != AsGlobal)
                break;
            assert(false);
        default: break;
    }

    if (is_declaration(node)) {
        Context declctx = *ctx;
        declctx.bb = NULL;
        return recreate_node_identity(&declctx.rewriter, node);
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* spirv_lift_globals_ssbo(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config
    };

    Nodes old_decls = get_module_declarations(src);
    LARRAY(const Type*, member_tys, old_decls.count);
    LARRAY(String, member_names, old_decls.count);

    Nodes annotations = mk_nodes(a, annotation(a, (Annotation) { .name = "Generated" }));
    annotations = empty(a);

    annotations = append_nodes(a, annotations, annotation_value(a, (AnnotationValue) { .name = "DescriptorSet", .value = int32_literal(a, 0) }));
    annotations = append_nodes(a, annotations, annotation_value(a, (AnnotationValue) { .name = "DescriptorBinding", .value = int32_literal(a, 0) }));
    annotations = append_nodes(a, annotations, annotation(a, (Annotation) { .name = "Constants" }));

    size_t lifted_globals_count = 0;
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* odecl = old_decls.nodes[i];
        if (odecl->tag != GlobalVariable_TAG || odecl->payload.global_variable.address_space != AsGlobal)
            continue;

        member_tys[lifted_globals_count] = rewrite_node(&ctx.rewriter, odecl->type);
        member_names[lifted_globals_count] = get_declaration_name(odecl);

        register_processed(&ctx.rewriter, odecl, int32_literal(a, lifted_globals_count));
        lifted_globals_count++;
    }

    if (lifted_globals_count > 0) {
        const Type* lifted_globals_struct_t = record_type(a, (RecordType) {
            .members = nodes(a, lifted_globals_count, member_tys),
            .names = strings(a, lifted_globals_count, member_names),
            .special = DecorateBlock
        });
        ctx.lifted_globals_decl = global_var(dst, annotations, lifted_globals_struct_t, "lifted_globals", AsShaderStorageBufferObject);
    }

    rewrite_module(&ctx.rewriter);

    lifted_globals_count = 0;
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* odecl = old_decls.nodes[i];
        if (odecl->tag != GlobalVariable_TAG || odecl->payload.global_variable.address_space != AsGlobal)
            continue;
        if (odecl->payload.global_variable.init)
            ctx.lifted_globals_decl->payload.global_variable.annotations = append_nodes(a, ctx.lifted_globals_decl->payload.global_variable.annotations, annotation_values(a, (AnnotationValues) {
                    .name = "InitialValue",
                    .values = mk_nodes(a, int32_literal(a, lifted_globals_count), rewrite_node(&ctx.rewriter, odecl->payload.global_variable.init))
            }));

        lifted_globals_count++;
    }

    destroy_rewriter(&ctx.rewriter);
    return dst;
}
