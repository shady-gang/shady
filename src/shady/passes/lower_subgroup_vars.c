#include "pass.h"

#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, NodeClass class, String op_name, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case PtrType_TAG: {
            AddressSpace as = node->payload.ptr_type.address_space;
            if (as == AsSubgroup) {
                return ptr_type(a, (PtrType) { .pointed_type = rewrite_op(&ctx->rewriter, NcType, "pointed_type", node->payload.ptr_type.pointed_type), .address_space = AsShared, .is_reference = node->payload.ptr_type.is_reference });
            }
            break;
        }
        case RefDecl_TAG: {
            const Node* odecl = node->payload.ref_decl.decl;
            if (odecl->tag != GlobalVariable_TAG || odecl->payload.global_variable.address_space != AsSubgroup)
                break;
            const Node* ndecl = rewrite_op(&ctx->rewriter, NcDeclaration, "decl", odecl);
            error("TODO")
            // assert(ctx->b);
            // const Node* index = gen_builtin_load(ctx->rewriter.dst_module, ctx->b, BuiltinSubgroupId);
            // const Node* slice = gen_lea(ctx->b, ref_decl_helper(a, ndecl), int32_literal(a, 0), mk_nodes(a, index));
            // return slice;
        }
        case GlobalVariable_TAG: {
            AddressSpace as = node->payload.global_variable.address_space;
            if (as == AsSubgroup) {
                const Type* ntype = rewrite_op(&ctx->rewriter, NcType, "type", node->payload.global_variable.type);
                const Type* atype = arr_type(a, (ArrType) {
                    .element_type = ntype,
                    .size = ref_decl_helper(a, rewrite_op(&ctx->rewriter, NcDeclaration, "decl", get_declaration(ctx->rewriter.src_module, "SUBGROUPS_PER_WG")))
                });

                assert(lookup_annotation(node, "Logical") && "All subgroup variables should be logical by now!");
                Node* new = global_var(ctx->rewriter.dst_module, rewrite_ops(&ctx->rewriter, NcAnnotation, "annotations", node->payload.global_variable.annotations), atype, node->payload.global_variable.name, AsShared);
                register_processed(&ctx->rewriter, node, new);

                if (node->payload.global_variable.init) {
                    new->payload.global_variable.init = fill(a, (Fill) {
                        .type = atype,
                        .value = rewrite_op(&ctx->rewriter, NcValue, "init", node->payload.global_variable.init)
                    });
                }
                return new;
            }
            break;
        }
        default: break;
    }

    const Node* new = recreate_node_identity(&ctx->rewriter, node);
    if (is_instruction(new))
        register_processed(r, node, new);
    return new;
}

Module* lower_subgroup_vars(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_op_rewriter(src, dst, (RewriteOpFn) process),
        .config = config
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
