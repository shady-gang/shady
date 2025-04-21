#include "shady/pass.h"

#include "shady/ir/annotation.h"
#include "shady/ir/type.h"
#include "shady/ir/function.h"
#include "shady/ir/mem.h"
#include "shady/ir/cast.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node){
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            if (!shd_lookup_annotation(node, "EntryPoint"))
                break;
            Function opayload = node->payload.fun;
            Function payload = shd_rewrite_function_head_payload(r, opayload);
            LARRAY(const Type*, arr, payload.params.count);
            for (size_t i = 0; i < payload.params.count; i++) {
                const Node* oparam = opayload.params.nodes[i];
                const Node* byval = shd_lookup_annotation(oparam, "ByVal");
                if (byval) {
                    assert(byval->tag == AnnotationId_TAG);
                    const Type* t = shd_rewrite_node(r, byval->payload.annotation_id.id);
                    arr[i] = t;
                    const Node* param = param_helper(a, qualified_type_helper(a, shd_get_qualified_type_scope(oparam->type), t));
                    const Type* ptr_t = shd_rewrite_node(r, shd_get_unqualified_type(oparam->type));
                    assert(ptr_t->tag == PtrType_TAG);
                    assert(ptr_t->payload.ptr_type.address_space == AsGeneric);
                    shd_remove_annotation_by_name(param, "ByVal");
                    payload.params = shd_change_node_at_index(a, payload.params, i,param);
                } else {
                    arr[i] = NULL;
                    shd_register_processed(r, oparam, payload.params.nodes[i]);
                }
            }
            Node* new = shd_function(r->dst_module, payload);
            shd_register_processed(r, node, new);
            // shd_register_processed_list(r, get_abstraction_params(node), get_abstraction_params(new));
            shd_rewrite_annotations(r, node, new);
            BodyBuilder* builder = shd_bld_begin(a, shd_get_abstraction_mem(new));
            for (size_t i = 0; i < payload.params.count; i++) {
                if (arr[i]) {
                    const Node* oparam = get_abstraction_params(node).nodes[i];
                    const Node* ptr = shd_bld_stack_alloc(builder, arr[i]);
                    shd_bld_store(builder, ptr, payload.params.nodes[i]);
                    ptr = shd_bld_generic_ptr_cast(builder, ptr);
                    ptr = shd_bld_bitcast(builder, shd_get_unqualified_type(shd_rewrite_node(r, oparam->type)), ptr);
                    ptr = scope_cast_helper(a, shd_get_qualified_type_scope(oparam->type), ptr);
                    shd_register_processed(r, oparam, ptr);
                }
            }
            shd_register_processed(r, shd_get_abstraction_mem(node), shd_bld_mem(builder));
            shd_set_abstraction_body(new, shd_bld_finish(builder, shd_rewrite_node(r, get_abstraction_body(node))));
            //shd_recreate_node_body(r, node, new);
            return new;
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* l2s_promote_byval_params(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
