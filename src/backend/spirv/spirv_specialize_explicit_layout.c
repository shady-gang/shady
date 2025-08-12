#include <shady/ir/composite.h>

#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/decl.h"
#include "shady/ir/annotation.h"
#include "shady/ir/function.h"
#include "shady/ir/debug.h"

#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    Rewriter aggregate_types;
    const CompilerConfig* config;
} Context;

static bool has_explicit_layout(const TargetConfig* target, AddressSpace as) {
    switch (as) {
        // despite not being physical, they require explicit layout
        case AsShaderStorageBufferObject:
        case AsUniform:
        case AsPushConstant: return true;
        default: break;
    }
    return target->memory.address_spaces[as].physical;
}

static const Type* rebuild_aggregate_type(Rewriter* r, const Type* t) {
    IrArena* a = r->dst_arena;
    switch (is_type(t)) {
        case Type_StructType_TAG: {
            StructType payload = shd_rewrite_struct_type_head_payload(r, t->payload.struct_type);
            payload.flags |= ShdStructFlagExplicitLayout;
            Type* nst = struct_type(a, payload);
            shd_register_processed(r, t, nst);
            shd_rewrite_annotations(r, t, nst);
            shd_set_debug_name(nst, shd_fmt_string_irarena(a, "%s_explicit", shd_get_node_name_safe(nst)));
            shd_recreate_node_body(r, t, nst);
            return nst;
        }
        case Type_ArrType_TAG: {
            ArrType payload = t->payload.arr_type;
            payload.element_type = shd_rewrite_node(r, payload.element_type);
            payload.size = shd_rewrite_node(r, payload.size);
            payload.flags |= ShdArrayFlagExplicitLayout;
            return arr_type(a, payload);
        }
        // derived aggregates
        case Type_QualifiedType_TAG:
        case Type_VectorType_TAG:
        case Type_MatrixType_TAG:
        case Type_ExtType_TAG:
        default: break;
        // non-aggregates, irrelevant
        case Type_NoRet_TAG: break;
        case Type_Int_TAG: break;
        case Type_Float_TAG: break;
        case Type_Bool_TAG: break;
        case Type_ImageType_TAG:break;
        case Type_SamplerType_TAG:break;
        case Type_SampledImageType_TAG:break;
        // only deal with values, not memory
        case Type_FnType_TAG:break;
        case Type_BBType_TAG:break;
        case Type_LamType_TAG:break;
        case Type_JoinPointType_TAG: break;
        case Type_TupleType_TAG:break;
            // handled by first rewrite pass
        case Type_PtrType_TAG: break;
    }
    // if (shd_is_node_nominal(t))
    //     return t;
    return t;
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case GlobalVariable_TAG: {
            GlobalVariable payload = shd_rewrite_global_head_payload(r, node->payload.global_variable);
            if (has_explicit_layout(&shd_get_arena_config(a)->target, payload.address_space)) {
                payload.type = shd_rewrite_node(&ctx->aggregate_types, payload.type);
            }
            Node* ngv = shd_global_var(r->dst_module, payload);
            shd_register_processed(r, node, ngv);
            shd_rewrite_annotations(r, node, ngv);
            shd_recreate_node_body(r, node, ngv);
            return ngv;
        }
        case PtrType_TAG: {
            PtrType payload = node->payload.ptr_type;
            payload.pointed_type = shd_rewrite_node(&ctx->rewriter, payload.pointed_type);
            if (has_explicit_layout(&shd_get_arena_config(a)->target, payload.address_space))
                payload.pointed_type = shd_rewrite_node(&ctx->aggregate_types, payload.pointed_type);
            return ptr_type(a, payload);
        }
        case Load_TAG: {
            Load payload = node->payload.load;
            const Type* old_ptr_t = payload.ptr->type;
            shd_deconstruct_qualified_type(&old_ptr_t);
            assert(old_ptr_t->tag == PtrType_TAG);
            const Type* pointee_type = shd_rewrite_node(r, shd_get_pointer_type_element(old_ptr_t));
            payload.mem = shd_rewrite_node(r, payload.mem);
            payload.ptr = shd_rewrite_node(r, payload.ptr);
            const Node* nld = load(a, payload);
            // get rid of explicit layout at the value level
            if (has_explicit_layout(&shd_get_arena_config(a)->target, old_ptr_t->payload.ptr_type.address_space) && shd_is_aggregate_t(pointee_type))
                nld = mem_and_value_helper(a, nld, aggregate_cast_helper(a, pointee_type, nld));
            return nld;
        }
        case Store_TAG: {
            Store payload = node->payload.store;
            const Type* old_ptr_t = payload.ptr->type;
            shd_deconstruct_qualified_type(&old_ptr_t);
            assert(old_ptr_t->tag == PtrType_TAG);
            const Type* pointee_type = shd_rewrite_node(r, shd_get_pointer_type_element(old_ptr_t));
            payload.mem = shd_rewrite_node(r, payload.mem);
            payload.ptr = shd_rewrite_node(r, payload.ptr);
            payload.value = shd_rewrite_node(r, payload.value);
            // add explicit layout when storing
            if (has_explicit_layout(&shd_get_arena_config(a)->target, old_ptr_t->payload.ptr_type.address_space) && shd_is_aggregate_t(pointee_type))
                payload.value = aggregate_cast_helper(a, shd_rewrite_node(&ctx->aggregate_types, pointee_type), payload.value);
            return store(a, payload);
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* shd_spvbe_pass_specialize_explicit_layout(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .aggregate_types = shd_create_node_rewriter(dst, dst, (RewriteNodeFn) rebuild_aggregate_type),
        .config = config
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.aggregate_types);
    return dst;
}
