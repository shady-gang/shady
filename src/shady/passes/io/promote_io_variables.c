#include "shady/ir/builtin.h"
#include "shady/ir/function.h"
#include "shady/ir/mem.h"
#include "shady/ir/debug.h"

#include "shady/pass.h"

#include "shady/ir/annotation.h"
#include "shady/ir/decl.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;

    BodyBuilder* init_bld;
    BodyBuilder* fini_bld;
} Context;

static const Node* promote_to_physical(Context* ctx, AddressSpace as, const Node* io) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;
    const Type* ptr_t = shd_get_unqualified_type(io->type);
    assert(ptr_t->tag == PtrType_TAG);
    PtrType ptr_payload = ptr_t->payload.ptr_type;
    Node* phy = global_variable_helper(m, ptr_payload.pointed_type, as);
    shd_set_debug_name(phy, shd_fmt_string_irarena(a, "%s_physical", shd_get_node_name_safe(io)));

    switch (ptr_payload.address_space) {
        case AsPushConstant:
        case AsUniformConstant:
        case AsUInput:
        case AsInput: {
            const Node* value = shd_bld_load(ctx->init_bld, io);
            shd_bld_store(ctx->init_bld, phy, value);
            // shd_bld_add_instruction(ctx->init_bld, copy_bytes_helper(a, shd_bld_mem(ctx->init_bld), phy, io, ))
            break;
        }
        case AsOutput: {
            const Node* value = shd_bld_load(ctx->fini_bld, phy);
            shd_bld_store(ctx->fini_bld, io, value);
            break;
        }
        default: assert(false);
    }

    const Type* tgt_ptr_t = ptr_type(a, (PtrType) { .address_space = AsGeneric, .pointed_type = ptr_payload.pointed_type });
    return conversion_helper(a, tgt_ptr_t, phy);
}

static const Type* make_ptr_generic(const Type* old) {
    PtrType payload = old->payload.ptr_type;
    payload.address_space = AsGeneric;
    return ptr_type(old->arena, payload);
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;
    switch (node->tag) {
        case GlobalVariable_TAG: {
            GlobalVariable payload = node->payload.global_variable;
            payload = shd_rewrite_global_head_payload(r, payload);
            const Node* io = NULL;
            ShdScope scope;
            const Node* io_annotation = shd_lookup_annotation(node, "IO");
            const Node* builtin_annotation = shd_lookup_annotation(node, "Builtin");
            if (io_annotation) {
                payload.address_space = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(io_annotation)), false);
                shd_remove_annotation_by_name(node, "IO");
                scope = shd_get_addr_space_scope(payload.address_space);
            }
            if (builtin_annotation) {
                Builtin b = shd_get_builtin_by_name(shd_get_annotation_string_payload(builtin_annotation));
                io = shd_get_or_create_builtin(m, b);
                scope = shd_get_builtin_scope(b);
            } else if (io_annotation) {
                io = shd_global_var(r->dst_module, payload);
                shd_rewrite_annotations(r, node, (Node*) io);
            } else break;

            assert(io);
            bool can_be_physical = shd_is_physical_data_type(payload.type);

            // don't perform physical conversion if there's no suitable scratch address space
            // this happens in the case of uniform variables in non-compute stages
            AddressSpace as = (scope <= ShdScopeSubgroup) ? AsSubgroup : AsPrivate;
            can_be_physical &= (shd_ir_arena_get_config(a)->target.memory.address_spaces[as].allowed);

            if (can_be_physical) {
                io = promote_to_physical(ctx, as, io);
            } else {
                io = conversion_helper(a, make_ptr_generic(shd_get_unqualified_type(io->type)), io);
            }

            shd_register_processed(r, node, io);
            return io;
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* shd_pass_promote_io_variables(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    // if (!config->specialization.entry_point)
    //     return src;
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };

    Rewriter* r = &ctx.rewriter;
    Node* init, *fini;
    const Node* oinit = shd_module_get_init_fn(src);
    const Node* ofini = shd_module_get_fini_fn(src);
    ctx.init_bld = shd_bld_begin_fn_rewrite(r, oinit, &init);
    ctx.fini_bld = shd_bld_begin_fn_rewrite(r, ofini, &fini);
    shd_rewrite_module(r);
    shd_bld_finish_fn_rewrite(r, oinit, init, ctx.init_bld);
    shd_bld_finish_fn_rewrite(r, ofini, fini, ctx.fini_bld);

    shd_destroy_rewriter(r);
    return dst;
}
