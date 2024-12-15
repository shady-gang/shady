#include "shady/ir/builtin.h"
#include "shady/ir/function.h"
#include "shady/ir/mem.h"

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

static const Node* promote_to_physical(Context* ctx, ShdScope scope, const Node* io) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;
    assert(io->tag == GlobalVariable_TAG);
    Node* phy = global_var(m, shd_empty(a), io->payload.global_variable.type, shd_fmt_string_irarena(a, "%s_physical", io->payload.global_variable.name), scope >= ShdScopeInvocation ? AsPrivate : AsSubgroup);
    const Type* pt = ptr_type(a, (PtrType) { .address_space = AsGeneric, .pointed_type = io->payload.global_variable.type });
    const Node* converted = prim_op_helper(a, convert_op, shd_singleton(pt), shd_singleton(ref_decl_helper(a, phy)));
    phy = constant(m, shd_empty(a), pt, shd_fmt_string_irarena(a, "%s_generic", io->payload.global_variable.name));
    phy->payload.constant.value = converted;

    switch (io->payload.global_variable.address_space) {
        case AsPushConstant:
        case AsUniformConstant:
        case AsUInput:
        case AsInput: {
            const Node* value = shd_bld_load(ctx->init_bld, ref_decl_helper(a, io));
            shd_bld_store(ctx->init_bld, ref_decl_helper(a, phy), value);
            // shd_bld_add_instruction(ctx->init_bld, copy_bytes_helper(a, shd_bld_mem(ctx->init_bld), phy, io, ))
            break;
        }
        case AsOutput: {
            const Node* value = shd_bld_load(ctx->fini_bld, ref_decl_helper(a, phy));
            shd_bld_store(ctx->fini_bld, ref_decl_helper(a, io), value);
            break;
        }
        default: assert(false);
    }
    return phy;
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;
    switch (node->tag) {
        case GlobalVariable_TAG: {
            GlobalVariable payload = node->payload.global_variable;
            const Node* io = NULL;
            ShdScope scope;
            const Node* io_annotation = shd_lookup_annotation(node, "IO");
            const Node* builtin_annotation = shd_lookup_annotation(node, "Builtin");
            if (io_annotation) {
                AddressSpace as = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(io_annotation)), false);
                io = global_var(m, shd_filter_out_annotation(a, shd_rewrite_nodes(r, payload.annotations), "IO"), shd_rewrite_node(r, payload.type), payload.name, as);
                scope = shd_get_addr_space_scope(as);
            } else if (builtin_annotation) {
                Builtin b = shd_get_builtin_by_name(shd_get_annotation_string_payload(builtin_annotation));
                io = shd_get_or_create_builtin(m, b, payload.name);
                scope = shd_get_builtin_scope(b);
            } else break;

            assert(io && io->tag == GlobalVariable_TAG);

            bool can_be_physical = shd_is_physical_data_type(payload.type);

            if (can_be_physical)
                io = promote_to_physical(ctx, scope, io);

            shd_register_processed(r, node, io);
            return io;
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

static BodyBuilder* begin_wrapper_rewrite(Rewriter* r, Module* src, String name, const Node** old, Node** new) {
    *old = shd_module_get_declaration(src, name);
    assert(*old);
    *new = shd_recreate_node_head(r, *old);
    shd_register_processed(r, *old, *new);
    BodyBuilder* bld = shd_bld_begin(r->dst_arena, shd_get_abstraction_mem(*new));
    return bld;
}

static void finish_wrapper_rewrite(Rewriter* r, const Node* old, Node* new, BodyBuilder* bld) {
    shd_register_processed(r, shd_get_abstraction_mem(old), shd_bld_mem(bld));
    shd_set_abstraction_body(new, shd_bld_finish(bld, shd_rewrite_node(r, get_abstraction_body(old))));
}

Module* shd_pass_promote_io_variables(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    if (!config->specialization.entry_point)
        return src;
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };

    Rewriter* r = &ctx.rewriter;
    Node* init, *fini;
    const Node* oinit, *ofini;
    ctx.init_bld = begin_wrapper_rewrite(r, src, "generated_init", &oinit, &init);
    ctx.fini_bld = begin_wrapper_rewrite(r, src, "generated_fini", &ofini, &fini);
    shd_rewrite_module(r);
    finish_wrapper_rewrite(r, oinit, init, ctx.init_bld);
    finish_wrapper_rewrite(r, ofini, fini, ctx.fini_bld);

    shd_destroy_rewriter(r);
    return dst;
}
