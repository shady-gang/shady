#include "shady/pass.h"
#include "shady/dict.h"
#include "shady/ir/cast.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
    Node2Node map;
} Context;

static const Node* make_nullptr(Context* ctx, const Type* t) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* found = shd_node2node_find(ctx->map, t);
    if (found)
        return found;

    BodyBuilder* bb = shd_bld_begin_pure(a);
    const Node* nul = shd_bld_reinterpret_cast(bb, t, shd_uint64_literal(a, 0));
    Node* decl = constant_helper(ctx->rewriter.dst_module, t);
    shd_set_debug_name(decl, shd_fmt_string_irarena(a, "nullptr_%s", shd_get_type_name(a, t)));
    shd_add_annotation_named(decl, "Generated");
    decl->payload.constant.value = shd_bld_to_instr_pure_with_values(bb, shd_singleton(nul));
    const Node* ref = decl;
    shd_node2node_insert(ctx->map, t, ref);
    return ref;
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case NullPtr_TAG: {
            const Type* t = shd_rewrite_node(r, node->payload.null_ptr.ptr_type);
            assert(t->tag == PtrType_TAG);
            return make_nullptr(ctx, t);
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

///TODO: The way this pass is implemented is dubious, shouldn't null refs turn into undef ?
Module* shd_pass_lower_nullptr(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .map = shd_new_node2node(),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_node2node(ctx.map);
    return dst;
}
