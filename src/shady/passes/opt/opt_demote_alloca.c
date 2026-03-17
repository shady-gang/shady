#include "ir_private.h"

#include "shady/pass.h"
#include "shady/visit.h"
#include "shady/ir/cast.h"
#include "shady/analysis/uses.h"
#include "shady/analysis/ptr.h"
#include "shady/print.h"

#include "log.h"
#include "portability.h"
#include "dict.h"
#include "util.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    PtrAnalysis* analysis;
    bool* todo;
} Context;

static const Node* handle_alloc(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    const Node* omem = is_mem(old) ? shd_get_parent_mem(old) : NULL;
    const AllocaInfo* k = shd_get_memory_declaration_info(ctx->analysis, old);
    if (!k->leaks && !k->read_from) {
        *ctx->todo |= true;
        const Node* new = undef(a, (Undef) { .type = shd_get_unqualified_type(shd_rewrite_node(r, old->type)) });

        const Node* nmem = shd_rewrite_node(r, omem);
        if (nmem)
            new = mem_and_value(a, (MemAndValue) { .value = new, .mem = nmem });
        return new;
    }
    const Node* new = shd_recreate_node(r, old);
    return new;
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case Function_TAG: {
            Node* fun = shd_recreate_node_head(&ctx->rewriter, old);
            Context fun_ctx = *ctx;
            fun_ctx.rewriter = shd_create_children_rewriter(&ctx->rewriter);
            fun_ctx.disable_lowering = shd_lookup_annotation_with_string_payload(old, "DisableOpt", "demote_alloca");
            if (old->payload.fun.body)
                shd_set_abstraction_body(fun, shd_rewrite_node(&fun_ctx.rewriter, old->payload.fun.body));
            shd_destroy_rewriter(&fun_ctx.rewriter);
            return fun;
        }
        case Constant_TAG: {
            Context fun_ctx = *ctx;
            return shd_recreate_node(&fun_ctx.rewriter, old);
        }
        case GlobalVariable_TAG:
        case LocalAlloc_TAG: return handle_alloc(ctx, old);
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, old);
}

bool shd_opt_demote_alloca(SHADY_UNUSED const void* unused, Module** m) {
    Module* src = *m;
    IrArena* a = shd_module_get_arena(src);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    const UsesMap* uses = shd_new_uses_map_module(src, NcType);
    bool todo = false;
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .analysis = shd_new_ptr_analysis(src, uses),
        .todo = &todo,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_ptr_analysis(ctx.analysis);
    shd_destroy_uses_map(uses);
    *m = dst;
    return todo;
}
