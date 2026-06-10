#include "shady/pass.h"

#include "shady/analysis/ptr.h"

#include "ir_private.h"
#include "analysis/cfg.h"

#include "list.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
    CFG* cfg;
    bool* todo;

    PtrAnalysis* ptr_analysis;
} Context;

typedef struct {
    const Node* src;
    Nodes indices;
};

static const Node* get_last_stored_value(Context* ctx, const Node* ptr, const Node* mem, const AllocaInfo* ptr_alloca_info) {
    while (mem) {
        switch (mem->tag) {
            case AbsMem_TAG: {
                const Node* abs = mem->payload.abs_mem.abs;
                CFNode* n = shd_cfg_lookup(ctx->cfg, abs);
                if (shd_list_count(n->pred_edges) == 1) {
                    CFEdge e = shd_read_list(CFEdge, n->pred_edges)[0];
                    mem = get_terminator_mem(e.terminator);
                    continue;
                }
                break;
            }
            case IndirectCall_TAG:
            case Call_TAG: {
                // global variables don't quite "leak" but other functions can touch them directly
                if (ptr_alloca_info->node->tag != LocalAlloc_TAG) {
                    return NULL;
                }
                break;
            }
            case Store_TAG: {
                Store payload = mem->payload.store;
                if (payload.ptr == ptr)
                    return payload.value;
                if (shd_find_memory_declaration(ctx->ptr_analysis, payload.ptr, true) == ptr_alloca_info)
                    return NULL;
                break;
            }
            default: break;
        }
        mem = shd_get_parent_mem(mem);
    }
    return NULL;
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Node* new = shd_recreate_node_head(r, node);
            Context fun_ctx = *ctx;
            fun_ctx.cfg = build_fn_cfg(node);
            shd_recreate_node_body(&fun_ctx.rewriter, node, new);
            shd_destroy_cfg(fun_ctx.cfg);
            return new;
        }
        case Load_TAG: {
            Load payload = node->payload.load;
            const AllocaInfo* alloca_info = shd_find_memory_declaration(ctx->ptr_analysis, payload.ptr, true);
            // for now, only simplify loads from non-leaking allocas
            if (!alloca_info || alloca_info->leaks)
                break;
            const Node* ovalue = get_last_stored_value(ctx, payload.ptr, payload.mem, alloca_info);
            if (ovalue) {
                *ctx->todo = true;
                const Node* value = shd_rewrite_node(r, ovalue);
                value = scope_cast_helper(a, shd_get_qualified_type_scope(node->type), value);
                return mem_and_value(a, (MemAndValue) { .mem = shd_rewrite_node(r, payload.mem), .value = value });
            }
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

bool shd_opt_mem2reg(SHADY_UNUSED void* unused, Module** m) {
    Module* src = *m;
    IrArena* a = shd_module_get_arena(src);

    const UsesMap* uses = shd_new_uses_map_module(src, 0);
    PtrAnalysis* ptr_analysis = shd_new_ptr_analysis(src, uses);

    Module* dst = NULL;
    bool todo = false;
    dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .todo = &todo,
        .ptr_analysis = ptr_analysis,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_ptr_analysis(ptr_analysis);
    shd_destroy_uses_map(uses);
    assert(dst);
    *m = dst;
    return todo;
}
