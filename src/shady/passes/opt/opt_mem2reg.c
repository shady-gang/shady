#include "shady/pass.h"

#include "ir_private.h"
#include "analysis/cfg.h"

#include "list.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
    CFG* cfg;
    bool* todo;
} Context;

typedef struct {
    const Node* src;
    Nodes indices;
};

static const Node* get_ptr_source(const Node* ptr) {
    IrArena* a = ptr->arena;
    while (true) {
        switch (ptr->tag) {
            case PtrCompositeElement_TAG: {
                PtrCompositeElement payload = ptr->payload.ptr_composite_element;
                ptr = payload.ptr;
                break;
            }
            case PtrArrayElementOffset_TAG: {
                PtrArrayElementOffset payload = ptr->payload.ptr_array_element_offset;
                ptr = payload.ptr;
                break;
            }
            case PrimOp_TAG: {
                PrimOp payload = ptr->payload.prim_op;
                switch (payload.op) {
                    case reinterpret_op:
                    case convert_op: {
                        const Node* src = shd_first(payload.operands);
                        if (shd_get_unqualified_type(src->type)->tag == PtrType_TAG) {
                            ptr = src;
                            continue;
                        }
                        break;
                    }
                    default: break;
                }
                break;
            }
            default: break;
        }
        return ptr;
    }
}

static const Node* get_last_stored_value(Context* ctx, const Node* ptr, const Node* mem, const Type* expected_type) {
    const Node* ptr_source = get_ptr_source(ptr);
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
            case Store_TAG: {
                Store payload = mem->payload.store;
                if (payload.ptr == ptr)
                    return payload.value;
                if (get_ptr_source(payload.ptr) == ptr_source)
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
            const Node* src = get_ptr_source(payload.ptr);
            if (src->tag != LocalAlloc_TAG)
                break;
            // for now, only simplify loads from non-leaking allocas
            const Node* ovalue = get_last_stored_value(ctx, payload.ptr, payload.mem, shd_get_unqualified_type(node->type));
            if (ovalue) {
                *ctx->todo = true;
                const Node* value = shd_rewrite_node(r, ovalue);
                value = scope_cast_helper(a, value, shd_get_qualified_type_scope(node->type));
                return mem_and_value(a, (MemAndValue) { .mem = shd_rewrite_node(r, payload.mem), .value = value });
            }
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

bool shd_opt_mem2reg(SHADY_UNUSED const CompilerConfig* config, Module** m) {
    Module* src = *m;
    IrArena* a = shd_module_get_arena(src);

    Module* dst = NULL;
    bool todo = false;
    dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .todo = &todo
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    assert(dst);
    *m = dst;
    return todo;
}
