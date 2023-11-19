#include "passes.h"

#include "../rewrite.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    BodyBuilder* bb;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;
    IrArena* a = ctx->rewriter.dst_arena;

    BodyBuilder* abs_bb = NULL;
    Context c = *ctx;
    ctx = &c;
    if (is_abstraction(node)) {
        c.bb = abs_bb = begin_body(a);
    }

    switch (node->tag) {
        case Constant_TAG: return NULL;
        case RefDecl_TAG: {
            if (node->payload.ref_decl.decl->tag == Constant_TAG) {
                // TODO: actually _copy_ the instruction so we can duplicate the code safely!
                return process(ctx, node->payload.ref_decl.decl->payload.constant.instruction);
            }
            break;
        }
        default: break;
    }

    Node* new = (Node*) recreate_node_identity(&ctx->rewriter, node);
    if (abs_bb) {
        assert(is_abstraction(new));
        if (get_abstraction_body(new))
            set_abstraction_body(new, finish_body(abs_bb, get_abstraction_body(new)));
        else
            cancel_body(abs_bb);
    }
    return new;
}

Module* eliminate_constants(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process)
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
