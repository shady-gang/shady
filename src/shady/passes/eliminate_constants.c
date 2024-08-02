#include "pass.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

typedef struct {
    Rewriter rewriter;
    BodyBuilder* bb;
    bool all;
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
        case Constant_TAG:
            if (!node->payload.constant.instruction)
                break;
            if (!ctx->all && !lookup_annotation(node, "Inline"))
                break;
            return NULL;
        case RefDecl_TAG: {
            const Node* decl = node->payload.ref_decl.decl;
            if (decl->tag == Constant_TAG && decl->payload.constant.instruction) {
                const Node* value = resolve_node_to_definition(decl->payload.constant.instruction, (NodeResolveConfig) { 0 });
                if (value)
                    return rewrite_node(&ctx->rewriter, value);
                c.rewriter.map = clone_dict(c.rewriter.map);
                assert(ctx->bb);
                // TODO: actually _copy_ the instruction so we can duplicate the code safely!
                const Node* rewritten = first(bind_instruction(ctx->bb, rewrite_node(&ctx->rewriter, decl->payload.constant.instruction)));
                destroy_dict(c.rewriter.map);
                return rewritten;
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

static Module* eliminate_constants_(SHADY_UNUSED const CompilerConfig* config, Module* src, bool all) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .all = all,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}

Module* eliminate_constants(const CompilerConfig* config, Module* src) {
    return eliminate_constants_(config, src, true);
}

Module* eliminate_inlineable_constants(const CompilerConfig* config, Module* src) {
    return eliminate_constants_(config, src, false);
}
