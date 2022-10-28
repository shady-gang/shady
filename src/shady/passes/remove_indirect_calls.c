#include "passes.h"

#include "dict.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"

#include "../analysis/callgraph.h"

typedef struct {
    Rewriter rewriter;
    CallGraph* graph;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Call_TAG: {
            if (!node->payload.call_instr.is_indirect)
                goto skip;
            const Node* ocallee = node->payload.call_instr.callee;
            if (ocallee->tag != FnAddr_TAG)
                goto skip;
            ocallee = ocallee->payload.fn_addr.fn;
            assert(ocallee && ocallee->tag == Lambda_TAG);
            CGNode* callee_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
            if (callee_node->is_recursive || callee_node->is_address_captured)
                goto skip;
            debug_print("Call to %s is not recursive, turning the call direct\n", ocallee->payload.lam.name);
            Nodes nargs = rewrite_nodes(&ctx->rewriter, node->payload.call_instr.args);
            return call_instr(ctx->rewriter.dst_arena, (Call) {
                .is_indirect = false,
                .callee = process(ctx, ocallee),
                .args = nargs
            });
        }
        case Function_TAG: {
            CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, node);
            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.lam.annotations);
            if (fn_node->is_address_captured || fn_node->is_recursive) {
                annotations = append_nodes(arena, annotations, annotation(arena, (Annotation) {
                    .name = "IndirectlyCalled",
                    .payload_type = AnPayloadNone
                }));
            }
            Node* new = function(ctx->rewriter.dst_module, recreate_variables(&ctx->rewriter, node->payload.lam.params), node->payload.lam.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.lam.return_types));
            for (size_t i = 0; i < new->payload.lam.params.count; i++)
                register_processed(&ctx->rewriter, node->payload.lam.params.nodes[i], new->payload.lam.params.nodes[i]);
            register_processed(&ctx->rewriter, node, new);

            recreate_decl_body_identity(&ctx->rewriter, node, new);
            return new;
        }
        skip:
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void remove_indirect_calls(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .graph = get_callgraph(src)
    };
    rewrite_module(&ctx.rewriter);
    dispose_callgraph(ctx.graph);
    destroy_rewriter(&ctx.rewriter);
}
