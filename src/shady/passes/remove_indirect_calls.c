#include "shady/ir.h"

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
        case Root_TAG: {
            Nodes old_decls = node->payload.root.declarations;
            size_t new_decls_count = 0;
            LARRAY(const Node*, decls, old_decls.count);
            for (size_t i = 0; i < old_decls.count; i++) {
                if (old_decls.nodes[i]->tag == Constant_TAG) continue;
                decls[new_decls_count++] = process(ctx, old_decls.nodes[i]);
            }
            return root(arena, (Root) { .declarations = nodes(arena, new_decls_count, decls) });
        }
        case Call_TAG: {
            if (!node->payload.call_instr.is_indirect)
                goto skip;
            const Node* ocallee = node->payload.call_instr.callee;
            if (ocallee->tag != FnAddr_TAG)
                goto skip;
            ocallee = ocallee->payload.fn_addr.fn;
            assert(ocallee && ocallee->tag == Function_TAG);
            CGNode* callee_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
            if (callee_node->is_recursive || callee_node->is_address_captured)
                goto skip;
            debug_print("Call to %s is not recursive, turning the call direct\n", ocallee->payload.fn.name);
            Nodes nargs = rewrite_nodes(&ctx->rewriter, node->payload.call_instr.args);
            return call_instr(ctx->rewriter.dst_arena, (Call) {
                .is_indirect = false,
                .callee = process(ctx, ocallee),
                .args = nargs
            });
        }
        case Function_TAG: {
            if (!node->payload.fn.is_basic_block) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, node);
                Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fn.annotations);
                if (fn_node->is_address_captured || fn_node->is_recursive) {
                    annotations = append_nodes(arena, annotations, annotation(arena, (Annotation) {
                        .name = "IndirectlyCalled",
                        .payload_type = AnPayloadNone
                    }));
                }
                Node* new = fn(arena, annotations, node->payload.fn.name, node->payload.fn.is_basic_block, recreate_variables(&ctx->rewriter, node->payload.fn.params), rewrite_nodes(&ctx->rewriter, node->payload.fn.return_types));
                for (size_t i = 0; i < new->payload.fn.params.count; i++)
                    register_processed(&ctx->rewriter, node->payload.fn.params.nodes[i], new->payload.fn.params.nodes[i]);
                register_processed(&ctx->rewriter, node, new);

                recreate_decl_body_identity(&ctx->rewriter, node, new);
                return new;
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        skip:
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* remove_indirect_calls(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    Context ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_fn = (RewriteFn) process,
            .processed = done,
        },
        .graph = get_callgraph(src_program)
    };
    const Node* rewritten = process(&ctx, src_program);
    destroy_dict(done);
    dispose_callgraph(ctx.graph);
    return rewritten;
}
