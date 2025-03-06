#include "l2s_private.h"

#include "shady/rewrite.h"

#include "portability.h"
#include "dict.h"
#include "list.h"
#include "log.h"
#include "arena.h"

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    Parser* p;
    Arena* arena;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Param_TAG: {
            assert(false);
        }
        case Constant_TAG: {
            Node* new = (Node*) shd_recreate_node(r, node);
            BodyBuilder* bb = shd_bld_begin_pure(a);
            const Node* value = new->payload.constant.value;
            value = scope_cast_helper(a, ctx->config->target.scopes.constants, value);
            new->payload.constant.value = shd_bld_to_instr_pure_with_values(bb, shd_singleton(value));
            return new;
        }
        case Function_TAG: {
            Function opayload = node->payload.fun;
            Function payload = shd_rewrite_function_head_payload(r, opayload);

            Nodes annotations = shd_empty(a);
            ParsedAnnotation* an = l2s_find_annotation(ctx->p, node);
            Op primop_intrinsic = PRIMOPS_COUNT;
            while (an) {
                if (strcmp(get_annotation_name(an->payload), "PrimOpIntrinsic") == 0) {
                    assert(!node->payload.fun.body);
                    Op op;
                    size_t i;
                    for (i = 0; i < PRIMOPS_COUNT; i++) {
                        if (strcmp(shd_get_primop_name(i), shd_get_annotation_string_payload(an->payload)) == 0) {
                            op = (Op) i;
                            break;
                        }
                    }
                    assert(i != PRIMOPS_COUNT);
                    primop_intrinsic = op;
                } else if (strcmp(get_annotation_name(an->payload), "EntryPoint") == 0) {
                    for (size_t i = 0; i < payload.params.count; i++) {
                        const Node* oparam = opayload.params.nodes[i];
                        const Node* nparam = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.constants, shd_rewrite_node(r, shd_get_unqualified_type(oparam->payload.param.type))));
                        payload.params = shd_change_node_at_index(a, payload.params, i, nparam);
                        shd_rewrite_annotations(r, oparam, nparam);
                    }
                }
                annotations = shd_nodes_append(a, annotations, shd_rewrite_node(r, an->payload));
                an = an->next;
            }

            Node* nfun = shd_function(r->dst_module, payload);
            shd_register_processed_list(r, get_abstraction_params(node), payload.params);
            shd_register_processed(r, node, nfun);
            nfun->annotations = annotations;
            shd_rewrite_annotations(r, node, nfun);

            if (primop_intrinsic != PRIMOPS_COUNT) {
                shd_set_abstraction_body(nfun, fn_ret(a, (Return) {
                    .args = shd_singleton(prim_op_helper(a, primop_intrinsic, shd_empty(a), get_abstraction_params(nfun))),
                    .mem = shd_get_abstraction_mem(nfun),
                }));
            } else if (get_abstraction_body(node)) {
                shd_set_abstraction_body(nfun, shd_rewrite_node(r, get_abstraction_body(node)));
            }
            return nfun;
        }
        case GlobalVariable_TAG: {
            Node* decl = shd_recreate_node_head(r, node);

            ParsedAnnotation* an = l2s_find_annotation(ctx->p, node);
            while (an) {
                shd_add_annotation(decl, shd_rewrite_node(r, an->payload));
                // NOTE: @IO is handled later by promote_io_variables
                an = an->next;
            }

            shd_register_processed(r, node, decl);
            const Node* old_init = node->payload.global_variable.init;
            if (old_init)
                decl->payload.global_variable.init = shd_rewrite_node(r, old_init);
            return decl;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

void l2s_postprocess(Parser* p, Module* src, Module* dst) {
    assert(src != dst);
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = p->config,
        .p = p,
        .arena = shd_new_arena(),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_arena(ctx.arena);
    shd_destroy_rewriter(&ctx.rewriter);
}
