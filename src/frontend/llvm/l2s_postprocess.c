#include "l2s_private.h"

#include "portability.h"
#include "dict.h"
#include "list.h"
#include "log.h"
#include "arena.h"

#include "../shady/rewrite.h"
#include "../shady/type.h"

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    Parser* p;
    Arena* arena;
} Context;

static Nodes remake_params(Context* ctx, Nodes old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    LARRAY(const Node*, nvars, old.count);
    for (size_t i = 0; i < old.count; i++) {
        const Node* node = old.nodes[i];
        const Type* t = NULL;
        if (node->payload.param.type) {
            if (node->payload.param.type->tag == QualifiedType_TAG)
                t = shd_rewrite_node(r, node->payload.param.type);
            else
                t = shd_as_qualified_type(shd_rewrite_node(r, node->payload.param.type), false);
        }
        nvars[i] = param(a, t, node->payload.param.name);
        assert(nvars[i]->tag == Param_TAG);
    }
    return shd_nodes(a, old.count, nvars);
}

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Param_TAG: {
            assert(false);
        }
        case Constant_TAG: {
            Node* new = (Node*) shd_recreate_node(r, node);
            BodyBuilder* bb = begin_block_pure(a);
            const Node* value = new->payload.constant.value;
            value = prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(value));
            new->payload.constant.value = yield_values_and_wrap_in_compound_instruction(bb, shd_singleton(value));
            return new;
        }
        case Function_TAG: {
            Nodes new_params = remake_params(ctx, node->payload.fun.params);
            Nodes old_annotations = node->payload.fun.annotations;
            ParsedAnnotation* an = find_annotation(ctx->p, node);
            Op primop_intrinsic = PRIMOPS_COUNT;
            while (an) {
                if (strcmp(get_annotation_name(an->payload), "PrimOpIntrinsic") == 0) {
                    assert(!node->payload.fun.body);
                    Op op;
                    size_t i;
                    for (i = 0; i < PRIMOPS_COUNT; i++) {
                        if (strcmp(get_primop_name(i), shd_get_annotation_string_payload(an->payload)) == 0) {
                            op = (Op) i;
                            break;
                        }
                    }
                    assert(i != PRIMOPS_COUNT);
                    primop_intrinsic = op;
                } else if (strcmp(get_annotation_name(an->payload), "EntryPoint") == 0) {
                    for (size_t i = 0; i < new_params.count; i++)
                        new_params = shd_change_node_at_index(a, new_params, i, param(a, shd_as_qualified_type(
                                get_unqualified_type(new_params.nodes[i]->payload.param.type), true), new_params.nodes[i]->payload.param.name));
                }
                old_annotations = shd_nodes_append(a, old_annotations, an->payload);
                an = an->next;
            }
            shd_register_processed_list(r, node->payload.fun.params, new_params);
            Nodes new_annotations = shd_rewrite_nodes(r, old_annotations);
            Node* decl = function(ctx->rewriter.dst_module, new_params, get_abstraction_name(node), new_annotations, shd_rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            shd_register_processed(&ctx->rewriter, node, decl);
            if (primop_intrinsic != PRIMOPS_COUNT) {
                set_abstraction_body(decl, fn_ret(a, (Return) {
                    .args = shd_singleton(prim_op_helper(a, primop_intrinsic, shd_empty(a), get_abstraction_params(decl))),
                    .mem = get_abstraction_mem(decl),
                }));
            } else if (get_abstraction_body(node))
                set_abstraction_body(decl, shd_rewrite_node(r, get_abstraction_body(node)));
            return decl;
        }
        case GlobalVariable_TAG: {
            // if (lookup_annotation(node, "LLVMMetaData"))
            //     return NULL;
            AddressSpace as = node->payload.global_variable.address_space;
            const Node* old_init = node->payload.global_variable.init;
            Nodes annotations = shd_rewrite_nodes(r, node->payload.global_variable.annotations);
            const Type* type = shd_rewrite_node(r, node->payload.global_variable.type);
            ParsedAnnotation* an = find_annotation(ctx->p, node);
            AddressSpace old_as = as;
            while (an) {
                annotations = shd_nodes_append(a, annotations, shd_rewrite_node(r, an->payload));
                if (strcmp(get_annotation_name(an->payload), "Builtin") == 0)
                    old_init = NULL;
                if (strcmp(get_annotation_name(an->payload), "AddressSpace") == 0)
                    as = get_int_literal_value(*resolve_to_int_literal(shd_get_annotation_value(an->payload)), false);
                an = an->next;
            }
            Node* decl = global_var(ctx->rewriter.dst_module, annotations, type, get_declaration_name(node), as);
            Node* result = decl;
            if (old_as != as) {
                const Type* pt = ptr_type(a, (PtrType) { .address_space = old_as, .pointed_type = type });
                Node* c = constant(ctx->rewriter.dst_module, shd_singleton(annotation(a, (Annotation) {
                    .name = "Inline"
                })), pt, shd_fmt_string_irarena(a, "%s_proxy", get_declaration_name(decl)));
                c->payload.constant.value = prim_op_helper(a, convert_op, shd_singleton(pt), shd_singleton(
                    ref_decl_helper(a, decl)));
                result = c;
            }

            shd_register_processed(r, node, result);
            if (old_init)
                decl->payload.global_variable.init = shd_rewrite_node(r, old_init);
            return result;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

void postprocess(Parser* p, Module* src, Module* dst) {
    assert(src != dst);
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = p->config,
        .p = p,
        .arena = shd_new_arena(),
    };

    ctx.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_arena(ctx.arena);
    shd_destroy_rewriter(&ctx.rewriter);
}
