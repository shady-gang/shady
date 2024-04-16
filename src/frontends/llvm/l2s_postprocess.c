#include "l2s_private.h"

#include "portability.h"
#include "dict.h"
#include "log.h"

#include "../shady/rewrite.h"
#include "../shady/type.h"
#include "../shady/ir_private.h"
#include "../shady/analysis/scope.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    Parser* p;
    Scope* curr_scope;
    const Node* old_fn_or_bb;
    struct Dict* controls;
} Context;

typedef struct {
    Nodes tokens, destinations;
} Controls;

static void initialize_controls(Context* ctx, Controls* controls, const Node* fn_or_bb) {
    IrArena* a = ctx->rewriter.dst_arena;
    *controls = (Controls) {
        .destinations = empty(a),
        .tokens = empty(a),
    };
    insert_dict(const Node*, Controls*, ctx->controls, fn_or_bb, controls);
}

static const Node* wrap_in_controls(Context* ctx, Controls* controls, const Node* body) {
    IrArena* a = ctx->rewriter.dst_arena;
    if (!body)
        return NULL;
    for (size_t i = 0; i < controls->destinations.count; i++) {
        const Node* token = controls->tokens.nodes[i];
        const Node* dst = controls->destinations.nodes[i];
        Nodes o_dst_params = get_abstraction_params(dst);
        LARRAY(const Node*, new_control_params, o_dst_params.count);
        for (size_t j = 0; j < o_dst_params.count; j++)
            new_control_params[j] = var(a, o_dst_params.nodes[j]->payload.var.type, unique_name(a, "v"));
        Nodes nparams = nodes(a, o_dst_params.count, new_control_params);
        body = let(a, control(a, (Control) {
            .yield_types = get_variables_types(a, o_dst_params),
            .inside = case_(a, singleton(token), body)
        }), case_(a, nparams, jump_helper(a, rewrite_node(&ctx->rewriter, dst), nparams)));
    }
    return body;
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

bool lexical_scope_is_nested(Nodes scope, Nodes parentMaybe) {
    if (scope.count <= parentMaybe.count)
        return false;
    for (size_t i = 0; i < parentMaybe.count; i++) {
        if (scope.nodes[i] != parentMaybe.nodes[i])
            return false;
    }
    return true;
}

bool compare_nodes(Nodes* a, Nodes* b);

static Nodes remake_variables(Context* ctx, Nodes old) {
    IrArena* a = ctx->rewriter.dst_arena;
    LARRAY(const Node*, nvars, old.count);
    for (size_t i = 0; i < old.count; i++) {
        const Node* node = old.nodes[i];
            nvars[i] = var(a, node->payload.var.type ? qualified_type_helper(rewrite_node(&ctx->rewriter, node->payload.var.type), false) : NULL, node->payload.var.name);
        assert(nvars[i]->tag == Variable_TAG);
    }
    return nodes(a, old.count, nvars);
}

static const Node* process_op(Context* ctx, NodeClass op_class, String op_name, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Variable_TAG: {
            assert(node->payload.var.type);
            if (node->payload.var.type->tag == QualifiedType_TAG)
                return var(a, node->payload.var.type ? rewrite_node(&ctx->rewriter, node->payload.var.type) : NULL, node->payload.var.name);
            return var(a, qualified_type_helper(rewrite_node(&ctx->rewriter, node->payload.var.type), false), node->payload.var.name);
        }
        case Block_TAG: {
            Nodes yield_types = rewrite_nodes(r, node->payload.block.yield_types);
            const Node* ninside = rewrite_node(r, node->payload.block.inside);
            const Node* term = get_abstraction_body(ninside);
            while (term->tag == Let_TAG) {
                term = get_abstraction_body(get_let_tail(term));
            }
            assert(term->tag == Yield_TAG);
            yield_types = get_values_types(a, term->payload.yield.args);
            return block(a, (Block) {
                .yield_types = yield_types,
                .inside = ninside,
            });
        }
        case Constant_TAG: {
            Node* new = recreate_node_identity(r, node);
            BodyBuilder* bb = begin_body(a);
            const Node* value = first(bind_instruction(bb, new->payload.constant.instruction));
            value = first(bind_instruction(bb, prim_op_helper(a, subgroup_assume_uniform_op, empty(a), singleton(value))));
            new->payload.constant.instruction = yield_values_and_wrap_in_block(bb, singleton(value));
            return new;
        }
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.curr_scope = new_scope(node);
            fn_ctx.old_fn_or_bb = node;
            Controls controls;
            initialize_controls(ctx, &controls, node);
            Nodes new_params = recreate_variables(&fn_ctx.rewriter, node->payload.fun.params);
            Nodes old_annotations = node->payload.fun.annotations;
            ParsedAnnotation* an = find_annotation(ctx->p, node);
            Op primop_intrinsic = PRIMOPS_COUNT;
            while (an) {
                if (strcmp(get_annotation_name(an->payload), "PrimOpIntrinsic") == 0) {
                    assert(!node->payload.fun.body);
                    Op op;
                    size_t i;
                    for (i = 0; i < PRIMOPS_COUNT; i++) {
                        if (strcmp(get_primop_name(i), get_annotation_string_payload(an->payload)) == 0) {
                            op = (Op) i;
                            break;
                        }
                    }
                    assert(i != PRIMOPS_COUNT);
                    primop_intrinsic = op;
                } else if (strcmp(get_annotation_name(an->payload), "EntryPoint") == 0) {
                    for (size_t i = 0; i < new_params.count; i++)
                        new_params = change_node_at_index(a, new_params, i, var(a, qualified_type_helper(get_unqualified_type(new_params.nodes[i]->payload.var.type), true), new_params.nodes[i]->payload.var.name));
                }
                old_annotations = append_nodes(a, old_annotations, an->payload);
                an = an->next;
            }
            register_processed_list(&fn_ctx.rewriter, node->payload.fun.params, new_params);
            Nodes new_annotations = rewrite_nodes(&fn_ctx.rewriter, old_annotations);
            Node* decl = function(ctx->rewriter.dst_module, new_params, get_abstraction_name(node), new_annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            register_processed(&ctx->rewriter, node, decl);
            if (primop_intrinsic != PRIMOPS_COUNT) {
                decl->payload.fun.body = fn_ret(a, (Return) {
                        .args = singleton(prim_op_helper(a, primop_intrinsic, empty(a), get_abstraction_params(decl)))
                });
            } else
                decl->payload.fun.body = rewrite_node(&fn_ctx.rewriter, node->payload.fun.body);
            destroy_scope(fn_ctx.curr_scope);
            return decl;
        }
        case BasicBlock_TAG: {
            Context bb_ctx = *ctx;
            bb_ctx.old_fn_or_bb = node;
            Controls controls;
            initialize_controls(ctx, &controls, node);
            Node* new_bb = (Node*) recreate_node_identity(&bb_ctx.rewriter, node);
            // new_bb->payload.basic_block.body = wrap_in_controls(ctx, &controls, new_bb->payload.basic_block.body);
            return new_bb;
        }
        case Jump_TAG: {
            const Node* src = ctx->old_fn_or_bb;
            const Node* dst = node->payload.jump.target;
            assert(src && dst);
            rewrite_node(&ctx->rewriter, dst);

            if (!ctx->config->hacks.recover_structure)
                break;
            Nodes* src_lexical_scope = find_value_dict(const Node*, Nodes, ctx->p->scopes, src);
            Nodes* dst_lexical_scope = find_value_dict(const Node*, Nodes, ctx->p->scopes, dst);
            if (!src_lexical_scope) {
                warn_print("Failed to find jump source node ");
                log_node(WARN, src);
                warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
            } else if (!dst_lexical_scope) {
                warn_print("Failed to find jump target node ");
                log_node(WARN, dst);
                warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
            } else if (lexical_scope_is_nested(*src_lexical_scope, *dst_lexical_scope)) {
                debug_print("Jump from %s to %s exits one or more nested lexical scopes, it might reconverge.\n", get_abstraction_name(src), get_abstraction_name(dst));

                CFNode* src_cfnode = scope_lookup(ctx->curr_scope, src);
                assert(src_cfnode->node);
                CFNode* target_cfnode = scope_lookup(ctx->curr_scope, dst);
                assert(src_cfnode && target_cfnode);
                CFNode* dom = src_cfnode->idom;
                while (dom) {
                    if (dom->node->tag == BasicBlock_TAG || dom->node->tag == Function_TAG) {
                        debug_print("Considering %s as a location for control\n", get_abstraction_name(dom->node));
                        Nodes* dom_lexical_scope = find_value_dict(const Node*, Nodes, ctx->p->scopes, dom->node);
                        if (!dom_lexical_scope) {
                            warn_print("Basic block %s did not have an entry in the lexical_scopes map. Is debug information enabled ?\n", get_abstraction_name(dom->node));
                        } else if (lexical_scope_is_nested(*dst_lexical_scope, *dom_lexical_scope)) {
                            error_print("We went up too far: %s is a parent of the jump destination scope.\n", get_abstraction_name(dom->node));
                        } else if (compare_nodes(dom_lexical_scope, dst_lexical_scope)) {
                            debug_print("We need to introduce a control() block at %s, pointing at %s\n.", get_abstraction_name(dom->node), get_abstraction_name(dst));
                            Controls** found = find_value_dict(const Node, Controls*, ctx->controls, dom->node);
                            if (found) {
                                Controls* controls = *found;
                                const Node* join_token = NULL;
                                for (size_t i = 0; i < controls->destinations.count; i++) {
                                    if (controls->destinations.nodes[i] == dst) {
                                        join_token = controls->tokens.nodes[i];
                                        break;
                                    }
                                }
                                if (!join_token) {
                                    const Type* jp_type = join_point_type(a, (JoinPointType) {
                                        .yield_types = get_variables_types(a, get_abstraction_params(dst))
                                    });
                                    join_token = var(a, qualified_type_helper(jp_type, false), get_abstraction_name(dst));
                                    controls->tokens = append_nodes(a, controls->tokens, join_token);
                                    controls->destinations = append_nodes(a, controls->destinations, dst);
                                }
                                Nodes nparams = remake_variables(ctx, get_abstraction_params(dst));
                                //register_processed_list(&ctx->rewriter, get_abstraction_params(dst), nparams);
                                Node* fn = src;
                                if (fn->tag == BasicBlock_TAG)
                                    fn = (Node*) fn->payload.basic_block.fn;
                                assert(fn->tag == Function_TAG);
                                fn = rewrite_node(r, fn);
                                Node* wrapper = basic_block(a, fn, nparams, format_string_arena(a->arena, "wrapper_to_%s", get_abstraction_name(dst)));
                                wrapper->payload.basic_block.body = join(a, (Join) {
                                    .args = nparams,
                                    .join_point = join_token
                                });
                                return jump_helper(a, wrapper, rewrite_nodes(&ctx->rewriter, node->payload.jump.args));
                            } else {
                                assert(false);
                            }
                        } else {
                            dom = dom->idom;
                            continue;
                        }
                        break;
                    }
                    dom = dom->idom;
                }
            }
            break;
        }
        case GlobalVariable_TAG: {
            if (lookup_annotation(node, "LLVMMetaData"))
                return NULL;
            AddressSpace as = node->payload.global_variable.address_space;
            const Node* old_init = node->payload.global_variable.init;
            Nodes annotations = rewrite_nodes(r, node->payload.global_variable.annotations);
            const Type* type = rewrite_node(r, node->payload.global_variable.type);
            ParsedAnnotation* an = find_annotation(ctx->p, node);
            while (an) {
                annotations = append_nodes(a, annotations, rewrite_node(r, an->payload));
                if (strcmp(get_annotation_name(an->payload), "Builtin") == 0)
                    old_init = NULL;
                if (strcmp(get_annotation_name(an->payload), "UniformConstant") == 0)
                    as = AsUniformConstant;
                an = an->next;
            }
            Node* decl = global_var(ctx->rewriter.dst_module, annotations, type, get_declaration_name(node), as);
            register_processed(r, node, decl);
            if (old_init)
                decl->payload.global_variable.init = rewrite_node(r, old_init);
            return decl;
        }
        default: break;
    }

    if (op_class == NcTerminator && node->tag != Let_TAG) {
        Controls** found = find_value_dict(const Node, Controls*, ctx->controls, ctx->old_fn_or_bb);
        assert(found);
        Controls* controls = *found;
        return wrap_in_controls(ctx, controls, recreate_node_identity(&ctx->rewriter, node));
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* process_node(Context* ctx, const Node* old) {
    return process_op(ctx, 0, NULL, old);
}

void postprocess(Parser* p, Module* src, Module* dst) {
    assert(src != dst);
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = p->config,
        .p = p,
        .controls = new_dict(const Node*, Controls*, (HashFn) hash_node, (CmpFn) compare_node),
    };

    ctx.rewriter.rewrite_op_fn = (RewriteOpFn) process_op;
    ctx.rewriter.config.process_variables = true;
    ctx.rewriter.config.search_map = true;
    // ctx.rewriter.config.write_map = false;

    rewrite_module(&ctx.rewriter);
    destroy_dict(ctx.controls);
    destroy_rewriter(&ctx.rewriter);
}
