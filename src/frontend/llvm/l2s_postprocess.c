#include "l2s_private.h"

#include "portability.h"
#include "dict.h"
#include "log.h"

#include "../shady/rewrite.h"
#include "../shady/type.h"
#include "../shady/ir_private.h"
#include "../shady/analysis/cfg.h"
#include "../shady/transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    Parser* p;
    CFG* cfg;
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

static void wrap_in_controls(Context* ctx, Controls* controls, Node* nabs, const Node* oabs) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    const Node* obody = get_abstraction_body(oabs);
    if (!obody)
        return;
    // const Node* mem = get_abstraction_mem(nabs);
    Node* c = case_(a, empty(a));
    register_processed(r, get_abstraction_mem(oabs), get_abstraction_mem(c));
    set_abstraction_body(c, rewrite_node(r, obody));
    for (size_t i = 0; i < controls->destinations.count; i++) {
        const Node* token = controls->tokens.nodes[i];
        const Node* dst = controls->destinations.nodes[i];
        Nodes o_dst_params = get_abstraction_params(dst);
        Node* control_case = case_(a, singleton(token));
        set_abstraction_body(control_case, jump_helper(a, c, empty(a), get_abstraction_mem(control_case)));

        Node* c2 = case_(a, empty(a));
        BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(c2));
        Nodes results = gen_control(bb, get_param_types(a, o_dst_params), control_case);
        set_abstraction_body(c2, finish_body(bb, jump_helper(a, rewrite_node(&ctx->rewriter, dst), results, bb_mem(bb))));
        c = c2;
    }
    const Node* body = jump_helper(a, c, empty(a), get_abstraction_mem(nabs));
    return set_abstraction_body(nabs, body);
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

static Nodes remake_params(Context* ctx, Nodes old) {
    IrArena* a = ctx->rewriter.dst_arena;
    LARRAY(const Node*, nvars, old.count);
    for (size_t i = 0; i < old.count; i++) {
        const Node* node = old.nodes[i];
            nvars[i] = param(a, node->payload.param.type ? qualified_type_helper(rewrite_node(&ctx->rewriter, node->payload.param.type), false) : NULL, node->payload.param.name);
        assert(nvars[i]->tag == Param_TAG);
    }
    return nodes(a, old.count, nvars);
}

static const Node* process_op(Context* ctx, NodeClass op_class, String op_name, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Param_TAG: {
            assert(node->payload.param.type);
            if (node->payload.param.type->tag == QualifiedType_TAG)
                return param(a, node->payload.param.type ? rewrite_node(&ctx->rewriter, node->payload.param.type) : NULL, node->payload.param.name);
            return param(a, qualified_type_helper(rewrite_node(&ctx->rewriter, node->payload.param.type), false), node->payload.param.name);
        }
        case Constant_TAG: {
            Node* new = (Node*) recreate_node_identity(r, node);
            BodyBuilder* bb = begin_block_pure(a);
            const Node* value = first(bind_instruction(bb, new->payload.constant.value));
            value = first(bind_instruction(bb, prim_op_helper(a, subgroup_assume_uniform_op, empty(a), singleton(value))));
            new->payload.constant.value = yield_values_and_wrap_in_compound_instruction(bb, singleton(value));
            return new;
        }
        case PrimOp_TAG: {
            Nodes old_operands = node->payload.prim_op.operands;
            switch (node->payload.prim_op.op) {
                default: break;
            }
            break;
        }
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.cfg = build_fn_cfg(node);
            fn_ctx.old_fn_or_bb = node;
            Controls controls;
            initialize_controls(ctx, &controls, node);
            Nodes new_params = recreate_params(&fn_ctx.rewriter, node->payload.fun.params);
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
                        new_params = change_node_at_index(a, new_params, i, param(a, qualified_type_helper(get_unqualified_type(new_params.nodes[i]->payload.param.type), true), new_params.nodes[i]->payload.param.name));
                }
                old_annotations = append_nodes(a, old_annotations, an->payload);
                an = an->next;
            }
            register_processed_list(&fn_ctx.rewriter, node->payload.fun.params, new_params);
            Nodes new_annotations = rewrite_nodes(&fn_ctx.rewriter, old_annotations);
            Node* decl = function(ctx->rewriter.dst_module, new_params, get_abstraction_name(node), new_annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            register_processed(&ctx->rewriter, node, decl);
            if (primop_intrinsic != PRIMOPS_COUNT) {
                set_abstraction_body(decl, fn_ret(a, (Return) {
                        .args = singleton(prim_op_helper(a, primop_intrinsic, empty(a), get_abstraction_params(decl))),
                        .mem = get_abstraction_mem(decl),
                }));
            } else
                wrap_in_controls(ctx, &controls, decl, node);
            destroy_cfg(fn_ctx.cfg);
            return decl;
        }
        case BasicBlock_TAG: {
            Context bb_ctx = *ctx;
            bb_ctx.old_fn_or_bb = node;
            Controls controls;
            initialize_controls(ctx, &controls, node);
            Nodes nparams = recreate_params(&ctx->rewriter, get_abstraction_params(node));
            register_processed_list(r, get_abstraction_params(node), nparams);
            Node* new_bb = (Node*) basic_block(a, nparams, get_abstraction_name_unsafe(node));
            register_processed(r, node, new_bb);
            wrap_in_controls(ctx, &controls, new_bb, node);
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
            bool src_is_wrapper = find_value_dict(const Node*, const Node*, ctx->p->wrappers_map, src);
            const Node** found_dst_wrapper = find_value_dict(const Node*, const Node*, ctx->p->wrappers_map, dst);
            if (found_dst_wrapper)
                dst = *found_dst_wrapper;
            Nodes* dst_lexical_scope = find_value_dict(const Node*, Nodes, ctx->p->scopes, dst);
            if (src_is_wrapper) {
                // silent
            } else if (!src_lexical_scope) {
                warn_print("Failed to find jump source node ");
                log_node(WARN, src);
                warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
            } else if (!dst_lexical_scope) {
                warn_print("Failed to find jump target node ");
                log_node(WARN, dst);
                warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
            } else if (lexical_scope_is_nested(*src_lexical_scope, *dst_lexical_scope)) {
                debug_print("Jump from %s to %s exits one or more nested lexical scopes, it might reconverge.\n", get_abstraction_name_safe(src), get_abstraction_name_safe(dst));

                CFNode* src_cfnode = cfg_lookup(ctx->cfg, src);
                assert(src_cfnode->node);
                CFNode* target_cfnode = cfg_lookup(ctx->cfg, dst);
                assert(src_cfnode && target_cfnode);
                CFNode* dom = src_cfnode->idom;
                while (dom) {
                    if (dom->node->tag == BasicBlock_TAG || dom->node->tag == Function_TAG) {
                        debug_print("Considering %s as a location for control\n", get_abstraction_name_safe(dom->node));
                        Nodes* dom_lexical_scope = find_value_dict(const Node*, Nodes, ctx->p->scopes, dom->node);
                        if (!dom_lexical_scope) {
                            warn_print("Basic block %s did not have an entry in the lexical_scopes map. Is debug information enabled ?\n", get_abstraction_name_safe(dom->node));
                            dom = dom->idom;
                            continue;
                        } else if (lexical_scope_is_nested(*dst_lexical_scope, *dom_lexical_scope)) {
                            error_print("We went up too far: %s is a parent of the jump destination scope.\n", get_abstraction_name_safe(dom->node));
                        } else if (compare_nodes(dom_lexical_scope, dst_lexical_scope)) {
                            debug_print("We need to introduce a control block at %s, pointing at %s\n.", get_abstraction_name_safe(dom->node), get_abstraction_name_safe(dst));
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
                                        .yield_types = get_param_types(a, get_abstraction_params(dst))
                                    });
                                    join_token = param(a, qualified_type_helper(jp_type, false), get_abstraction_name_unsafe(dst));
                                    controls->tokens = append_nodes(a, controls->tokens, join_token);
                                    controls->destinations = append_nodes(a, controls->destinations, dst);
                                }
                                Nodes nparams = remake_params(ctx, get_abstraction_params(dst));
                                //register_processed_list(&ctx->rewriter, get_abstraction_params(dst), nparams);
                                Node* wrapper = basic_block(a, nparams, format_string_arena(a->arena, "wrapper_to_%s", get_abstraction_name_safe(dst)));
                                wrapper->payload.basic_block.body = join(a, (Join) {
                                    .args = nparams,
                                    .join_point = join_token,
                                    .mem = get_abstraction_mem(wrapper),
                                });
                                return jump_helper(a, wrapper, rewrite_nodes(&ctx->rewriter, node->payload.jump.args), rewrite_node(r, node->payload.jump.mem));
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
            AddressSpace old_as = as;
            while (an) {
                annotations = append_nodes(a, annotations, rewrite_node(r, an->payload));
                if (strcmp(get_annotation_name(an->payload), "Builtin") == 0)
                    old_init = NULL;
                if (strcmp(get_annotation_name(an->payload), "AddressSpace") == 0)
                    as = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(an->payload)), false);
                an = an->next;
            }
            Node* decl = global_var(ctx->rewriter.dst_module, annotations, type, get_declaration_name(node), as);
            Node* result = decl;
            if (old_as != as) {
                const Type* pt = ptr_type(a, (PtrType) { .address_space = old_as, .pointed_type = type });
                Node* c = constant(ctx->rewriter.dst_module, singleton(annotation(a, (Annotation) {
                    .name = "Inline"
                })), pt, format_string_interned(a, "%s_proxy", get_declaration_name(decl)));
                c->payload.constant.value = prim_op_helper(a, convert_op, singleton(pt), singleton(
                        ref_decl_helper(a, decl)));
                result = c;
            }

            register_processed(r, node, result);
            if (old_init)
                decl->payload.global_variable.init = rewrite_node(r, old_init);
            return result;
        }
        default: break;
    }

    // This is required so we don't wrap jumps that are part of branches!
    // if (ctx->old_fn_or_bb && op_class == NcTerminator) {
    //     Controls** found = find_value_dict(const Node, Controls*, ctx->controls, ctx->old_fn_or_bb);
    //     assert(found);
    //     Controls* controls = *found;
    //     return wrap_in_controls(ctx, controls, recreate_node_identity(&ctx->rewriter, node));
    // }

    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* process_node(Context* ctx, const Node* old) {
    return process_op(ctx, 0, NULL, old);
}


void postprocess(Parser* p, Module* src, Module* dst) {
    assert(src != dst);
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = p->config,
        .p = p,
        .controls = new_dict(const Node*, Controls*, (HashFn) hash_node, (CmpFn) compare_node),
    };

    ctx.rewriter.config.process_params = true;
    ctx.rewriter.config.search_map = true;
    ctx.rewriter.rewrite_op_fn = (RewriteOpFn) process_op;
    // ctx.rewriter.config.write_map = false;

    rewrite_module(&ctx.rewriter);
    destroy_dict(ctx.controls);
    destroy_rewriter(&ctx.rewriter);
}
