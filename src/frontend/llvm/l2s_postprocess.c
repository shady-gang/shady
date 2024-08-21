#include "l2s_private.h"

#include "portability.h"
#include "dict.h"
#include "list.h"
#include "log.h"
#include "arena.h"

#include "../shady/rewrite.h"
#include "../shady/type.h"
#include "../shady/ir_private.h"
#include "../shady/analysis/cfg.h"
#include "../shady/transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    Parser* p;
    Arena* arena;
    struct Dict* controls;
    struct Dict* jump2wrapper;
} Context;

typedef struct {
    Nodes tokens, destinations;
} Controls;

static Nodes remake_params(Context* ctx, Nodes old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    LARRAY(const Node*, nvars, old.count);
    for (size_t i = 0; i < old.count; i++) {
        const Node* node = old.nodes[i];
        const Type* t = NULL;
        if (node->payload.param.type) {
            if (node->payload.param.type->tag == QualifiedType_TAG)
                t = rewrite_node(r, node->payload.param.type);
            else
                t = qualified_type_helper(rewrite_node(r, node->payload.param.type), false);
        }
        nvars[i] = param(a, t, node->payload.param.name);
        assert(nvars[i]->tag == Param_TAG);
    }
    return nodes(a, old.count, nvars);
}

static Controls* get_or_create_controls(Context* ctx, const Node* fn_or_bb) {
    Controls** found = find_value_dict(const Node, Controls*, ctx->controls, fn_or_bb);
    if (found)
        return *found;
    IrArena* a = ctx->rewriter.dst_arena;
    Controls* controls = arena_alloc(ctx->arena, sizeof(Controls));
    *controls = (Controls) {
        .destinations = empty(a),
        .tokens = empty(a),
    };
    insert_dict(const Node*, Controls*, ctx->controls, fn_or_bb, controls);
    return controls;
}

static void wrap_in_controls(Context* ctx, Node* nabs, const Node* oabs) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    const Node* obody = get_abstraction_body(oabs);
    if (!obody)
        return;

    Controls** found = find_value_dict(const Node, Controls*, ctx->controls, oabs);
    if (found) {
        Controls* controls = *found;
        Node* c = case_(a, empty(a));
        register_processed(r, get_abstraction_mem(oabs), get_abstraction_mem(c));
        set_abstraction_body(c, rewrite_node(r, obody));
        for (size_t i = 0; i < controls->destinations.count; i++) {
            const Node* token = controls->tokens.nodes[i];
            const Node* dst = controls->destinations.nodes[i];
            Node* control_case = case_(a, singleton(token));
            set_abstraction_body(control_case, jump_helper(a, c, empty(a), get_abstraction_mem(control_case)));

            Node* c2 = case_(a, empty(a));
            BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(c2));
            Nodes results = gen_control(bb, get_param_types(a, get_abstraction_params(dst)), control_case);
            set_abstraction_body(c2, finish_body(bb, jump_helper(a, rewrite_node(&ctx->rewriter, dst), results, bb_mem(bb))));
            c = c2;
        }
        const Node* body = jump_helper(a, c, empty(a), get_abstraction_mem(nabs));
        return set_abstraction_body(nabs, body);
    }
    return set_abstraction_body(nabs, rewrite_node(r, obody));
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

static const Nodes* find_scope_info(const Node* abs) {
    assert(is_abstraction(abs));
    const Node* terminator = get_abstraction_body(abs);
    const Node* mem = get_terminator_mem(terminator);
    Nodes* info = NULL;
    while (mem) {
        if (mem->tag == ExtInstr_TAG && strcmp(mem->payload.ext_instr.set, "shady.scope") == 0) {
            info = &mem->payload.ext_instr.operands;
        }
        mem = get_parent_mem(mem);
    }
    return info;
}

bool compare_nodes(Nodes* a, Nodes* b);

static void process_edge(Context* ctx, CFG* cfg, CFEdge edge) {
    assert(edge.type == JumpEdge && edge.jump);
    const Node* src = edge.src->node;
    const Node* dst = edge.dst->node;

    IrArena* a = ctx->rewriter.dst_arena;
    // if (!ctx->config->hacks.recover_structure)
    //     break;
    const Nodes* src_lexical_scope = find_scope_info(src);
    const Nodes* dst_lexical_scope = find_scope_info(dst);
    if (!src_lexical_scope) {
        warn_print("Failed to find jump source node ");
        log_node(WARN, src);
        warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
    } else if (!dst_lexical_scope) {
        warn_print("Failed to find jump target node ");
        log_node(WARN, dst);
        warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
    } else if (lexical_scope_is_nested(*src_lexical_scope, *dst_lexical_scope)) {
        debug_print("Jump from %s to %s exits one or more nested lexical scopes, it might reconverge.\n", get_abstraction_name_safe(src), get_abstraction_name_safe(dst));

        CFNode* src_cfnode = cfg_lookup(cfg, src);
        assert(src_cfnode->node);
        CFNode* target_cfnode = cfg_lookup(cfg, dst);
        assert(src_cfnode && target_cfnode);
        CFNode* dom = src_cfnode->idom;
        while (dom) {
            debug_print("Considering %s as a location for control\n", get_abstraction_name_safe(dom->node));
            Nodes* dom_lexical_scope = find_scope_info(dom->node);
            if (!dom_lexical_scope) {
                warn_print("Basic block %s did not have an entry in the lexical_scopes map. Is debug information enabled ?\n", get_abstraction_name_safe(dom->node));
                dom = dom->idom;
                continue;
            } else if (lexical_scope_is_nested(*dst_lexical_scope, *dom_lexical_scope)) {
                error_print("We went up too far: %s is a parent of the jump destination scope.\n", get_abstraction_name_safe(dom->node));
            } else if (compare_nodes(dom_lexical_scope, dst_lexical_scope)) {
                debug_print("We need to introduce a control block at %s, pointing at %s\n.", get_abstraction_name_safe(dom->node), get_abstraction_name_safe(dst));

                Controls* controls = get_or_create_controls(ctx, dom->node);
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
                register_processed_list(&ctx->rewriter, get_abstraction_params(dst), nparams);

                Node* wrapper = basic_block(a, nparams, format_string_arena(a->arena, "wrapper_to_%s", get_abstraction_name_safe(dst)));
                wrapper->payload.basic_block.body = join(a, (Join) {
                    .args = nparams,
                    .join_point = join_token,
                    .mem = get_abstraction_mem(wrapper),
                });

                insert_dict(const Node*, const Node*, ctx->jump2wrapper, edge.jump, wrapper);
                // return jump_helper(a, wrapper, rewrite_nodes(&ctx->rewriter, node->payload.jump.args), rewrite_node(r, node->payload.jump.mem));
            } else {
                dom = dom->idom;
                continue;
            }
            break;
        }
    }
}

static void prepare_function(Context* ctx, const Node* old_fn) {
    CFG* cfg = build_fn_cfg(old_fn);
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = cfg->rpo[i];
        for (size_t j = 0; j < entries_count_list(n->succ_edges); j++) {
            process_edge(ctx, cfg, read_list(CFEdge, n->succ_edges)[j]);
        }
    }
    destroy_cfg(cfg);
}

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Param_TAG: {
            assert(false);
        }
        case Constant_TAG: {
            Node* new = (Node*) recreate_node_identity(r, node);
            BodyBuilder* bb = begin_block_pure(a);
            const Node* value = new->payload.constant.value;
            value = prim_op_helper(a, subgroup_assume_uniform_op, empty(a), singleton(value));
            new->payload.constant.value = yield_values_and_wrap_in_compound_instruction(bb, singleton(value));
            return new;
        }
        case Function_TAG: {
            prepare_function(ctx, node);

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
            register_processed_list(r, node->payload.fun.params, new_params);
            Nodes new_annotations = rewrite_nodes(r, old_annotations);
            Node* decl = function(ctx->rewriter.dst_module, new_params, get_abstraction_name(node), new_annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            register_processed(&ctx->rewriter, node, decl);
            if (primop_intrinsic != PRIMOPS_COUNT) {
                set_abstraction_body(decl, fn_ret(a, (Return) {
                    .args = singleton(prim_op_helper(a, primop_intrinsic, empty(a), get_abstraction_params(decl))),
                    .mem = get_abstraction_mem(decl),
                }));
            } else
                wrap_in_controls(ctx, decl, node);
            return decl;
        }
        case BasicBlock_TAG: {
            Nodes nparams = remake_params(ctx, get_abstraction_params(node));
            register_processed_list(r, get_abstraction_params(node), nparams);
            Node* new_bb = (Node*) basic_block(a, nparams, get_abstraction_name_unsafe(node));
            register_processed(r, node, new_bb);
            wrap_in_controls(ctx, new_bb, node);
            // new_bb->payload.basic_block.body = wrap_in_controls(ctx, &controls, new_bb->payload.basic_block.body);
            return new_bb;
        }
        // Eliminate now-useless scope instructions
        case ExtInstr_TAG: {
            if (strcmp(node->payload.ext_instr.set, "shady.scope") == 0) {
                return rewrite_node(r, node->payload.ext_instr.mem);
            }
            break;
        }
        case Jump_TAG: {
            const Node** found = find_value_dict(const Node*, const Node*, ctx->jump2wrapper, node);
            if (found)
                return jump_helper(a, *found, rewrite_nodes(&ctx->rewriter, node->payload.jump.args), rewrite_node(r, node->payload.jump.mem));
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

    return recreate_node_identity(&ctx->rewriter, node);
}

void postprocess(Parser* p, Module* src, Module* dst) {
    assert(src != dst);
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = p->config,
        .p = p,
        .arena = new_arena(),
        .controls = new_dict(const Node*, Controls*, (HashFn) hash_node, (CmpFn) compare_node),
        .jump2wrapper = new_dict(const Node*, Controls*, (HashFn) hash_node, (CmpFn) compare_node),
    };

    ctx.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

    rewrite_module(&ctx.rewriter);
    destroy_dict(ctx.controls);
    destroy_dict(ctx.jump2wrapper);
    destroy_arena(ctx.arena);
    destroy_rewriter(&ctx.rewriter);
}
