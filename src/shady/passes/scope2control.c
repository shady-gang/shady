#include "shady/pass.h"

#include "shady/rewrite.h"

#include "../ir_private.h"
#include "../analysis/cfg.h"
#include "../analysis/scheduler.h"

#include "portability.h"
#include "dict.h"
#include "list.h"
#include "log.h"
#include "arena.h"
#include "util.h"

#include <string.h>

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    Arena* arena;
    struct Dict* controls;
    struct Dict* jump2wrapper;
} Context;

typedef struct {
    const Node* wrapper;
} Wrapped;

typedef struct {
    Node* wrapper;
    const Node* token;
    const Node* destination;
} AddControl;

typedef struct {
    struct Dict* control_destinations;
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
                t = shd_rewrite_node(r, node->payload.param.type);
            else
                t = shd_as_qualified_type(shd_rewrite_node(r, node->payload.param.type), false);
        }
        nvars[i] = param(a, t, node->payload.param.name);
        assert(nvars[i]->tag == Param_TAG);
    }
    return shd_nodes(a, old.count, nvars);
}

static Controls* get_or_create_controls(Context* ctx, const Node* fn_or_bb) {
    Controls** found = shd_dict_find_value(const Node, Controls*, ctx->controls, fn_or_bb);
    if (found)
        return *found;
    IrArena* a = ctx->rewriter.dst_arena;
    Controls* controls = shd_arena_alloc(ctx->arena, sizeof(Controls));
    *controls = (Controls) {
        .control_destinations = shd_new_dict(const Node*, AddControl, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };
    shd_dict_insert(const Node*, Controls*, ctx->controls, fn_or_bb, controls);
    return controls;
}

static void wrap_in_controls(Context* ctx, CFG* cfg, Node* nabs, const Node* oabs) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    const Node* obody = get_abstraction_body(oabs);
    if (!obody)
        return;

    CFNode* n = shd_cfg_lookup(cfg, oabs);
    size_t num_dom = shd_list_count(n->dominates);
    LARRAY(Node*, nbbs, num_dom);
    for (size_t i = 0; i < num_dom; i++) {
        CFNode* dominated = shd_read_list(CFNode*, n->dominates)[i];
        const Node* obb = dominated->node;
        assert(obb->tag == BasicBlock_TAG);
        Nodes nparams = remake_params(ctx, get_abstraction_params(obb));
        shd_register_processed_list(r, get_abstraction_params(obb), nparams);
        nbbs[i] = basic_block(a, nparams, shd_get_abstraction_name_unsafe(obb));
        shd_register_processed(r, obb, nbbs[i]);
    }

    // We introduce a dummy case now because we don't know yet whether the body of the abstraction will be wrapped
    Node* c = case_(a, shd_empty(a));
    Node* oc = c;
    shd_register_processed(r, shd_get_abstraction_mem(oabs), shd_get_abstraction_mem(c));

    for (size_t k = 0; k < num_dom; k++) {
        CFNode* dominated = shd_read_list(CFNode*, n->dominates)[k];
        const Node* obb = dominated->node;
        wrap_in_controls(ctx, cfg, nbbs[k], obb);
    }

    Controls* controls = get_or_create_controls(ctx, oabs);

    shd_set_abstraction_body(oc, shd_rewrite_node(r, obody));

    size_t i = 0;
    AddControl add_control;
    while(shd_dict_iter(controls->control_destinations, &i, NULL, &add_control)) {
        const Node* dst = add_control.destination;
        Node* control_case = case_(a, shd_singleton(add_control.token));
        shd_set_abstraction_body(control_case, jump_helper(a, shd_get_abstraction_mem(control_case), c, shd_empty(a)));

        Node* c2 = case_(a, shd_empty(a));
        BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(c2));
        const Type* jp_type = add_control.token->type;
        shd_deconstruct_qualified_type(&jp_type);
        assert(jp_type->tag == JoinPointType_TAG);
        Nodes results = shd_bld_control(bb, jp_type->payload.join_point_type.yield_types, control_case);

        Nodes original_params = get_abstraction_params(dst);
        for (size_t j = 0; j < results.count; j++) {
            if (shd_is_qualified_type_uniform(original_params.nodes[j]->type))
                results = shd_change_node_at_index(a, results, j, prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(results.nodes[j])));
        }

        c = c2;
        shd_set_abstraction_body(c2, shd_bld_finish(bb, jump_helper(a, shd_bb_mem(bb), shd_find_processed(r, dst), results)));
    }

    const Node* body = jump_helper(a, shd_get_abstraction_mem(nabs), c, shd_empty(a));
    shd_set_abstraction_body(nabs, body);
}

static bool lexical_scope_is_nested(Nodes scope, Nodes parentMaybe) {
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
            if (!info || info->count > mem->payload.ext_instr.operands.count)
                info = &mem->payload.ext_instr.operands;
        }
        mem = shd_get_parent_mem(mem);
    }
    return info;
}

bool shd_compare_nodes(Nodes* a, Nodes* b);

static void process_edge(Context* ctx, CFG* cfg, Scheduler* scheduler, CFEdge edge) {
    assert(edge.type == JumpEdge && edge.jump);
    const Node* src = edge.src->node;
    const Node* dst = edge.dst->node;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = ctx->rewriter.dst_arena;
    // if (!ctx->config->hacks.recover_structure)
    //     break;
    const Nodes* src_lexical_scope = find_scope_info(src);
    const Nodes* dst_lexical_scope = find_scope_info(dst);
    if (!src_lexical_scope) {
        shd_warn_print("Failed to find jump source node ");
        shd_log_node(WARN, src);
        shd_warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
    } else if (!dst_lexical_scope) {
        shd_warn_print("Failed to find jump target node ");
        shd_log_node(WARN, dst);
        shd_warn_print(" in lexical_scopes map. Is debug information enabled ?\n");
    } else if (lexical_scope_is_nested(*src_lexical_scope, *dst_lexical_scope)) {
        shd_debug_print("Jump from %s to %s exits one or more nested lexical scopes, it might reconverge.\n", shd_get_abstraction_name_safe(src), shd_get_abstraction_name_safe(dst));

        CFNode* src_cfnode = shd_cfg_lookup(cfg, src);
        assert(src_cfnode->node);
        CFNode* dst_cfnode = shd_cfg_lookup(cfg, dst);
        assert(src_cfnode && dst_cfnode);

        // if(!cfg_is_dominated(dst_cfnode, src_cfnode))
        //     return;

        CFNode* dom = src_cfnode->idom;
        while (dom) {
            shd_debug_print("Considering %s as a location for control\n", shd_get_abstraction_name_safe(dom->node));
            Nodes* dom_lexical_scope = find_scope_info(dom->node);
            if (!dom_lexical_scope) {
                shd_warn_print("Basic block %s did not have an entry in the lexical_scopes map. Is debug information enabled ?\n", shd_get_abstraction_name_safe(dom->node));
                dom = dom->idom;
                continue;
            } else if (lexical_scope_is_nested(*dst_lexical_scope, *dom_lexical_scope)) {
                shd_error_print("We went up too far: %s is a parent of the jump destination scope.\n", shd_get_abstraction_name_safe(dom->node));
            } else if (shd_compare_nodes(dom_lexical_scope, dst_lexical_scope)) {
                // if (cfg_is_dominated(target_cfnode, dom)) {
                if (!shd_cfg_is_dominated(dom, dst_cfnode) && dst_cfnode != dom) {
                    // assert(false);
                }

                shd_debug_print("We need to introduce a control block at %s, pointing at %s\n.", shd_get_abstraction_name_safe(dom->node), shd_get_abstraction_name_safe(dst));

                Controls* controls = get_or_create_controls(ctx, dom->node);
                AddControl* found = shd_dict_find_value(const Node, AddControl, controls->control_destinations, dst);
                Wrapped wrapped;
                if (found) {
                    wrapped.wrapper = found->wrapper;
                } else {
                    Nodes wrapper_params = remake_params(ctx, get_abstraction_params(dst));
                    Nodes join_args = wrapper_params;
                    Nodes yield_types = shd_rewrite_nodes(r, shd_strip_qualifiers(a, shd_get_param_types(a, get_abstraction_params(dst))));

                    const Type* jp_type = join_point_type(a, (JoinPointType) {
                        .yield_types = yield_types
                    });
                    const Node* join_token = param(a, shd_as_qualified_type(jp_type, false), shd_get_abstraction_name_unsafe(dst));

                    Node* wrapper = basic_block(a, wrapper_params, shd_format_string_arena(a->arena, "wrapper_to_%s", shd_get_abstraction_name_safe(dst)));
                    wrapper->payload.basic_block.body = join(a, (Join) {
                        .args = join_args,
                        .join_point = join_token,
                        .mem = shd_get_abstraction_mem(wrapper),
                    });

                    AddControl add_control = {
                        .destination = dst,
                        .token = join_token,
                        .wrapper = wrapper,
                    };
                    wrapped.wrapper = wrapper;
                    shd_dict_insert(const Node*, AddControl, controls->control_destinations, dst, add_control);
                }

                shd_dict_insert(const Node*, Wrapped, ctx->jump2wrapper, edge.jump, wrapped);
                // return jump_helper(a, wrapper, rewrite_nodes(&ctx->rewriter, node->payload.jump.args), rewrite_node(r, node->payload.jump.mem));
            } else {
                dom = dom->idom;
                continue;
            }
            break;
        }
    }
}

static void prepare_function(Context* ctx, CFG* cfg, const Node* old_fn) {
    Scheduler* scheduler = shd_new_scheduler(cfg);
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = cfg->rpo[i];
        for (size_t j = 0; j < shd_list_count(n->succ_edges); j++) {
            process_edge(ctx, cfg, scheduler, shd_read_list(CFEdge, n->succ_edges)[j]);
        }
    }
    shd_destroy_scheduler(scheduler);
}

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Function_TAG: {
            CFG* cfg = build_fn_cfg(node);
            prepare_function(ctx, cfg, node);
            Node* decl = shd_recreate_node_head(r, node);
            wrap_in_controls(ctx, cfg, decl, node);
            shd_destroy_cfg(cfg);
            return decl;
        }
        case BasicBlock_TAG: {
            assert(false);
        }
        // Eliminate now-useless scope instructions
        case ExtInstr_TAG: {
            if (strcmp(node->payload.ext_instr.set, "shady.scope") == 0) {
                return shd_rewrite_node(r, node->payload.ext_instr.mem);
            }
            break;
        }
        case Jump_TAG: {
            Wrapped* found = shd_dict_find_value(const Node*, Wrapped, ctx->jump2wrapper, node);
            if (found)
                return jump_helper(a, shd_rewrite_node(r, node->payload.jump.mem), found->wrapper,
                                   shd_rewrite_nodes(r, node->payload.jump.args));
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_scope2control(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.optimisations.inline_single_use_bbs = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
        .arena = shd_new_arena(),
        .controls = shd_new_dict(const Node*, Controls*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .jump2wrapper = shd_new_dict(const Node*, Wrapped, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };

    ctx.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

    shd_rewrite_module(&ctx.rewriter);

    size_t i = 0;
    Controls* controls;
    while (shd_dict_iter(ctx.controls, &i, NULL, &controls)) {
        //size_t j = 0;
        //AddControl add_control;
        // while (dict_iter(controls.control_destinations, &j, NULL, &add_control)) {
        //     destroy_list(add_control.lift);
        // }
        shd_destroy_dict(controls->control_destinations);
    }

    shd_destroy_dict(ctx.controls);
    shd_destroy_dict(ctx.jump2wrapper);
    shd_destroy_arena(ctx.arena);
    shd_destroy_rewriter(&ctx.rewriter);

    return dst;
}
