#include "shady/pass.h"

#include "portability.h"
#include "dict.h"
#include "list.h"
#include "log.h"
#include "arena.h"
#include "util.h"

#include "../rewrite.h"
#include "../type.h"
#include "../ir_private.h"
#include "../analysis/cfg.h"
#include "../analysis/scheduler.h"
#include "../analysis/free_frontier.h"
#include "../transform/ir_gen_helpers.h"

#include <string.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

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
        .control_destinations = new_dict(const Node*, AddControl, (HashFn) hash_node, (CmpFn) compare_node),
    };
    insert_dict(const Node*, Controls*, ctx->controls, fn_or_bb, controls);
    return controls;
}

static void wrap_in_controls(Context* ctx, CFG* cfg, Node* nabs, const Node* oabs) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    const Node* obody = get_abstraction_body(oabs);
    if (!obody)
        return;

    CFNode* n = cfg_lookup(cfg, oabs);
    size_t num_dom = entries_count_list(n->dominates);
    LARRAY(Node*, nbbs, num_dom);
    for (size_t i = 0; i < num_dom; i++) {
        CFNode* dominated = read_list(CFNode*, n->dominates)[i];
        const Node* obb = dominated->node;
        assert(obb->tag == BasicBlock_TAG);
        Nodes nparams = remake_params(ctx, get_abstraction_params(obb));
        register_processed_list(r, get_abstraction_params(obb), nparams);
        nbbs[i] = basic_block(a, nparams, get_abstraction_name_unsafe(obb));
        register_processed(r, obb, nbbs[i]);
    }

    // We introduce a dummy case now because we don't know yet whether the body of the abstraction will be wrapped
    Node* c = case_(a, empty(a));
    Node* oc = c;
    register_processed(r, get_abstraction_mem(oabs), get_abstraction_mem(c));

    for (size_t k = 0; k < num_dom; k++) {
        CFNode* dominated = read_list(CFNode*, n->dominates)[k];
        const Node* obb = dominated->node;
        wrap_in_controls(ctx, cfg, nbbs[k], obb);
    }

    Controls* controls = get_or_create_controls(ctx, oabs);

    set_abstraction_body(oc, rewrite_node(r, obody));

    size_t i = 0;
    AddControl add_control;
    while(dict_iter(controls->control_destinations, &i, NULL, &add_control)) {
        const Node* dst = add_control.destination;
        Node* control_case = case_(a, singleton(add_control.token));
        set_abstraction_body(control_case, jump_helper(a, c, empty(a), get_abstraction_mem(control_case)));

        Node* c2 = case_(a, empty(a));
        BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(c2));
        const Type* jp_type = add_control.token->type;
        deconstruct_qualified_type(&jp_type);
        assert(jp_type->tag == JoinPointType_TAG);
        Nodes results = gen_control(bb, jp_type->payload.join_point_type.yield_types, control_case);

        c = c2;
        set_abstraction_body(c2, finish_body(bb, jump_helper(a, find_processed(r, dst), results, bb_mem(bb))));
    }

    const Node* body = jump_helper(a, c, empty(a), get_abstraction_mem(nabs));
    set_abstraction_body(nabs, body);
}

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
            if (!info || info->count > mem->payload.ext_instr.operands.count)
                info = &mem->payload.ext_instr.operands;
        }
        mem = get_parent_mem(mem);
    }
    return info;
}

bool compare_nodes(Nodes* a, Nodes* b);

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
                AddControl* found = find_value_dict(const Node, AddControl, controls->control_destinations, dst);
                Wrapped wrapped;
                if (found) {
                    wrapped.wrapper = found->wrapper;
                } else {
                    Nodes wrapper_params = remake_params(ctx, get_abstraction_params(dst));
                    Nodes join_args = wrapper_params;
                    Nodes yield_types = rewrite_nodes(r, strip_qualifiers(a, get_param_types(a, get_abstraction_params(dst))));

                    const Type* jp_type = join_point_type(a, (JoinPointType) {
                        .yield_types = yield_types
                    });
                    const Node* join_token = param(a, qualified_type_helper(jp_type, false), get_abstraction_name_unsafe(dst));

                    Node* wrapper = basic_block(a, wrapper_params, format_string_arena(a->arena, "wrapper_to_%s", get_abstraction_name_safe(dst)));
                    wrapper->payload.basic_block.body = join(a, (Join) {
                        .args = join_args,
                        .join_point = join_token,
                        .mem = get_abstraction_mem(wrapper),
                    });

                    AddControl add_control = {
                        .destination = dst,
                        .token = join_token,
                        .wrapper = wrapper,
                    };
                    wrapped.wrapper = wrapper;
                    insert_dict(const Node*, AddControl, controls->control_destinations, dst, add_control);
                }

                insert_dict(const Node*, Wrapped, ctx->jump2wrapper, edge.jump, wrapped);
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
    Scheduler* scheduler = new_scheduler(cfg);
    for (size_t i = 0; i < cfg->size; i++) {
        CFNode* n = cfg->rpo[i];
        for (size_t j = 0; j < entries_count_list(n->succ_edges); j++) {
            process_edge(ctx, cfg, scheduler, read_list(CFEdge, n->succ_edges)[j]);
        }
    }
    destroy_scheduler(scheduler);
}

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Function_TAG: {
            CFG* cfg = build_fn_cfg(node);
            prepare_function(ctx, cfg, node);
            Node* decl = recreate_decl_header_identity(r, node);
            wrap_in_controls(ctx, cfg, decl, node);
            destroy_cfg(cfg);
            return decl;
        }
        case BasicBlock_TAG: {
            assert(false);
        }
        // Eliminate now-useless scope instructions
        case ExtInstr_TAG: {
            if (strcmp(node->payload.ext_instr.set, "shady.scope") == 0) {
                return rewrite_node(r, node->payload.ext_instr.mem);
            }
            break;
        }
        case Jump_TAG: {
            Wrapped* found = find_value_dict(const Node*, Wrapped, ctx->jump2wrapper, node);
            if (found)
                return jump_helper(a, found->wrapper, rewrite_nodes(r, node->payload.jump.args), rewrite_node(r, node->payload.jump.mem));
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* scope2control(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.optimisations.inline_single_use_bbs = true;
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
        .arena = new_arena(),
        .controls = new_dict(const Node*, Controls*, (HashFn) hash_node, (CmpFn) compare_node),
        .jump2wrapper = new_dict(const Node*, Wrapped, (HashFn) hash_node, (CmpFn) compare_node),
    };

    ctx.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

    size_t i = 0;
    Controls controls;
    while (dict_iter(ctx.controls, &i, NULL, &controls)) {
        size_t j = 0;
        AddControl add_control;
        // while (dict_iter(controls.control_destinations, &j, NULL, &add_control)) {
        //     destroy_list(add_control.lift);
        // }
        destroy_dict(controls.control_destinations);
    }

    rewrite_module(&ctx.rewriter);
    destroy_dict(ctx.controls);
    destroy_dict(ctx.jump2wrapper);
    destroy_arena(ctx.arena);
    destroy_rewriter(&ctx.rewriter);

    return dst;
}
