#include "pass.h"

#include "../type.h"
#include "../ir_private.h"

#include "../transform/ir_gen_helpers.h"
#include "../analysis/cfg.h"
#include "../analysis/free_variables.h"
#include "../analysis/uses.h"
#include "../analysis/leak.h"
#include "../analysis/verify.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "util.h"

#include <assert.h>
#include <string.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct Context_ {
    Rewriter rewriter;
    CFG* cfg;
    const UsesMap* uses;

    struct Dict* lifted;
    bool disable_lowering;
    const CompilerConfig* config;

    bool* todo;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

typedef struct {
    const Node* old_cont;
    const Node* lifted_fn;
    struct List* save_values;
} LiftedCont;

#pragma GCC diagnostic error "-Wswitch"

static const Node* add_spill_instrs(Context* ctx, BodyBuilder* builder, struct List* spilled_vars) {
    IrArena* a = ctx->rewriter.dst_arena;

    size_t recover_context_size = entries_count_list(spilled_vars);
    for (size_t i = 0; i < recover_context_size; i++) {
        const Node* ovar = read_list(const Node*, spilled_vars)[i];
        const Node* nvar = rewrite_node(&ctx->rewriter, ovar);
        const Type* t = nvar->type;
        deconstruct_qualified_type(&t);
        assert(t->tag != PtrType_TAG || !t->payload.ptr_type.is_reference && "References cannot be spilled");
        gen_push_value_stack(builder, nvar);
    }

    const Node* sp = gen_get_stack_size(builder);

    return sp;
}

static void add_to_recover_context(struct List* recover_context, struct Dict* set, Nodes except) {
    size_t i = 0;
    const Node* item;
    while (dict_iter(set, &i, &item, NULL)) {
        if (find_in_nodes(except, item))
            continue;
        append_list(const Node*, recover_context, item );
    }
}

static LiftedCont* lambda_lift(Context* ctx, const Node* liftee, Nodes ovariables) {
    assert(is_basic_block(liftee));
    LiftedCont** found = find_value_dict(const Node*, LiftedCont*, ctx->lifted, liftee);
    if (found)
        return *found;

    IrArena* a = ctx->rewriter.dst_arena;
    //Nodes oparams = get_abstraction_params(liftee);
    const Node* obody = get_abstraction_body(liftee);

    String name = get_abstraction_name_safe(liftee);

    // Compute the live stuff we'll need
    CFG* cfg_rooted_in_liftee = build_cfg(ctx->cfg->entry->node, liftee, NULL, false);
    CFNode* cf_node = cfg_lookup(cfg_rooted_in_liftee, liftee);
    struct Dict* live_vars = compute_cfg_variables_map(cfg_rooted_in_liftee, CfgVariablesAnalysisFlagFreeSet);
    CFNodeVariables* node_vars = *find_value_dict(CFNode*, CFNodeVariables*, live_vars, cf_node);
    struct List* recover_context = new_list(const Node*);

    add_to_recover_context(recover_context, node_vars->free_set, ovariables);
    size_t recover_context_size = entries_count_list(recover_context);

    destroy_cfg_variables_map(live_vars);
    destroy_cfg(cfg_rooted_in_liftee);

    debugv_print("lambda_lift: free (to-be-spilled) variables at '%s' (count=%d): ", get_abstraction_name_safe(liftee), recover_context_size);
    for (size_t i = 0; i < recover_context_size; i++) {
        const Node* item = read_list(const Node*, recover_context)[i];
        String item_name = get_value_name_unsafe(item);
        debugv_print("%s %%%d", item_name ? item_name : "", item->id);
        if (i + 1 < recover_context_size)
            debugv_print(", ");
    }
    debugv_print("\n");

    // Create and register new parameters for the lifted continuation
    LARRAY(const Node*, new_params_arr, ovariables.count);
    for (size_t i = 0; i < ovariables.count; i++)
        new_params_arr[i] = param(a, rewrite_node(&ctx->rewriter, ovariables.nodes[i]->type), get_value_name_unsafe(ovariables.nodes[i]));
    Nodes new_params = nodes(a, ovariables.count, new_params_arr);

    LiftedCont* lifted_cont = calloc(sizeof(LiftedCont), 1);
    lifted_cont->old_cont = liftee;
    lifted_cont->save_values = recover_context;
    insert_dict(const Node*, LiftedCont*, ctx->lifted, liftee, lifted_cont);

    Context lifting_ctx = *ctx;
    lifting_ctx.rewriter = create_children_rewriter(&ctx->rewriter);
    Rewriter* r = &lifting_ctx.rewriter;
    register_processed_list(r, ovariables, new_params);

    const Node* payload = param(a, qualified_type_helper(uint32_type(a), false), "sp");

    // Keep annotations the same
    Nodes annotations = nodes(a, 0, NULL);
    new_params = prepend_nodes(a, new_params, payload);
    Node* new_fn = function(ctx->rewriter.dst_module, new_params, name, annotations, nodes(a, 0, NULL));
    lifted_cont->lifted_fn = new_fn;

    // Recover that stuff inside the new body
    BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(new_fn));
    gen_set_stack_size(bb, payload);
    for (size_t i = recover_context_size - 1; i < recover_context_size; i--) {
        const Node* ovar = read_list(const Node*, recover_context)[i];
        // assert(ovar->tag == Variable_TAG);

        const Type* value_type = rewrite_node(r, ovar->type);

        String param_name = get_value_name_unsafe(ovar);
        const Node* recovered_value = gen_pop_value_stack(bb, get_unqualified_type(value_type));
        if (param_name)
            set_value_name(recovered_value, param_name);

        if (is_qualified_type_uniform(ovar->type))
            recovered_value = first(bind_instruction_named(bb, prim_op(a, (PrimOp) { .op = subgroup_assume_uniform_op, .operands = singleton(recovered_value) }), &param_name));

        register_processed(r, ovar, recovered_value);
    }

    register_processed(r, get_abstraction_mem(liftee), bb_mem(bb));
    const Node* substituted = rewrite_node(r, obody);
    destroy_rewriter(r);

    assert(is_terminator(substituted));
    set_abstraction_body(new_fn, finish_body(bb, substituted));

    return lifted_cont;
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    switch (is_declaration(node)) {
        case Function_TAG: {
            while (ctx->rewriter.parent)
                ctx = (Context*) ctx->rewriter.parent;

            Context fn_ctx = *ctx;
            fn_ctx.cfg = build_fn_cfg(node);
            fn_ctx.uses = create_uses_map(node, (NcDeclaration | NcType));
            fn_ctx.disable_lowering = lookup_annotation(node, "Internal");
            ctx = &fn_ctx;

            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            recreate_decl_body_identity(&ctx->rewriter, node, new);

            destroy_uses_map(ctx->uses);
            destroy_cfg(ctx->cfg);
            return new;
        }
        default:
            break;
    }

    if (ctx->disable_lowering)
         return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        case Control_TAG: {
            const Node* oinside = node->payload.control.inside;
            if (!is_control_static(ctx->uses, node) || ctx->config->hacks.force_join_point_lifting) {
                *ctx->todo = true;

                const Node* otail = get_structured_construct_tail(node);
                BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, node->payload.control.mem));
                LiftedCont* lifted_tail = lambda_lift(ctx, otail, get_abstraction_params(otail));
                const Node* sp = add_spill_instrs(ctx, bb, lifted_tail->save_values);
                const Node* tail_ptr = fn_addr_helper(a, lifted_tail->lifted_fn);

                const Node* jp = gen_primop_e(bb, create_joint_point_op, rewrite_nodes(&ctx->rewriter, node->payload.control.yield_types), mk_nodes(a, tail_ptr, sp));
                // dumbass hack
                jp = gen_primop_e(bb, subgroup_assume_uniform_op, empty(a), singleton(jp));

                register_processed(r, first(get_abstraction_params(oinside)), jp);
                return finish_body(bb, rewrite_node(&ctx->rewriter, get_abstraction_body(oinside)));
            }
            break;
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lift_indirect_targets(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = NULL;
    Module* dst;

    int round = 0;
    while (true) {
        debugv_print("lift_indirect_target: round %d\n", round++);
        IrArena* oa = a;
        a = new_ir_arena(&aconfig);
        dst = new_module(a, get_module_name(src));
        bool todo = false;
        Context ctx = {
            .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
            .lifted = new_dict(const Node*, LiftedCont*, (HashFn) hash_node, (CmpFn) compare_node),
            .config = config,

            .todo = &todo
        };

        rewrite_module(&ctx.rewriter);

        size_t iter = 0;
        LiftedCont* lifted_cont;
        while (dict_iter(ctx.lifted, &iter, NULL, &lifted_cont)) {
            destroy_list(lifted_cont->save_values);
            free(lifted_cont);
        }
        destroy_dict(ctx.lifted);
        destroy_rewriter(&ctx.rewriter);
        // log_module(DEBUGVV, config, dst);
        verify_module(config, dst);
        src = dst;
        if (oa)
            destroy_ir_arena(oa);
        if (!todo) {
            break;
        }
    }

    // this will be safe now since we won't lift any more code after this pass
    aconfig.optimisations.weaken_non_leaking_allocas = true;
    IrArena* a2 = new_ir_arena(&aconfig);
    dst = new_module(a2, get_module_name(src));
    Rewriter r = create_importer(src, dst);
    rewrite_module(&r);
    destroy_rewriter(&r);
    destroy_ir_arena(a);
    return dst;
}
