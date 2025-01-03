#include "join_point_ops.h"

#include "shady/pass.h"
#include "shady/visit.h"
#include "shady/ir/stack.h"
#include "shady/ir/ext.h"

#include "ir_private.h"

#include "analysis/cfg.h"
#include "analysis/uses.h"
#include "analysis/leak.h"
#include "analysis/verify.h"
#include "analysis/scheduler.h"
#include "analysis/free_frontier.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "util.h"

#include <assert.h>

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

typedef struct {
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
    Nodes save_values;
} LiftedCont;

#pragma GCC diagnostic error "-Wswitch"

static const Node* add_spill_instrs(Context* ctx, BodyBuilder* builder, Nodes spilled_vars) {
    for (size_t i = 0; i < spilled_vars.count; i++) {
        const Node* ovar = spilled_vars.nodes[i];
        const Node* nvar = shd_rewrite_node(&ctx->rewriter, ovar);
        const Type* t = nvar->type;
        shd_deconstruct_qualified_type(&t);
        assert(t->tag != PtrType_TAG || !t->payload.ptr_type.is_reference && "References cannot be spilled");
        shd_bld_stack_push_value(builder, nvar);
    }

    return shd_bld_get_stack_size(builder);
}

static Nodes set2nodes(IrArena* a, struct Dict* set) {
    size_t count = shd_dict_count(set);
    LARRAY(const Node*, tmp, count);
    size_t i = 0, j = 0;
    const Node* key;
    while (shd_dict_iter(set, &i, &key, NULL)) {
        tmp[j++] = key;
    }
    assert(j == count);
    return shd_nodes(a, count, tmp);
}

static LiftedCont* lambda_lift(Context* ctx, CFG* cfg, const Node* liftee) {
    assert(is_basic_block(liftee));
    LiftedCont** found = shd_dict_find_value(const Node*, LiftedCont*, ctx->lifted, liftee);
    if (found)
        return *found;

    IrArena* a = ctx->rewriter.dst_arena;
    const Node* obody = get_abstraction_body(liftee);
    String name = shd_get_abstraction_name_safe(liftee);

    Scheduler* scheduler = shd_new_scheduler(cfg);
    struct Dict* frontier_set = shd_free_frontier(scheduler, cfg, liftee);
    Nodes frontier = set2nodes(a, frontier_set);
    shd_destroy_dict(frontier_set);

    size_t recover_context_size = frontier.count;

    shd_destroy_scheduler(scheduler);

    Context lifting_ctx = *ctx;
    lifting_ctx.rewriter = shd_create_children_rewriter(shd_get_top_rewriter(&ctx->rewriter));
    Rewriter* r = &lifting_ctx.rewriter;

    Nodes ovariables = get_abstraction_params(liftee);
    shd_debugv_print("lambda_lift: free (to-be-spilled) variables at '%s' (count=%d): ", shd_get_abstraction_name_safe(liftee), recover_context_size);
    for (size_t i = 0; i < recover_context_size; i++) {
        const Node* item = frontier.nodes[i];
        if (!is_value(item)) {
            //lambda_lift()
            continue;
        }
        shd_debugv_print("%%%d", item->id);
        if (i + 1 < recover_context_size)
            shd_debugv_print(", ");
    }
    shd_debugv_print("\n");

    // Create and register new parameters for the lifted continuation
    LARRAY(const Node*, new_params_arr, ovariables.count);
    for (size_t i = 0; i < ovariables.count; i++)
        new_params_arr[i] = param_helper(a, shd_rewrite_node(&ctx->rewriter, ovariables.nodes[i]->type), shd_get_value_name_unsafe(ovariables.nodes[i]));
    Nodes new_params = shd_nodes(a, ovariables.count, new_params_arr);

    LiftedCont* lifted_cont = calloc(sizeof(LiftedCont), 1);
    lifted_cont->old_cont = liftee;
    lifted_cont->save_values = frontier;
    shd_dict_insert(const Node*, LiftedCont*, ctx->lifted, liftee, lifted_cont);

    shd_register_processed_list(r, ovariables, new_params);

    const Node* payload = param_helper(a, shd_as_qualified_type(shd_uint32_type(a), false), "sp");

    // Keep annotations the same
    Nodes annotations = shd_singleton(annotation(a, (Annotation) { .name = "Exported" }));
    new_params = shd_nodes_prepend(a, new_params, payload);
    Node* new_fn = function(ctx->rewriter.dst_module, new_params, name, annotations, shd_nodes(a, 0, NULL));
    lifted_cont->lifted_fn = new_fn;

    // Recover that stuff inside the new body
    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new_fn));
    shd_bld_set_stack_size(bb, payload);
    for (size_t i = recover_context_size - 1; i < recover_context_size; i--) {
        const Node* ovar = frontier.nodes[i];
        // assert(ovar->tag == Variable_TAG);

        const Type* value_type = shd_rewrite_node(r, ovar->type);

        //String param_name = get_value_name_unsafe(ovar);
        const Node* recovered_value = shd_bld_stack_pop_value(bb, shd_get_unqualified_type(value_type));
        //if (param_name)
        //    set_value_name(recovered_value, param_name);

        if (shd_is_qualified_type_uniform(ovar->type))
            recovered_value = prim_op(a, (PrimOp) { .op = subgroup_assume_uniform_op, .operands = shd_singleton(recovered_value) });

        shd_register_processed(r, ovar, recovered_value);
    }

    shd_register_processed(r, shd_get_abstraction_mem(liftee), shd_bld_mem(bb));
    shd_register_processed(r, liftee, new_fn);
    const Node* substituted = shd_rewrite_node(r, obody);
    shd_destroy_rewriter(r);

    assert(is_terminator(substituted));
    shd_set_abstraction_body(new_fn, shd_bld_finish(bb, substituted));

    return lifted_cont;
}

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    switch (is_declaration(node)) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.cfg = build_fn_cfg(node);
            fn_ctx.uses = shd_new_uses_map_fn(node, (NcDeclaration | NcType));
            fn_ctx.disable_lowering = shd_lookup_annotation(node, "Internal");
            ctx = &fn_ctx;

            Node* new = shd_recreate_node_head(&ctx->rewriter, node);
            shd_recreate_node_body(&ctx->rewriter, node, new);

            shd_destroy_uses_map(ctx->uses);
            shd_destroy_cfg(ctx->cfg);
            return new;
        }
        default:
            break;
    }

    if (ctx->disable_lowering)
         return shd_recreate_node(&ctx->rewriter, node);

    switch (node->tag) {
        case Control_TAG: {
            const Node* oinside = node->payload.control.inside;
            if (!shd_is_control_static(ctx->uses, node) || ctx->config->hacks.force_join_point_lifting) {
                *ctx->todo = true;

                const Node* otail = get_structured_construct_tail(node);
                BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, node->payload.control.mem));
                LiftedCont* lifted_tail = lambda_lift(ctx, ctx->cfg, otail);
                const Node* sp = add_spill_instrs(ctx, bb, lifted_tail->save_values);
                const Node* tail_ptr = fn_addr_helper(a, lifted_tail->lifted_fn);

                const Type* jp_type = join_point_type(a, (JoinPointType) {
                    .yield_types = shd_rewrite_nodes(&ctx->rewriter, node->payload.control.yield_types),
                });
                const Node* jp = shd_bld_ext_instruction(bb, "shady.internal", ShadyOpCreateJoinPoint,
                                                         shd_as_qualified_type(jp_type, true), mk_nodes(a, tail_ptr, sp));
                // dumbass hack
                jp = prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(jp));

                shd_register_processed(r, shd_first(get_abstraction_params(oinside)), jp);
                shd_register_processed(r, shd_get_abstraction_mem(oinside), shd_bld_mem(bb));
                shd_register_processed(r, oinside, NULL);
                return shd_bld_finish(bb, shd_rewrite_node(&ctx->rewriter, get_abstraction_body(oinside)));
            }
            break;
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_lift_indirect_targets(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = NULL;
    Module* dst;

    int round = 0;
    while (true) {
        shd_debugv_print("lift_indirect_target: round %d\n", round++);
        IrArena* oa = a;
        a = shd_new_ir_arena(&aconfig);
        dst = shd_new_module(a, shd_module_get_name(src));
        bool todo = false;
        Context ctx = {
            .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
            .lifted = shd_new_dict(const Node*, LiftedCont*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
            .config = config,

            .todo = &todo
        };

        shd_rewrite_module(&ctx.rewriter);

        size_t iter = 0;
        LiftedCont* lifted_cont;
        while (shd_dict_iter(ctx.lifted, &iter, NULL, &lifted_cont)) {
            free(lifted_cont);
        }
        shd_destroy_dict(ctx.lifted);
        shd_destroy_rewriter(&ctx.rewriter);
        shd_verify_module(config, dst);
        src = dst;
        if (oa)
            shd_destroy_ir_arena(oa);
        if (!todo) {
            break;
        }
    }

    // this will be safe now since we won't lift any more code after this pass
    aconfig.optimisations.weaken_non_leaking_allocas = true;
    IrArena* a2 = shd_new_ir_arena(&aconfig);
    dst = shd_new_module(a2, shd_module_get_name(src));
    Rewriter r = shd_create_importer(src, dst);
    shd_rewrite_module(&r);
    shd_destroy_rewriter(&r);
    shd_destroy_ir_arena(a);
    return dst;
}