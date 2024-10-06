#include "shady/pass.h"
#include "join_point_ops.h"

#include "../type.h"
#include "../ir_private.h"
#include "../visit.h"

#include "../transform/ir_gen_helpers.h"
#include "../analysis/cfg.h"
#include "../analysis/uses.h"
#include "../analysis/leak.h"
#include "../analysis/verify.h"
#include "../analysis/scheduler.h"
#include "../analysis/free_frontier.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "util.h"

#include <assert.h>

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
    Nodes save_values;
} LiftedCont;

#pragma GCC diagnostic error "-Wswitch"

static const Node* add_spill_instrs(Context* ctx, BodyBuilder* builder, Nodes spilled_vars) {
    for (size_t i = 0; i < spilled_vars.count; i++) {
        const Node* ovar = spilled_vars.nodes[i];
        const Node* nvar = rewrite_node(&ctx->rewriter, ovar);
        const Type* t = nvar->type;
        deconstruct_qualified_type(&t);
        assert(t->tag != PtrType_TAG || !t->payload.ptr_type.is_reference && "References cannot be spilled");
        gen_push_value_stack(builder, nvar);
    }

    return gen_get_stack_size(builder);
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
    String name = get_abstraction_name_safe(liftee);

    Scheduler* scheduler = new_scheduler(cfg);
    struct Dict* frontier_set = free_frontier(scheduler, cfg, liftee);
    Nodes frontier = set2nodes(a, frontier_set);
    shd_destroy_dict(frontier_set);

    size_t recover_context_size = frontier.count;

    destroy_scheduler(scheduler);

    Context lifting_ctx = *ctx;
    lifting_ctx.rewriter = create_decl_rewriter(&ctx->rewriter);
    Rewriter* r = &lifting_ctx.rewriter;

    Nodes ovariables = get_abstraction_params(liftee);
    shd_debugv_print("lambda_lift: free (to-be-spilled) variables at '%s' (count=%d): ", get_abstraction_name_safe(liftee), recover_context_size);
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
        new_params_arr[i] = param(a, rewrite_node(&ctx->rewriter, ovariables.nodes[i]->type), get_value_name_unsafe(ovariables.nodes[i]));
    Nodes new_params = shd_nodes(a, ovariables.count, new_params_arr);

    LiftedCont* lifted_cont = calloc(sizeof(LiftedCont), 1);
    lifted_cont->old_cont = liftee;
    lifted_cont->save_values = frontier;
    shd_dict_insert(const Node*, LiftedCont*, ctx->lifted, liftee, lifted_cont);

    register_processed_list(r, ovariables, new_params);

    const Node* payload = param(a, shd_as_qualified_type(shd_uint32_type(a), false), "sp");

    // Keep annotations the same
    Nodes annotations = shd_singleton(annotation(a, (Annotation) { .name = "Exported" }));
    new_params = shd_nodes_prepend(a, new_params, payload);
    Node* new_fn = function(ctx->rewriter.dst_module, new_params, name, annotations, shd_nodes(a, 0, NULL));
    lifted_cont->lifted_fn = new_fn;

    // Recover that stuff inside the new body
    BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(new_fn));
    gen_set_stack_size(bb, payload);
    for (size_t i = recover_context_size - 1; i < recover_context_size; i--) {
        const Node* ovar = frontier.nodes[i];
        // assert(ovar->tag == Variable_TAG);

        const Type* value_type = rewrite_node(r, ovar->type);

        //String param_name = get_value_name_unsafe(ovar);
        const Node* recovered_value = gen_pop_value_stack(bb, get_unqualified_type(value_type));
        //if (param_name)
        //    set_value_name(recovered_value, param_name);

        if (is_qualified_type_uniform(ovar->type))
            recovered_value = prim_op(a, (PrimOp) { .op = subgroup_assume_uniform_op, .operands = shd_singleton(recovered_value) });

        register_processed(r, ovar, recovered_value);
    }

    register_processed(r, get_abstraction_mem(liftee), bb_mem(bb));
    register_processed(r, liftee, new_fn);
    const Node* substituted = rewrite_node(r, obody);
    destroy_rewriter(r);

    assert(is_terminator(substituted));
    set_abstraction_body(new_fn, finish_body(bb, substituted));

    return lifted_cont;
}

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    switch (is_declaration(node)) {
        case Function_TAG: {
            while (ctx->rewriter.parent)
                ctx = (Context*) ctx->rewriter.parent;

            Context fn_ctx = *ctx;
            fn_ctx.cfg = build_fn_cfg(node);
            fn_ctx.uses = create_fn_uses_map(node, (NcDeclaration | NcType));
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
                LiftedCont* lifted_tail = lambda_lift(ctx, ctx->cfg, otail);
                const Node* sp = add_spill_instrs(ctx, bb, lifted_tail->save_values);
                const Node* tail_ptr = fn_addr_helper(a, lifted_tail->lifted_fn);

                const Type* jp_type = join_point_type(a, (JoinPointType) {
                    .yield_types = rewrite_nodes(&ctx->rewriter, node->payload.control.yield_types),
                });
                const Node* jp = gen_ext_instruction(bb, "shady.internal", ShadyOpCreateJoinPoint,
                                                     shd_as_qualified_type(jp_type, true), mk_nodes(a, tail_ptr, sp));
                // dumbass hack
                jp = gen_primop_e(bb, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(jp));

                register_processed(r, shd_first(get_abstraction_params(oinside)), jp);
                register_processed(r, get_abstraction_mem(oinside), bb_mem(bb));
                register_processed(r, oinside, NULL);
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
        shd_debugv_print("lift_indirect_target: round %d\n", round++);
        IrArena* oa = a;
        a = new_ir_arena(&aconfig);
        dst = new_module(a, get_module_name(src));
        bool todo = false;
        Context ctx = {
            .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
            .lifted = shd_new_dict(const Node*, LiftedCont*, (HashFn) hash_node, (CmpFn) compare_node),
            .config = config,

            .todo = &todo
        };

        rewrite_module(&ctx.rewriter);

        size_t iter = 0;
        LiftedCont* lifted_cont;
        while (shd_dict_iter(ctx.lifted, &iter, NULL, &lifted_cont)) {
            free(lifted_cont);
        }
        shd_destroy_dict(ctx.lifted);
        destroy_rewriter(&ctx.rewriter);
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
