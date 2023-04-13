#include "shady/ir.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include "../transform/ir_gen_helpers.h"
#include "../analysis/scope.h"
#include "../analysis/free_variables.h"

#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct Context_ {
    Rewriter rewriter;
    struct Dict* lifted;
    bool disable_lowering;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

typedef struct {
    const Node* old_cont;
    const Node* lifted_fn;
    struct List* save_values;
} LiftedCont;

#pragma GCC diagnostic error "-Wswitch"

static void add_spill_instrs(Context* ctx, BodyBuilder* builder, struct List* spilled_vars) {
    IrArena* arena = ctx->rewriter.dst_arena;

    size_t recover_context_size = entries_count_list(spilled_vars);
    for (size_t i = 0; i < recover_context_size; i++) {
        const Node* ovar = read_list(const Node*, spilled_vars)[i];
        const Node* nvar = rewrite_node(&ctx->rewriter, ovar);

        const Node* save_instruction = prim_op(arena, (PrimOp) {
            .op = push_stack_op,
            .type_arguments = singleton(get_unqualified_type(nvar->type)),
            .operands = singleton(nvar),
        });
        bind_instruction(builder, save_instruction);
    }
}

static LiftedCont* lambda_lift(Context* ctx, const Node* cont, String given_name) {
    assert(is_basic_block(cont) || is_anonymous_lambda(cont));
    LiftedCont** found = find_value_dict(const Node*, LiftedCont*, ctx->lifted, cont);
    if (found)
        return *found;

    IrArena* arena = ctx->rewriter.dst_arena;
    Nodes oparams = get_abstraction_params(cont);
    const Node* obody = get_abstraction_body(cont);

    String name = is_basic_block(cont) ? format_string(arena, "%s_%s", get_abstraction_name(cont->payload.basic_block.fn), get_abstraction_name(cont)) : unique_name(arena, "given_name");

    // Compute the live stuff we'll need
    Scope* scope = new_scope(cont);
    struct List* recover_context = compute_free_variables(scope);
    size_t recover_context_size = entries_count_list(recover_context);
    destroy_scope(scope);

    debugv_print("free (spilled) variables at '%s': ", name);
    for (size_t i = 0; i < recover_context_size; i++) {
        const Node* item = read_list(const Node*, recover_context)[i];
        debugv_print("%s~%d", item->payload.var.name, item->payload.var.id);
        if (i + 1 < recover_context_size)
            debugv_print(", ");
    }
    debugv_print("\n");

    // Create and register new parameters for the lifted continuation
    Nodes new_params = recreate_variables(&ctx->rewriter, oparams);

    // Keep annotations the same
    Nodes annotations = nodes(arena, 0, NULL);
    Node* new_fn = function(ctx->rewriter.dst_module, new_params, name, annotations, nodes(arena, 0, NULL));

    LiftedCont* lifted_cont = calloc(sizeof(LiftedCont), 1);
    lifted_cont->old_cont = cont;
    lifted_cont->lifted_fn = new_fn;
    lifted_cont->save_values = recover_context;
    insert_dict(const Node*, LiftedCont*, ctx->lifted, cont, lifted_cont);

    Context lifting_ctx = *ctx;
    lifting_ctx.rewriter = create_rewriter(ctx->rewriter.src_module, ctx->rewriter.dst_module, (RewriteFn) process_node);
    register_processed_list(&lifting_ctx.rewriter, oparams, new_params);

    // Recover that stuff inside the new body
    BodyBuilder* builder = begin_body(arena);
    for (size_t i = recover_context_size - 1; i < recover_context_size; i--) {
        const Node* ovar = read_list(const Node*, recover_context)[i];
        assert(ovar->tag == Variable_TAG);

        const Type* value_type = rewrite_node(&ctx->rewriter, ovar->type);

        const Node* recovered_value = first(bind_instruction_named(builder, prim_op(arena, (PrimOp) {
            .op = pop_stack_op,
            .type_arguments = singleton(get_unqualified_type(value_type))
        }), &ovar->payload.var.name));

        if (is_qualified_type_uniform(ovar->type))
            recovered_value = first(bind_instruction_named(builder, prim_op(arena, (PrimOp) { .op = subgroup_broadcast_first_op, .operands = singleton(recovered_value) }), &ovar->payload.var.name));

        register_processed(&lifting_ctx.rewriter, ovar, recovered_value);
    }

    const Node* substituted = rewrite_node(&lifting_ctx.rewriter, obody);
    //destroy_dict(lifting_ctx.rewriter.processed);
    destroy_rewriter(&lifting_ctx.rewriter);

    assert(is_terminator(substituted));
    new_fn->payload.fun.body = finish_body(builder, substituted);

    return lifted_cont;
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    // TODO: share this code
    if (is_declaration(node)) {
        String name = get_decl_name(node);
        Nodes decls = get_module_declarations(ctx->rewriter.dst_module);
        for (size_t i = 0; i < decls.count; i++) {
            if (strcmp(get_decl_name(decls.nodes[i]), name) == 0)
                return decls.nodes[i];
        }
    }

    IrArena* arena = ctx->rewriter.dst_arena;

    if (ctx->disable_lowering)
         return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        // everywhere we might call a basic block, we insert appropriate spilling context
        case Jump_TAG: {
            BodyBuilder* bb = begin_body(arena);
            const Node* otarget = node->payload.jump.target;
            LiftedCont* lifted = lambda_lift(ctx, otarget, NULL);

            add_spill_instrs(ctx, bb, lifted->save_values);

            const Node* ncallee = fn_addr(arena, (FnAddr) { .fn = lifted->lifted_fn });
            assert(ncallee && is_value(ncallee));
            return finish_body(bb, tail_call(arena, (TailCall) {
                .target = ncallee,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.jump.args),
            }));
        }
        case Branch_TAG: {
            BodyBuilder* bb = begin_body(arena);
            const Node* ncallee = NULL;

            const Node* otargets[] = { node->payload.branch.true_target, node->payload.branch.false_target };
            const Node* ntargets[2];
            const Node* cases[2];
            for (size_t i = 0; i < 2; i++) {
                const Node* otarget = otargets[i];

                LiftedCont* lifted = lambda_lift(ctx, otarget, NULL);
                ntargets[i] = lifted->lifted_fn;

                BodyBuilder* case_builder = begin_body(arena);
                add_spill_instrs(ctx, case_builder, lifted->save_values);
                const Node* case_body = finish_body(case_builder, merge_selection(arena, (MergeSelection) { .args = nodes(arena, 0, NULL) }));
                cases[i] = lambda(arena, nodes(arena, 0, NULL), case_body);
            }

            // Put the spilling code inside a selection construct
            const Node* ncondition = rewrite_node(&ctx->rewriter, node->payload.branch.branch_condition);
            bind_instruction(bb, if_instr(arena, (If) { .condition = ncondition, .if_true = cases[0], .if_false = cases[1], .yield_types = nodes(arena, 0, NULL) }));

            // Make the callee selection a select
            ncallee = gen_primop_ce(bb, select_op, 3, (const Node* []) { ncondition, fn_addr(arena, (FnAddr) { .fn = ntargets[0] }), fn_addr(arena, (FnAddr) { .fn = ntargets[1] }) });

            assert(ncallee && is_value(ncallee));
            return finish_body(bb, tail_call(arena, (TailCall) {
                .target = ncallee,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.branch.args),
            }));
        }
        case Let_TAG: {
            const Node* oinstruction = get_let_instruction(node);
            if (oinstruction->tag == Control_TAG) {
                const Node* otail = get_let_tail(node);
                BodyBuilder* bb = begin_body(arena);
                LiftedCont* lifted_tail = lambda_lift(ctx, otail, unique_name(arena, "let_tail"));
                // if tail is a BB, add all the context-saving stuff in front
                add_spill_instrs(ctx, bb, lifted_tail->save_values);
                const Node* tail_ptr = fn_addr(arena, (FnAddr) { .fn = lifted_tail->lifted_fn });

                LiftedCont* lifted_body = lambda_lift(ctx, oinstruction->payload.control.inside, unique_name(arena, "control_body"));
                const Node* jp = gen_primop_e(bb, create_joint_point_op, rewrite_nodes(&ctx->rewriter, oinstruction->payload.control.yield_types), singleton(tail_ptr));
                add_spill_instrs(ctx, bb, lifted_body->save_values);
                const Node* lifted_body_ptr = fn_addr(arena, (FnAddr) { .fn = lifted_body->lifted_fn });
                return finish_body(bb, tail_call(arena, (TailCall) { .target = lifted_body_ptr, .args = singleton(jp) }));
            }

            return recreate_node_identity(&ctx->rewriter, node);
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void lower_continuations(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .lifted = new_dict(const Node*, LiftedCont*, (HashFn) hash_node, (CmpFn) compare_node),
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
}
