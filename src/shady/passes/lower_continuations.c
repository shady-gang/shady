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

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct Context_ {
    Rewriter rewriter;
    struct Dict* lifted;
    struct List* new_fns;
    bool disable_lowering;
} Context;

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

        const Node* args[] = { extract_operand_type(nvar->type), nvar };
        const Node* save_instruction = prim_op(arena, (PrimOp) {
            .op = is_operand_uniform(nvar->type) ? push_stack_uniform_op : push_stack_op,
            .operands = nodes(arena, 2, args)
        });
        bind_instruction(builder, save_instruction);
    }
}

static const Node* lift_lambda_into_function(Context* ctx, const Node* cont) {
    assert(cont->tag == Lambda_TAG);
    LiftedCont** found = find_value_dict(const Node*, LiftedCont*, ctx->lifted, cont);
    if (found)
        return (*found)->lifted_fn;

    IrArena* arena = ctx->rewriter.dst_arena;

    // Compute the live stuff we'll need
    Scope scope = build_scope(cont);
    struct List* recover_context = compute_free_variables(&scope);
    size_t recover_context_size = entries_count_list(recover_context);

    debug_print("free variables at '%s': ", cont->payload.lam.name);
    for (size_t i = 0; i < recover_context_size; i++) {
        debug_print("%s", read_list(const Node*, recover_context)[i]->payload.var.name);
        if (i + 1 < recover_context_size)
            debug_print(", ");
    }
    debug_print("\n");

    // Create and register new parameters for the lifted continuation
    Nodes new_params = recreate_variables(&ctx->rewriter, cont->payload.lam.params);
    register_processed_list(&ctx->rewriter, cont->payload.lam.params, new_params);

    // Keep annotations the same
    Nodes annotations = rewrite_nodes(&ctx->rewriter, cont->payload.lam.annotations);
    Node* new_fn = function(arena, new_params, cont->payload.lam.name, annotations, nodes(arena, 0, NULL));

    LiftedCont* lifted_cont = calloc(sizeof(LiftedCont), 1);
    lifted_cont->old_cont = cont;
    lifted_cont->lifted_fn = new_fn;
    lifted_cont->save_values = recover_context;
    insert_dict(const Node*, LiftedCont*, ctx->lifted, cont, lifted_cont);

    Context spilled_ctx = *ctx;

    // Rewrite the body once in the new arena with the new params
    const Node* pre_substitution = rewrite_node(&spilled_ctx.rewriter, cont->payload.lam.body);

    Rewriter substituter = create_substituter(arena);

    // Recover that stuff inside the new body
    BodyBuilder* builder = begin_body(arena);
    for (size_t i = recover_context_size - 1; i < recover_context_size; i--) {
        const Node* ovar = read_list(const Node*, recover_context)[i];
        assert(ovar->tag == Variable_TAG);

        const Node* nvar = rewrite_node(&ctx->rewriter, ovar);
        const Node* recovered_value = bind_instruction(builder, prim_op(arena, (PrimOp) {
            .op = is_operand_uniform(nvar->type) ? pop_stack_uniform_op : pop_stack_op,
            .operands = nodes(arena, 1, (const Node* []) { extract_operand_type(nvar->type) })
        })).nodes[0];

        // this dict overrides the 'processed' region
        register_processed(&substituter, nvar, recovered_value);
    }

    // Rewrite the body a second time in the new arena,
    // this time substituting the captured free variables with the recovered context
    const Node* substituted = rewrite_node(&substituter, pre_substitution);

    assert(is_terminator(substituted));
    new_fn->payload.lam.body = finish_body(builder, substituted);
    append_list(const Node*, ctx->new_fns, new_fn);

    return new_fn;
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;

    // we lift all basic blocks into functions
    if (node->tag == Lambda_TAG) {
        Context ctx2 = *ctx;
        if (node->payload.lam.tier == FnTier_BasicBlock)
            return lift_lambda_into_function(ctx, node);
        // leave other declarations alone
        return recreate_node_identity(&ctx->rewriter, node);
    }

    if (ctx->disable_lowering)
         return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        // everywhere we might call a basic block, we insert appropriate spilling context
        case Branch_TAG: {
            BodyBuilder* bb = begin_body(arena);
            const Node* ncallee = NULL;
            switch (node->payload.branch.branch_mode) {
                case BrJump: {
                    // make sure the target is rewritten before we lookup the 'lifted' dict
                    const Node* otarget = node->payload.branch.target;
                    const Node* ntarget = rewrite_node(&ctx->rewriter, otarget);

                    LiftedCont* lifted = *find_value_dict(const Node*, LiftedCont*, ctx->lifted, otarget);
                    assert(lifted->lifted_fn == ntarget);
                    add_spill_instrs(ctx, bb, lifted->save_values);

                    ncallee = fn_addr(arena, (FnAddr) { .fn = lifted->lifted_fn });
                    break;
                }
                case BrIfElse: {
                    const Node* otargets[] = { node->payload.branch.true_target, node->payload.branch.false_target };
                    const Node* ntargets[2];
                    Node* cases[2];
                    for (size_t i = 0; i < 2; i++) {
                        const Node* otarget = otargets[i];
                        const Node* ntarget = rewrite_node(&ctx->rewriter, otarget);
                        ntargets[i] = ntarget;

                        LiftedCont* lifted = *find_value_dict(const Node*, LiftedCont*, ctx->lifted, otarget);
                        assert(lifted->lifted_fn == ntarget);

                        BodyBuilder* case_builder = begin_body(arena);
                        add_spill_instrs(ctx, case_builder, lifted->save_values);
                        cases[i] = lambda(arena, nodes(arena, 0, NULL));
                        cases[i]->payload.lam.body = finish_body(case_builder, merge_construct(arena, (MergeConstruct) { .args = nodes(arena, 0, NULL), .construct = Selection }));
                    }

                    // Put the spilling code inside a selection construct
                    const Node* ncondition = rewrite_node(&ctx->rewriter, node->payload.branch.branch_condition);
                    bind_instruction(bb, if_instr(arena, (If) { .condition = ncondition, .if_true = cases[0], .if_false = cases[1], .yield_types = nodes(arena, 0, NULL) }));

                    // Make the callee selection a select
                    ncallee = gen_primop_ce(bb, select_op, 3, (const Node* []) { ncondition, fn_addr(arena, (FnAddr) { .fn = ntargets[0] }), fn_addr(arena, (FnAddr) { .fn = ntargets[1] }) });
                    break;
                }
                case BrSwitch: error("TODO")
            }
            assert(ncallee && is_value(ncallee));
            return finish_body(bb, tail_call(arena, (TailCall) {
                .target = ncallee,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.branch.args),
            }));
        }
        case Let_TAG: {
            const Node* otail = node->payload.let.tail;
            if (otail->payload.lam.tier == FnTier_Lambda)
                return recreate_node_identity(&ctx->rewriter, node);
            // if tail is a BB, add all the context-saving stuff in front
            assert(otail->payload.lam.tier == FnTier_BasicBlock);
            BodyBuilder* bb = begin_body(arena);
            const Node* ntail = rewrite_node(&ctx->rewriter, otail);
            LiftedCont* lifted = *find_value_dict(const Node*, LiftedCont*, ctx->lifted, otail);
            assert(lifted->lifted_fn == ntail);
            Context ctx2 = *ctx;
            add_spill_instrs(&ctx2, bb, lifted->save_values);
            return finish_body(bb, let(arena, false, rewrite_node(&ctx2.rewriter, node->payload.let.instruction), ntail));
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

const Node* lower_continuations(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    Context ctx = {
        .rewriter = create_rewriter(src_arena, dst_arena, (RewriteFn) process_node),
        .new_fns = new_list(const Node*),
        .lifted = new_dict(const Node*, LiftedCont*, (HashFn) hash_node, (CmpFn) compare_node),
    };

    assert(src_program->tag == Root_TAG);

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);
    Nodes new_decls = rewritten->payload.root.declarations;
    for (size_t i = 0; i < entries_count_list(ctx.new_fns); i++) {
        new_decls = append_nodes(dst_arena, new_decls, read_list(const Node*, ctx.new_fns)[i]);
    }
    rewritten = root(dst_arena, (Root) {
        .declarations = new_decls
    });

    destroy_list(ctx.new_fns);
    {
        size_t iter = 0;
        LiftedCont* lifted_cont;
        while (dict_iter(ctx.lifted, &iter, NULL, &lifted_cont)) {
            destroy_list(lifted_cont->save_values);
            free(lifted_cont);
        }
        destroy_dict(ctx.lifted);
    }
    destroy_rewriter(&ctx.rewriter);
    return rewritten;
}
