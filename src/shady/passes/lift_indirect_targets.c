#include "shady/ir.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include "../transform/ir_gen_helpers.h"
#include "../analysis/scope.h"
#include "../analysis/free_variables.h"
#include "../analysis/uses.h"

#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct Context_ {
    Rewriter rewriter;
    Scope* scope;
    ScopeUses* uses;

    struct Dict* lifted;
    bool disable_lowering;
    const CompilerConfig* config;
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
        assert(t->tag != PtrType_TAG || is_physical_as(t->payload.ptr_type.address_space));
        const Node* save_instruction = prim_op(a, (PrimOp) {
            .op = push_stack_op,
            .type_arguments = singleton(get_unqualified_type(nvar->type)),
            .operands = singleton(nvar),
        });
        bind_instruction(builder, save_instruction);
    }

    const Node* sp = gen_primop_ce(builder, get_stack_pointer_op, 0, NULL);

    return sp;
}

static LiftedCont* lambda_lift(Context* ctx, const Node* cont, String given_name) {
    assert(is_basic_block(cont) || is_anonymous_lambda(cont));
    LiftedCont** found = find_value_dict(const Node*, LiftedCont*, ctx->lifted, cont);
    if (found)
        return *found;

    IrArena* a = ctx->rewriter.dst_arena;
    Nodes oparams = get_abstraction_params(cont);
    const Node* obody = get_abstraction_body(cont);

    String name = is_basic_block(cont) ? format_string(a, "%s_%s", get_abstraction_name(cont->payload.basic_block.fn), get_abstraction_name(cont)) : unique_name(a, given_name);

    // Compute the live stuff we'll need
    Scope* scope = new_scope(cont);
    struct List* recover_context = compute_free_variables(scope, cont);
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

    LiftedCont* lifted_cont = calloc(sizeof(LiftedCont), 1);
    lifted_cont->old_cont = cont;
    lifted_cont->save_values = recover_context;
    insert_dict(const Node*, LiftedCont*, ctx->lifted, cont, lifted_cont);

    Context lifting_ctx = *ctx;
    lifting_ctx.rewriter = create_rewriter(ctx->rewriter.src_module, ctx->rewriter.dst_module, (RewriteFn) process_node);
    register_processed_list(&lifting_ctx.rewriter, oparams, new_params);

    const Node* payload = var(a, qualified_type_helper(uint32_type(a), false), "sp");

    // Keep annotations the same
    Nodes annotations = nodes(a, 0, NULL);
    new_params = prepend_nodes(a, new_params, payload);
    Node* new_fn = function(ctx->rewriter.dst_module, new_params, name, annotations, nodes(a, 0, NULL));
    lifted_cont->lifted_fn = new_fn;

    // Recover that stuff inside the new body
    BodyBuilder* bb = begin_body(a);
    gen_primop(bb, set_stack_pointer_op, empty(a), singleton(payload));
    for (size_t i = recover_context_size - 1; i < recover_context_size; i--) {
        const Node* ovar = read_list(const Node*, recover_context)[i];
        assert(ovar->tag == Variable_TAG);

        const Type* value_type = rewrite_node(&ctx->rewriter, ovar->type);

        const Node* recovered_value = first(bind_instruction_named(bb, prim_op(a, (PrimOp) {
            .op = pop_stack_op,
            .type_arguments = singleton(get_unqualified_type(value_type))
        }), &ovar->payload.var.name));

        if (is_qualified_type_uniform(ovar->type))
            recovered_value = first(bind_instruction_named(bb, prim_op(a, (PrimOp) { .op = subgroup_broadcast_first_op, .operands = singleton(recovered_value) }), &ovar->payload.var.name));

        register_processed(&lifting_ctx.rewriter, ovar, recovered_value);
    }

    const Node* substituted = rewrite_node(&lifting_ctx.rewriter, obody);
    //destroy_dict(lifting_ctx.rewriter.processed);
    destroy_rewriter(&lifting_ctx.rewriter);

    assert(is_terminator(substituted));
    new_fn->payload.fun.body = finish_body(bb, substituted);

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

    IrArena* a = ctx->rewriter.dst_arena;

    if (ctx->disable_lowering)
         return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.scope = new_scope(node);
            fn_ctx.uses = analyse_uses_scope(fn_ctx.scope);
            ctx = &fn_ctx;

            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            recreate_decl_body_identity(&ctx->rewriter, node, new);

            destroy_uses_scope(ctx->uses);
            destroy_scope(ctx->scope);
            return new;
        }
        case Let_TAG: {
            const Node* oinstruction = get_let_instruction(node);
            if (oinstruction->tag == Control_TAG) {
                const Node* oinside = oinstruction->payload.control.inside;
                assert(is_anonymous_lambda(oinside));
                if (!is_control_static(ctx->uses, oinstruction) || ctx->config->hacks.force_join_point_lifting) {
                    const Node* otail = get_let_tail(node);
                    BodyBuilder* bb = begin_body(a);
                    LiftedCont* lifted_tail = lambda_lift(ctx, otail, unique_name(a, format_string(a, "post_control_%s", get_abstraction_name(ctx->scope->entry->node))));
                    const Node* sp = add_spill_instrs(ctx, bb, lifted_tail->save_values);
                    const Node* tail_ptr = fn_addr_helper(a, lifted_tail->lifted_fn);

                    const Node* jp = gen_primop_e(bb, create_joint_point_op, rewrite_nodes(&ctx->rewriter, oinstruction->payload.control.yield_types), mk_nodes(a, tail_ptr, sp));

                    return finish_body(bb, let(a, quote_helper(a, singleton(jp)), rewrite_node(&ctx->rewriter, oinside)));
                }
            }

            return recreate_node_identity(&ctx->rewriter, node);
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void lift_indirect_targets(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .lifted = new_dict(const Node*, LiftedCont*, (HashFn) hash_node, (CmpFn) compare_node),
        .config = config,
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
