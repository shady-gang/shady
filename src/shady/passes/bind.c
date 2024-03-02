#include "passes.h"

#include "list.h"
#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../rewrite.h"

#include <assert.h>
#include <string.h>

typedef struct NamedBindEntry_ NamedBindEntry;
struct NamedBindEntry_ {
    const char* name;
    bool is_var;
    Node* node;
    NamedBindEntry* next;
};

typedef struct {
    Rewriter rewriter;

    const Node* current_function;
    NamedBindEntry* local_variables;
} Context;

typedef struct {
    bool is_var;
    const Node* node;
} Resolved;

static Resolved resolve_using_name(Context* ctx, const char* name) {
    for (NamedBindEntry* entry = ctx->local_variables; entry != NULL; entry = entry->next) {
        if (strcmp(entry->name, name) == 0) {
            return (Resolved) {
                .is_var = entry->is_var,
                .node = entry->node
            };
        }
    }

    Nodes new_decls = get_module_declarations(ctx->rewriter.dst_module);
    for (size_t i = 0; i < new_decls.count; i++) {
        const Node* decl = new_decls.nodes[i];
        if (strcmp(get_declaration_name(decl), name) == 0) {
            return (Resolved) {
                .is_var = decl->tag == GlobalVariable_TAG,
                .node = decl
            };
        }
    }

    Nodes old_decls = get_module_declarations(ctx->rewriter.src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* old_decl = old_decls.nodes[i];
        if (strcmp(get_declaration_name(old_decl), name) == 0) {
            Context top_ctx = *ctx;
            top_ctx.current_function = NULL;
            top_ctx.local_variables = NULL;
            const Node* decl = rewrite_node(&top_ctx.rewriter, old_decl);
            return (Resolved) {
                .is_var = decl->tag == GlobalVariable_TAG,
                .node = decl
            };
        }
    }

    error("could not resolve node %s", name)
}

static void add_binding(Context* ctx, bool is_var, String name, const Node* node) {
    NamedBindEntry* entry = arena_alloc(ctx->rewriter.dst_arena->arena, sizeof(NamedBindEntry));
    *entry = (NamedBindEntry) {
        .name = string(ctx->rewriter.dst_arena, name),
        .is_var = is_var,
        .node = (Node*) node,
        .next = NULL
    };
    entry->next = ctx->local_variables;
    ctx->local_variables = entry;
}

static const Node* get_node_address(Context* ctx, const Node* node);
static const Node* get_node_address_safe(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Unbound_TAG: {
            Resolved entry = resolve_using_name(ctx, node->payload.unbound.name);
            // can't take the address if it's not a var!
            if (!entry.is_var)
                return NULL;
            return entry.node;
        }
        case PrimOp_TAG: {
            if (node->tag == PrimOp_TAG && node->payload.prim_op.op == subscript_op) {
                const Node* src_ptr = get_node_address_safe(ctx, node->payload.prim_op.operands.nodes[0]);
                if (src_ptr == NULL)
                    return NULL;
                const Node* index = rewrite_node(&ctx->rewriter, node->payload.prim_op.operands.nodes[1]);
                return prim_op(a, (PrimOp) {
                    .op = lea_op,
                    .operands = nodes(a, 3, (const Node* []) {src_ptr, int32_literal(a, 0), index })
                });
            } else if (node->tag == PrimOp_TAG && node->payload.prim_op.op == deref_op) {
                return rewrite_node(&ctx->rewriter, first(node->payload.prim_op.operands));
            }
        }
        default: break;
    }
    return NULL;
}

static const Node* get_node_address(Context* ctx, const Node* node) {
    const Node* got = get_node_address_safe(ctx, node);
    if (!got)
        error("This doesn't really look like a place expression...")
    return got;
}

static const Node* desugar_let_mut(Context* ctx, const Node* node) {
    assert(node->tag == LetMut_TAG);
    IrArena* a = ctx->rewriter.dst_arena;
    const Node* ninstruction = rewrite_node(&ctx->rewriter, node->payload.let_mut.instruction);

    BodyBuilder* bb = begin_body(a);

    Nodes old_params = node->payload.let_mut.variables;
    Nodes initial_values = bind_instruction_outputs_count(bb, ninstruction, old_params.count, NULL);
    for (size_t i = 0; i < old_params.count; i++) {
        const Node* oparam = old_params.nodes[i];
        const Type* type_annotation = oparam->payload.var.type;
        assert(type_annotation);
        const Node* alloca = prim_op(a, (PrimOp) {
            .op = alloca_op,
            .type_arguments = nodes(a, 1, (const Node* []){rewrite_node(&ctx->rewriter, type_annotation) }),
            .operands = nodes(a, 0, NULL)
        });
        const Node* ptr = bind_instruction_outputs_count(bb, alloca, 1, &oparam->payload.var.name).nodes[0];
        const Node* store = prim_op(a, (PrimOp) {
            .op = store_op,
            .operands = nodes(a, 2, (const Node* []) {ptr, initial_values.nodes[0] })
        });
        bind_instruction_outputs_count(bb, store, 0, NULL);

        add_binding(ctx, true, oparam->payload.var.name, ptr);
        log_string(DEBUGV, "Lowered mutable variable ");
        log_node(DEBUGV, oparam);
        log_string(DEBUGV, ".\n;");
    }

    Nodes e = empty(a);
    return yield_values_and_wrap_in_block(bb, empty(a));
}

static const Node* rewrite_decl(Context* ctx, const Node* decl) {
    assert(is_declaration(decl));
    switch (decl->tag) {
        case GlobalVariable_TAG: {
            const GlobalVariable* ogvar = &decl->payload.global_variable;
            Node* bound = global_var(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, ogvar->annotations), rewrite_node(&ctx->rewriter, ogvar->type), ogvar->name, ogvar->address_space);
            register_processed(&ctx->rewriter, decl, bound);
            bound->payload.global_variable.init = rewrite_node(&ctx->rewriter, decl->payload.global_variable.init);
            return bound;
        }
        case Constant_TAG: {
            const Constant* cnst = &decl->payload.constant;
            Node* bound = constant(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, cnst->annotations), rewrite_node(&ctx->rewriter, decl->payload.constant.type_hint), cnst->name);
            register_processed(&ctx->rewriter, decl, bound);
            bound->payload.constant.instruction = rewrite_node(&ctx->rewriter, decl->payload.constant.instruction);
            return bound;
        }
        case Function_TAG: {
            Nodes new_fn_params = recreate_variables(&ctx->rewriter, decl->payload.fun.params);
            Node* bound = function(ctx->rewriter.dst_module, new_fn_params, decl->payload.fun.name, rewrite_nodes(&ctx->rewriter, decl->payload.fun.annotations), rewrite_nodes(&ctx->rewriter, decl->payload.fun.return_types));
            register_processed(&ctx->rewriter, decl, bound);
            Context fn_ctx = *ctx;
            for (size_t i = 0; i < new_fn_params.count; i++) {
                add_binding(&fn_ctx, false, decl->payload.fun.params.nodes[i]->payload.var.name, new_fn_params.nodes[i]);
            }
            register_processed_list(&ctx->rewriter, decl->payload.fun.params, new_fn_params);

            fn_ctx.current_function = bound;
            bound->payload.fun.body = rewrite_node(&fn_ctx.rewriter, decl->payload.fun.body);
            return bound;
        }
        case NominalType_TAG: {
            Node* bound = nominal_type(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, decl->payload.nom_type.annotations), decl->payload.nom_type.name);
            register_processed(&ctx->rewriter, decl, bound);
            bound->payload.nom_type.body = rewrite_node(&ctx->rewriter, decl->payload.nom_type.body);
            return bound;
        }
        default: error("unknown declaration kind");
    }

    error("unreachable")
    //register_processed(&ctx->rewriter, decl, bound);
    //return bound;
}

static const Node* bind_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    if (node == NULL)
        return NULL;

    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    // in case the node is an l-value, we load it
    const Node* lhs = get_node_address_safe(ctx, node);
    if (lhs) {
        return prim_op(a, (PrimOp) {
                .op = load_op,
                .operands = singleton(lhs)
        });
    }

    switch (node->tag) {
        case Function_TAG:
        case Constant_TAG:
        case GlobalVariable_TAG:
        case NominalType_TAG: {
            assert(is_declaration(node));
            return rewrite_decl(ctx, node);
        }
        case Variable_TAG: error("the binders should be handled such that this node is never reached");
        case Unbound_TAG: {
            Resolved entry = resolve_using_name(ctx, node->payload.unbound.name);
            assert(!entry.is_var);
            return entry.node;
        }
        case UnboundBBs_TAG: {
            Nodes unbound_blocks = node->payload.unbound_bbs.children_blocks;
            LARRAY(Node*, new_bbs, unbound_blocks.count);

            // First create stubs
            for (size_t i = 0; i < unbound_blocks.count; i++) {
                const Node* old_bb = unbound_blocks.nodes[i];
                assert(is_basic_block(old_bb));
                Nodes new_bb_params = recreate_variables(&ctx->rewriter, old_bb->payload.basic_block.params);
                Node* new_bb = basic_block(a, (Node*) ctx->current_function, new_bb_params, old_bb->payload.basic_block.name);
                new_bbs[i] = new_bb;
                add_binding(ctx, false, old_bb->payload.basic_block.name, new_bb);
                register_processed(&ctx->rewriter, old_bb, new_bb);
                debugv_print("Bound (stub) basic block %s\n", old_bb->payload.basic_block.name);
            }

            const Node* bound_body = rewrite_node(&ctx->rewriter, node->payload.unbound_bbs.body);

            // Rebuild the basic blocks now
            for (size_t i = 0; i < unbound_blocks.count; i++) {
                const Node* old_bb = unbound_blocks.nodes[i];
                Node* new_bb = new_bbs[i];

                Context bb_ctx = *ctx;
                Nodes new_bb_params = get_abstraction_params(new_bb);
                for (size_t j = 0; j < new_bb_params.count; j++)
                    add_binding(&bb_ctx, false, new_bb->payload.basic_block.params.nodes[j]->payload.var.name, new_bb_params.nodes[j]);

                new_bb->payload.basic_block.body = rewrite_node(&bb_ctx.rewriter, old_bb->payload.basic_block.body);
                debugv_print("Bound basic block %s\n", new_bb->payload.basic_block.name);
            }

            return bound_body;
        }
        case BasicBlock_TAG: {
            assert(is_basic_block(node));
            Nodes new_bb_params = recreate_variables(&ctx->rewriter, node->payload.basic_block.params);
            Node* new_bb = basic_block(a, (Node*) ctx->current_function, new_bb_params, node->payload.basic_block.name);
            Context bb_ctx = *ctx;
            ctx = &bb_ctx;
            add_binding(ctx, false, node->payload.basic_block.name, new_bb);
            register_processed(&ctx->rewriter, node, new_bb);
            register_processed_list(&ctx->rewriter, node->payload.basic_block.params, new_bb_params);
            new_bb->payload.basic_block.body = rewrite_node(&ctx->rewriter, node->payload.basic_block.body);
            return new_bb;
        }
        case Case_TAG: {
            Nodes old_params = node->payload.case_.params;
            Nodes new_params = recreate_variables(&ctx->rewriter, old_params);
            for (size_t i = 0; i < new_params.count; i++)
                add_binding(ctx, false, old_params.nodes[i]->payload.var.name, new_params.nodes[i]);
            register_processed_list(&ctx->rewriter, old_params, new_params);
            const Node* new_body = rewrite_node(&ctx->rewriter, node->payload.case_.body);
            return case_(a, new_params, new_body);
        }
        case LetMut_TAG: return desugar_let_mut(ctx, node);
        case Return_TAG: {
            assert(ctx->current_function);
            return fn_ret(a, (Return) {
                .fn = ctx->current_function,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.fn_ret.args)
            });
        }
        default: {
            if (node->tag == PrimOp_TAG && node->payload.prim_op.op == assign_op) {
                const Node* target_ptr = get_node_address(ctx, node->payload.prim_op.operands.nodes[0]);
                assert(target_ptr);
                const Node* value = rewrite_node(&ctx->rewriter, node->payload.prim_op.operands.nodes[1]);
                return prim_op(a, (PrimOp) {
                    .op = store_op,
                    .operands = nodes(a, 2, (const Node* []) {target_ptr, value })
                });
            } else if (node->tag == PrimOp_TAG && node->payload.prim_op.op == subscript_op) {
                return prim_op(a, (PrimOp) {
                    .op = extract_op,
                    .operands = mk_nodes(a, rewrite_node(&ctx->rewriter, node->payload.prim_op.operands.nodes[0]), rewrite_node(&ctx->rewriter, node->payload.prim_op.operands.nodes[1]))
                });
            } else if (node->tag == PrimOp_TAG && node->payload.prim_op.op == addrof_op) {
                const Node* target_ptr = get_node_address(ctx, node->payload.prim_op.operands.nodes[0]);
                return target_ptr;
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
    }
}

Module* bind_program(SHADY_UNUSED const CompilerConfig* compiler_config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    assert(!src->arena->config.name_bound);
    aconfig.name_bound = true;
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) bind_node),
        .local_variables = NULL,
        .current_function = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
