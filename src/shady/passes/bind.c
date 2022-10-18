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
    IrArena* src_arena;
    IrArena* dst_arena;
    const Node* old_root;

    const Node** new_decls;
    /// Top level declarations are bound lazily, they may be reordered
    size_t* new_decls_count;

    const Node* current_function;
    NamedBindEntry* local_variables;
} Context;

static const Node* rewrite_decl(Context* ctx, const Node* decl);

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

    for (size_t i = 0; i < *ctx->new_decls_count; i++) {
        const Node* decl = ctx->new_decls[i];
        if (strcmp(get_decl_name(decl), name) == 0) {
            return (Resolved) {
                .is_var = decl->tag == GlobalVariable_TAG,
                .node = decl
            };
        }
    }

    Nodes root_decls = ctx->old_root->payload.root.declarations;
    for (size_t i = 0; i < root_decls.count; i++) {
        const Node* old_decl = root_decls.nodes[i];
        if (strcmp(get_decl_name(old_decl), name) == 0) {
            Context top_ctx = *ctx;
            top_ctx.current_function = NULL;
            top_ctx.local_variables = NULL;
            const Node* decl = rewrite_decl(&top_ctx, old_decl);
            return (Resolved) {
                .is_var = decl->tag == GlobalVariable_TAG,
                .node = decl
            };
        }
    }

    error("could not resolve node %s", name)
}

static void add_binding(Context* ctx, NamedBindEntry* entry) {
    entry->next = ctx->local_variables;
    ctx->local_variables = entry;
}

static const Node* bind_node(Context* ctx, const Node* node);

static Nodes bind_nodes(Context* ctx, Nodes old) {
    LARRAY(const Node*, arr, old.count);
    for (size_t i = 0; i < old.count; i++)
        arr[i] = bind_node(ctx, old.nodes[i]);
    return nodes(ctx->dst_arena, old.count, arr);
}

static const Node* get_node_address(Context* ctx, const Node* node) {
    IrArena* dst_arena = ctx->dst_arena;
    switch (node->tag) {
        case Unbound_TAG: {
            Resolved entry = resolve_using_name(ctx, node->payload.unbound.name);
            assert(entry.is_var && "Cannot take the address");
            return entry.node;
        }
        case PrimOp_TAG: {
            if (node->tag == PrimOp_TAG && node->payload.prim_op.op == subscript_op) {
                const Node* src_ptr = get_node_address(ctx, node->payload.prim_op.operands.nodes[0]);
                const Node* index = bind_node(ctx, node->payload.prim_op.operands.nodes[1]);
                return prim_op(dst_arena, (PrimOp) {
                    .op = lea_op,
                    .operands = nodes(dst_arena, 3, (const Node* []) { src_ptr, NULL, index })
                });
            }
        }
        default: break;
    }
    error("This doesn't really look like a place expression...")
}

static Node* rewrite_lambda_head(Context* ctx, const Node* node) {
    assert(node != NULL && node->tag == Lambda_TAG);
    IrArena* dst_arena = ctx->dst_arena;

    // rebuild the parameters and shove them in the list
    size_t params_count = node->payload.lam.params.count;
    LARRAY(const Node*, nparams, params_count);
    for (size_t i = 0; i < params_count; i++) {
        const Variable* old_param = &node->payload.lam.params.nodes[i]->payload.var;
        const Node* new_param = var(dst_arena, bind_node(ctx, old_param->type), string(dst_arena, old_param->name));

        nparams[i] = new_param;
    }

    switch (node->payload.lam.tier) {
        case FnTier_Lambda:
            return lambda(dst_arena, nodes(dst_arena, params_count, nparams));
        case FnTier_BasicBlock:
            return basic_block(dst_arena, nodes(dst_arena, params_count, nparams), string(dst_arena, node->payload.lam.name));
        case FnTier_Function:
            return function(dst_arena, nodes(dst_arena, params_count, nparams), string(dst_arena, node->payload.lam.name), bind_nodes(ctx, node->payload.lam.annotations), bind_nodes(ctx, node->payload.lam.return_types));
    }
    SHADY_UNREACHABLE;
}

static void rewrite_lambda_body(Context* ctx, const Node* node, Node* target) {
    assert(node != NULL && node->tag == Lambda_TAG);
    IrArena* dst_arena = ctx->dst_arena;

    Context body_infer_ctx = *ctx;
    // bind the rebuilt parameters for rewriting the body
    for (size_t i = 0; i < node->payload.lam.params.count; i++) {
        const Node* param = target->payload.lam.params.nodes[i];

        NamedBindEntry* entry = arena_alloc(ctx->dst_arena->arena, sizeof(NamedBindEntry));
        *entry = (NamedBindEntry) {
            .name = string(dst_arena, param->payload.var.name),
            .is_var = false,
            .node = (Node*) param,
            .next = NULL
        };

        entry->node = (Node*) param;

        add_binding(&body_infer_ctx, entry);
        debug_print("Bound param %s\n", entry->name);
    }

    // ensure we bind basic blocks inside functions
    if (node->payload.lam.tier != FnTier_Function) {
        assert(body_infer_ctx.current_function && "basic blocks should be nested inside functions");
    } else {
        // and we don't bind functions inside functions
        assert(ctx->current_function == NULL);
        body_infer_ctx.current_function = target;
    }

    // handle basic blocks if we are a function
    size_t inner_conts_count = node->payload.lam.children_continuations.count;
    LARRAY(Node*, new_conts, inner_conts_count);
    assert(node->payload.lam.tier == FnTier_Function || inner_conts_count == 0);

    // First create stubs and inline that crap
    for (size_t i = 0; i < inner_conts_count; i++) {
        Node* new_cont = rewrite_lambda_head(ctx, node->payload.lam.children_continuations.nodes[i]);
        new_conts[i] = new_cont;
        NamedBindEntry* entry = arena_alloc(ctx->src_arena->arena, sizeof(NamedBindEntry));
        *entry = (NamedBindEntry) {
            .name = new_cont->payload.lam.name,
            .is_var = false,
            .node = new_cont,
            .next = NULL
        };
        add_binding(&body_infer_ctx, entry);
        debug_print("Bound (stub) basic block %s\n", entry->name);
    }

    target->payload.lam.body = bind_node(&body_infer_ctx, node->payload.lam.body);

    // Rebuild the basic blocks now
    for (size_t i = 0; i < inner_conts_count; i++) {
        rewrite_lambda_body(&body_infer_ctx, node->payload.lam.children_continuations.nodes[i], new_conts[i]);
        debug_print("Finished binding basic block %s\n", new_conts[i]->payload.lam.name);
    }
}

static const Node* bind_let(Context* ctx, const Node* node) {
    IrArena* dst_arena = ctx->dst_arena;
    Context body_infer_ctx = *ctx;
    const Node* ninstruction = bind_node(ctx, node->payload.let.instruction);
    if (node->payload.let.instruction) {
        const Node* old_lam = node->payload.let.tail;
        assert(old_lam && old_lam->tag == Lambda_TAG && old_lam->payload.lam.tier == FnTier_Lambda);

        BodyBuilder* bb = begin_body(dst_arena);

        Nodes initial_values = declare_local_variable(bb, ninstruction, false, NULL, old_lam->payload.lam.params.count, NULL);
        Nodes old_params = old_lam->payload.lam.params;
        for (size_t i = 0; i < old_params.count; i++) {
            const Node* oparam = old_params.nodes[i];
            assert(oparam->type);
            const Node* alloca = prim_op(dst_arena, (PrimOp) {
                .op = alloca_op,
                .operands = nodes(dst_arena, 1, (const Node* []){ bind_node(ctx, oparam->type) })
            });
            const Node* ptr = declare_local_variable(bb, alloca, false, NULL, 1, &oparam->payload.var.name).nodes[0];
            const Node* store = prim_op(dst_arena, (PrimOp) {
                .op = store_op,
                .operands = nodes(dst_arena, 2, (const Node* []) { ptr, initial_values.nodes[0] })
            });
            append_instruction(bb, store);

            NamedBindEntry* entry = arena_alloc(ctx->dst_arena->arena, sizeof(NamedBindEntry));
            *entry = (NamedBindEntry) {
                .name = string(dst_arena, oparam->payload.var.name),
                .is_var = true,
                .node = (Node*) ptr,
                .next = NULL
            };

            entry->node = (Node*) ptr;

            add_binding(&body_infer_ctx, entry);
            debug_print("Lowered mutable variable %s\n", entry->name);
        }

        const Node* terminator = bind_node(&body_infer_ctx, old_lam->payload.lam.body);

        return finish_body(bb, terminator);
    } else {
        Node* ntail = rewrite_lambda_head(ctx, node->payload.let.tail);
        rewrite_lambda_body(ctx, node->payload.let.tail, ntail);
        return let(dst_arena, false, ninstruction, ntail);
    }
}

static const Node* rewrite_decl(Context* ctx, const Node* decl) {
    assert(is_declaration(decl));
    IrArena* dst_arena = ctx->dst_arena;
    Node* bound = NULL;

    switch (decl->tag) {
        case GlobalVariable_TAG: {
            const GlobalVariable* ogvar = &decl->payload.global_variable;
            bound = global_var(dst_arena, bind_nodes(ctx, ogvar->annotations), bind_node(ctx, ogvar->type), ogvar->name, ogvar->address_space);
            break;
        }
        case Constant_TAG: {
            const Constant* cnst = &decl->payload.constant;
            Node* new_constant = constant(dst_arena, bind_nodes(ctx, cnst->annotations), cnst->name);
            new_constant->payload.constant.type_hint = bind_node(ctx, decl->payload.constant.type_hint);
            bound = new_constant;
            break;
        }
        case Lambda_TAG: {
            bound = rewrite_lambda_head(ctx, decl);
            break;
        }
        default: error("unknown declaration kind");
    }

    ctx->new_decls[(*ctx->new_decls_count)++] = bound;
    debug_print("Bound declaration %s\n", get_decl_name(bound));

    // Handle the bodies after registering the heads
    switch (decl->tag) {
        case Lambda_TAG: rewrite_lambda_body(ctx, decl, bound); break;
        case Constant_TAG: bound->payload.constant.value = bind_node(ctx, decl->payload.constant.value); break;
        case GlobalVariable_TAG: bound->payload.global_variable.init = bind_node(ctx, decl->payload.global_variable.init); break;
        default: SHADY_UNREACHABLE;
    }

    return bound;
}

static const Node* bind_node(Context* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    IrArena* dst_arena = ctx->dst_arena;
    switch (node->tag) {
        case Lambda_TAG:
        case Constant_TAG:
        case GlobalVariable_TAG: {
            assert(false);
            break;
        }
        case Root_TAG: assert(false);
        case Variable_TAG: error("the binders should be handled such that this node is never reached");
        case Unbound_TAG: {
            Resolved entry = resolve_using_name(ctx, node->payload.unbound.name);
            if (entry.is_var) {
                return prim_op(dst_arena, (PrimOp) {
                    .op = load_op,
                    .operands = nodes(dst_arena, 1, (const Node* []) { get_node_address(ctx, node) })
                });
            } else {
                return entry.node;
            }
        }
        case Let_TAG: return bind_let(ctx, node);
        case Loop_TAG: {
            Context loop_body_ctx = *ctx;
            Nodes old_params = node->payload.loop_instr.params;
            LARRAY(const Node*, new_params, old_params.count);
            for (size_t i = 0; i < old_params.count; i++) {
                const Variable* old_param = &old_params.nodes[i]->payload.var;
                const Node* new_param = var(dst_arena, bind_node(ctx, old_param->type), old_param->name);
                new_params[i] = new_param;

                NamedBindEntry* entry = arena_alloc(ctx->src_arena->arena, sizeof(NamedBindEntry));
                *entry = (NamedBindEntry) {
                    .name = string(dst_arena, old_param->name),
                    .is_var = false,
                    .node = (Node*) new_param,
                    .next = NULL
                };
                add_binding(&loop_body_ctx, entry);
                debug_print("Bound loop param %s\n", entry->name);
            }

            const Node* new_body = bind_node(&loop_body_ctx, node->payload.loop_instr.body);

            return loop_instr(dst_arena, (Loop) {
                .yield_types = import_nodes(dst_arena, node->payload.loop_instr.yield_types),
                .initial_args = bind_nodes(ctx, node->payload.loop_instr.initial_args),
                .body = new_body,
                .params = nodes(dst_arena, old_params.count, new_params)
            });
        }
        case Return_TAG: {
            assert(ctx->current_function);
            return fn_ret(dst_arena, (Return) {
                .fn = ctx->current_function,
                .values = bind_nodes(ctx, node->payload.fn_ret.values)
            });
        }
        default: {
            if (node->tag == PrimOp_TAG && node->payload.prim_op.op == assign_op) {
                const Node* target_ptr = get_node_address(ctx, node->payload.prim_op.operands.nodes[0]);
                const Node* value = bind_node(ctx, node->payload.prim_op.operands.nodes[1]);
                return prim_op(dst_arena, (PrimOp) {
                    .op = store_op,
                    .operands = nodes(dst_arena, 2, (const Node* []) { target_ptr, value })
                });
            } else if (node->tag == PrimOp_TAG && node->payload.prim_op.op == subscript_op) {
                return prim_op(dst_arena, (PrimOp) {
                    .op = load_op,
                    .operands = nodes(dst_arena, 1, (const Node* []) { get_node_address(ctx, node) })
                });
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
    }
}

const Node* bind_program(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* source) {
    Nodes decls = source->payload.root.declarations;
    LARRAY(const Node*, new_decls, decls.count);
    size_t decls_count = 0;

    Context ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_fn = (RewriteFn) bind_node,
        },
        .src_arena = src_arena,
        .dst_arena = dst_arena,
        .old_root = source,
        .local_variables = NULL,
        .current_function = NULL,
        .new_decls = new_decls,
        .new_decls_count = &decls_count,
    };

    for (size_t i = 0; i < decls.count; i++)
        resolve_using_name(&ctx, get_decl_name(decls.nodes[i]));

    assert(decls_count == decls.count);
    return root(dst_arena, (Root) {
        .declarations = nodes(dst_arena, decls.count, new_decls),
    });
}
