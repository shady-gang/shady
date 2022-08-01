#include "passes.h"

#include "list.h"

#include "../log.h"
#include "../portability.h"
#include "../arena.h"
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
    Rewriter unused;
    IrArena* src_arena;
    IrArena* dst_arena;
    NamedBindEntry* bound_variables;
    const Node* current_function;
} Context;

static const NamedBindEntry* resolve_using_name(const Context* ctx, const char* name) {
    for (NamedBindEntry* entry = ctx->bound_variables; entry != NULL; entry = entry->next) {
        if (strcmp(entry->name, name) == 0) {
            return entry;
        }
    }
    error("could not resolve node %s", name)
}

static void bind_named_entry(Context* ctx, NamedBindEntry* entry) {
    entry->next = ctx->bound_variables;
    ctx->bound_variables = entry;
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
            const NamedBindEntry* entry = resolve_using_name(ctx, node->payload.unbound.name);
            assert(entry->is_var);
            return entry->node;
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
    error("This doesn't really look like a place expression...");
    //return bind_node(ctx, node, true);
}

static Node* rewrite_fn_head(Context* ctx, const Node* node) {
    assert(node != NULL && node->tag == Function_TAG);
    IrArena* dst_arena = ctx->dst_arena;

    // rebuild the parameters and shove them in the list
    size_t params_count = node->payload.fn.params.count;
    LARRAY(const Node*, nparams, params_count);
    for (size_t i = 0; i < params_count; i++) {
        const Variable* old_param = &node->payload.fn.params.nodes[i]->payload.var;
        const Node* new_param = var(dst_arena, bind_node(ctx, old_param->type), string(dst_arena, old_param->name));
        nparams[i] = new_param;
    }

    return fn(dst_arena, bind_nodes(ctx, node->payload.fn.annotations), string(dst_arena, node->payload.fn.name), node->payload.fn.is_basic_block, nodes(dst_arena, params_count, nparams), bind_nodes(ctx, node->payload.fn.return_types));
}

static void rewrite_fn_body(Context* ctx, const Node* node, Node* target) {
    assert(node != NULL && node->tag == Function_TAG);
    IrArena* dst_arena = ctx->dst_arena;

    Context body_infer_ctx = *ctx;
    // bind the rebuilt parameters for rewriting the body
    for (size_t i = 0; i < node->payload.fn.params.count; i++) {
        const Node* param = target->payload.fn.params.nodes[i];
        NamedBindEntry* entry = arena_alloc(ctx->dst_arena, sizeof(NamedBindEntry));
        *entry = (NamedBindEntry) {
            .name = string(dst_arena, param->payload.var.name),
            .is_var = false,
            .node = (Node*) param,
            .next = NULL
        };
        bind_named_entry(&body_infer_ctx, entry);
        printf("Bound param %s\n", entry->name);
    }

    if (node->payload.fn.is_basic_block) {
        assert(body_infer_ctx.current_function && "basic blocks should be nested inside functions");
    } else {
        assert(ctx->current_function == NULL);
        body_infer_ctx.current_function = target;
    }
    target->payload.fn.block = bind_node(&body_infer_ctx, node->payload.fn.block);
}

static Nodes rewrite_instructions(Context* ctx, Nodes instructions) {
    IrArena* dst_arena = ctx->dst_arena;

    struct List* list = new_list(const Node*);
    for (size_t k = 0; k < instructions.count; k++) {
        const Node* old_instruction = instructions.nodes[k];
        switch (old_instruction->tag) {
            case Let_TAG: {
                const Node* bound_instr = bind_node(ctx, old_instruction->payload.let.instruction);

                size_t outputs_count = old_instruction->payload.let.variables.count;

                // TODO lift this into a helper FN
                LARRAY(const char*, names, outputs_count);
                for (size_t j = 0; j < outputs_count; j++)
                    names[j] = old_instruction->payload.let.variables.nodes[j]->payload.var.name;

                const Node* new_let = let(dst_arena, bound_instr, outputs_count, names);
                append_list(const Node*, list, new_let);

                for (size_t j = 0; j < outputs_count; j++) {
                    const Variable* old_var = &old_instruction->payload.let.variables.nodes[j]->payload.var;
                    NamedBindEntry* entry = arena_alloc(ctx->src_arena, sizeof(NamedBindEntry));

                    *entry = (NamedBindEntry) {
                        .name = string(dst_arena, old_var->name),
                        .is_var = old_instruction->payload.let.is_mutable,
                        .node = /* set later */ NULL,
                        .next = NULL
                    };

                    const Node* value = new_let->payload.let.variables.nodes[j];

                    if (old_instruction->payload.let.is_mutable) {
                        assert(old_var->type);
                        const Node* let_alloca = let(dst_arena, prim_op(dst_arena, (PrimOp) {
                            .op = alloca_op,
                            .operands = nodes(dst_arena, 1, (const Node* []){ old_var->type })
                        }), 1, &names[j]);
                        append_list(const Node*, list, let_alloca);
                        const Node* ptr = let_alloca->payload.let.variables.nodes[0];
                        const Node* store = prim_op(dst_arena, (PrimOp) {
                            .op = store_op,
                            .operands = nodes(dst_arena, 2, (const Node* []) { ptr, value })
                        });
                        append_list(const Node*, list, store);
                        // In this case, the node is a _pointer_, not the value !
                        entry->node = (Node*) ptr;
                    } else {
                        entry->node = (Node*) value;
                    }

                    bind_named_entry(ctx, entry);
                    printf("Bound primop result %s\n", entry->name);
                }

                break;
            }
            default: {
                const Node* ninstruction = bind_node(ctx, old_instruction);
                append_list(const Node*, list, ninstruction);
                break;
            }
        }
    }
    Nodes ninstructions = nodes(dst_arena, entries_count_list(list), read_list(const Node*, list));
    destroy_list(list);
    return ninstructions;
}

static const Node* bind_node(Context* ctx, const Node* node) {
    if (node == NULL)
        return NULL;

    IrArena* dst_arena = ctx->dst_arena;
    switch (node->tag) {
        case Root_TAG: {
            const Root* src_root = &node->payload.root;
            const size_t count = src_root->declarations.count;

            Context root_context = *ctx;
            LARRAY(const Node*, new_decls, count);

            for (size_t i = 0; i < count; i++) {
                const Node* decl = src_root->declarations.nodes[i];

                const Node* bound = NULL;
                NamedBindEntry* entry = arena_alloc(ctx->src_arena, sizeof(NamedBindEntry));
                entry->next = NULL;

                switch (decl->tag) {
                    case GlobalVariable_TAG: {
                        const GlobalVariable* ogvar = &decl->payload.global_variable;
                        bound = global_var(dst_arena, bind_nodes(ctx, ogvar->annotations), bind_node(ctx, ogvar->type), ogvar->name, ogvar->address_space);
                        entry->name = ogvar->name;
                        entry->is_var = true;
                        break;
                    }
                    case Constant_TAG: {
                        const Constant* cnst = &decl->payload.constant;
                        Node* new_constant = constant(dst_arena, bind_nodes(ctx, cnst->annotations), cnst->name);
                        new_constant->payload.constant.type_hint = decl->payload.constant.type_hint;
                        bound = new_constant;
                        entry->name = cnst->name;
                        break;
                    }
                    case Function_TAG: {
                        const Function* ofn = &decl->payload.fn;
                        bound = rewrite_fn_head(ctx, decl);
                        entry->name = ofn->name;
                        break;
                    }
                    default: error("unknown declaration kind");
                }

                entry->node = (Node*) bound;
                bind_named_entry(&root_context, entry);
                printf("Bound root def %s\n", entry->name);

                new_decls[i] = bound;
            }

            for (size_t i = 0; i < count; i++) {
                const Node* odecl = src_root->declarations.nodes[i];
                if (odecl->tag != GlobalVariable_TAG)
                    new_decls[i] = bind_node(&root_context, odecl);
            }

            return root(dst_arena, (Root) {
                .declarations = nodes(dst_arena, count, new_decls),
            });
        }
        case Variable_TAG: error("the binders should be handled such that this node is never reached");
        case Unbound_TAG: {
            const NamedBindEntry* entry = resolve_using_name(ctx, node->payload.unbound.name);
            if (entry->is_var) {
                return prim_op(dst_arena, (PrimOp) {
                    .op = load_op,
                    .operands = nodes(dst_arena, 1, (const Node* []) { get_node_address(ctx, node) })
                });
            } else {
                return entry->node;
            }
        }
        case Let_TAG: error("rewrite_instructions should handle this");
        case Loop_TAG: {
            Context loop_body_ctx = *ctx;
            Nodes old_params = node->payload.loop_instr.params;
            LARRAY(const Node*, new_params, old_params.count);
            for (size_t i = 0; i < old_params.count; i++) {
                const Variable* old_param = &old_params.nodes[i]->payload.var;
                const Node* new_param = var(dst_arena, bind_node(ctx, old_param->type), old_param->name);
                new_params[i] = new_param;

                NamedBindEntry* entry = arena_alloc(ctx->src_arena, sizeof(NamedBindEntry));
                *entry = (NamedBindEntry) {
                    .name = string(dst_arena, old_param->name),
                    .is_var = false,
                    .node = (Node*) new_param,
                    .next = NULL
                };
                bind_named_entry(&loop_body_ctx, entry);
                printf("Bound loop param %s\n", entry->name);
            }

            const Node* new_body = bind_node(&loop_body_ctx, node->payload.loop_instr.body);

            return loop_instr(dst_arena, (Loop) {
                .yield_types = import_nodes(dst_arena, node->payload.loop_instr.yield_types),
                .initial_args = bind_nodes(ctx, node->payload.loop_instr.initial_args),
                .body = new_body,
                .params = nodes(dst_arena, old_params.count, new_params)
            });
        }
        case ParsedBlock_TAG: {
            const ParsedBlock* pblock = &node->payload.parsed_block;
            Context pblock_ctx = *ctx;

            size_t inner_conts_count = pblock->continuations_vars.count;
            LARRAY(Node*, new_conts, inner_conts_count);

            // First create stubs and inline that crap
            for (size_t i = 0; i < inner_conts_count; i++) {
                Node* new_cont = rewrite_fn_head(ctx, pblock->continuations.nodes[i]);
                new_conts[i] = new_cont;
                NamedBindEntry* entry = arena_alloc(ctx->src_arena, sizeof(NamedBindEntry));
                *entry = (NamedBindEntry) {
                    .name = string(dst_arena, pblock->continuations_vars.nodes[i]->payload.var.name),
                    .is_var = false,
                    .node = new_cont,
                    .next = NULL
                };
                bind_named_entry(&pblock_ctx, entry);
                printf("Bound (stub) continuation %s\n", entry->name);
            }

            const Node* new_block = block(dst_arena, (Block) {
                .instructions = rewrite_instructions(&pblock_ctx, pblock->instructions),
                .terminator = bind_node(&pblock_ctx, pblock->terminator)
            });

            // Rebuild the actual continuations now
            for (size_t i = 0; i < inner_conts_count; i++) {
                rewrite_fn_body(&pblock_ctx, pblock->continuations.nodes[i], new_conts[i]);
                printf("Processed (full) continuation %s\n", new_conts[i]->payload.fn.name);
            }

            return new_block;
        }
        case Block_TAG: {
             const Node* new_block = block(dst_arena, (Block) {
                 .instructions = rewrite_instructions(ctx, node->payload.block.instructions),
                 .terminator = bind_node(ctx, node->payload.block.terminator)
             });
             return new_block;
        }
        case Return_TAG: {
            assert(ctx->current_function);
            return fn_ret(dst_arena, (Return) {
                .fn = ctx->current_function,
                .values = bind_nodes(ctx, node->payload.fn_ret.values)
            });
        }
        case Function_TAG: {
            Node* head = resolve_using_name(ctx, node->payload.fn.name)->node;
            rewrite_fn_body(ctx, node, head);
            return head;
        }
        case Constant_TAG: {
            Node* head = resolve_using_name(ctx, node->payload.fn.name)->node;
            head->payload.constant.value = bind_node(ctx, node->payload.constant.value);
            return head;
        }
        default:
        /*case Call_TAG:
        case PrimOp_TAG:*/ {
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

            Rewriter rewriter = {
                .src_arena = ctx->src_arena,
                .dst_arena = dst_arena,
                .rewrite_fn = (RewriteFn) bind_node,
            };
            Context n_ctx = *ctx;
            n_ctx.unused = rewriter;
            return recreate_node_identity(&n_ctx.unused, node);
        }
        //default: error("Unhandled node %s", node_tags[node->tag]);
    }
}

const Node* bind_program(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* source) {
    Context ctx = {
        .src_arena = src_arena,
        .dst_arena = dst_arena,
        .bound_variables = NULL
    };

    const Node* rewritten = bind_node(&ctx, source);
    return rewritten;
}
