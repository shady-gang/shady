#include "SlimFrontendOps.h"

#include "shady/pass.h"
#include "shady/fe/slim.h"
#include "shady/ir/debug.h"
#include "shady/analysis/uses.h"

#include "../shady/ir_private.h"

#include "list.h"
#include "log.h"
#include "portability.h"

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
    const UsesMap* uses;

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

    const Node* top_level_bindings = shd_module_get_exported(ctx->rewriter.src_module, "_top_level_bindings");
    assert(top_level_bindings);
    for (size_t i = 0; i < top_level_bindings->annotations.count; i++) {
        if (top_level_bindings->annotations.nodes[i]->tag != AnnotationId_TAG) continue;
        AnnotationId payload = top_level_bindings->annotations.nodes[i]->payload.annotation_id;
        if (strcmp(payload.name, name) == 0) {
            Context* root_ctx = (Context*) shd_get_top_rewriter(&ctx->rewriter);
            assert(!root_ctx->local_variables);
            const Node* odecl = payload.id;
            const Node* decl = shd_rewrite_node(&root_ctx->rewriter, odecl);
            return (Resolved) {
                .is_var = odecl->tag == GlobalVariable_TAG,
                .node = decl
            };
        }
    }

    /*const Node* ndecl = shd_module_get_declaration(ctx->rewriter.dst_module, name);
    if (ndecl)
        return (Resolved) {
            .is_var = ndecl->tag == GlobalVariable_TAG,
            .node = ndecl
        };

    const Node* odecl = shd_module_get_declaration(ctx->rewriter.src_module, name);
    if (odecl) {
        Context top_ctx = *ctx;
        top_ctx.local_variables = NULL;
        const Node* decl = shd_rewrite_node(&top_ctx.rewriter, odecl);
        return (Resolved) {
            .is_var = decl->tag == GlobalVariable_TAG,
            .node = decl
        };
    }*/

    shd_error("could not resolve node %s", name)
}

static void add_binding(Context* ctx, bool is_var, String name, const Node* node) {
    assert(name);
    NamedBindEntry* entry = shd_arena_alloc(ctx->rewriter.dst_arena->arena, sizeof(NamedBindEntry));
    *entry = (NamedBindEntry) {
        .name = shd_string(ctx->rewriter.dst_arena, name),
        .is_var = is_var,
        .node = (Node*) node,
        .next = NULL
    };
    entry->next = ctx->local_variables;
    ctx->local_variables = entry;
}

static const Node* get_node_address(Context* ctx, const Node* node);

static const Node* get_node_address_maybe(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case ExtInstr_TAG: {
            ExtInstr payload = node->payload.ext_instr;
            if (strcmp(payload.set, "shady.frontend") == 0) {
                if (payload.opcode == SlimFrontendOpsSlimSubscriptSHADY) {
                    assert(payload.operands.count == 2);
                    const Node* src_ptr = get_node_address_maybe(ctx, shd_first(payload.operands));
                    if (src_ptr == NULL)
                        return NULL;
                    const Node* index = shd_rewrite_node(&ctx->rewriter, payload.operands.nodes[1]);
                    return mem_and_value(a, (MemAndValue) {
                        .mem = shd_rewrite_node(r, payload.mem),
                        .value = ptr_composite_element(a, (PtrCompositeElement) { .ptr = src_ptr, .index = index }),
                    });
                } else if (payload.opcode == SlimFrontendOpsSlimDereferenceSHADY) {
                    assert(payload.operands.count == 1);
                    return mem_and_value(a, (MemAndValue) {
                        .mem = shd_rewrite_node(r, payload.mem),
                        .value = shd_rewrite_node(&ctx->rewriter, shd_first(payload.operands)),
                    });
                } else if (payload.opcode == SlimFrontendOpsSlimUnboundSHADY) {
                    if (payload.mem)
                        shd_rewrite_node(&ctx->rewriter, payload.mem);
                    Resolved entry = resolve_using_name(ctx, shd_get_string_literal(a, shd_first(payload.operands)));
                    // can't take the address if it's not a var!
                    if (!entry.is_var)
                        return NULL;
                    return entry.node;
                }
            }
            break;
        }
        default: break;
    }
    return NULL;
}

static const Node* get_node_address(Context* ctx, const Node* node) {
    const Node* got = get_node_address_maybe(ctx, node);
    if (!got) shd_error("This doesn't really look like a place expression...")
    return got;
}

static const Node* desugar_bind_identifiers(Context* ctx, ExtInstr instr) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    BodyBuilder* bb = instr.mem ? shd_bld_begin(a, shd_rewrite_node(r, instr.mem)) : shd_bld_begin_pure(a);

    switch (instr.opcode) {
        case SlimFrontendOpsSlimBindValSHADY: {
            size_t names_count = instr.operands.count - 1;
            const Node** names = &instr.operands.nodes[1];
            const Node* value = shd_rewrite_node(r, shd_first(instr.operands));
            Nodes results = shd_deconstruct_composite(a, value, names_count);
            for (size_t i = 0; i < names_count; i++) {
                String name = shd_get_string_literal(a, names[i]);
                shd_log_fmt(DEBUGV, "Bound immutable variable '%s'\n", name);
                add_binding(ctx, false, name, results.nodes[i]);
            }
            break;
        }
        case SlimFrontendOpsSlimBindVarSHADY: {
            size_t names_count = (instr.operands.count - 1) / 2;
            const Node** names = &instr.operands.nodes[1];
            const Node** types = &instr.operands.nodes[1 + names_count];
            const Node* value = shd_rewrite_node(r, shd_first(instr.operands));
            Nodes results = shd_deconstruct_composite(a, value, names_count);
            for (size_t i = 0; i < names_count; i++) {
                String name = shd_get_string_literal(a, names[i]);
                const Type* type_annotation = types[i];
                assert(type_annotation);
                const Node* alloca = stack_alloc(a, (StackAlloc) { .type = shd_rewrite_node(&ctx->rewriter, type_annotation), .mem = shd_bld_mem(bb) });
                const Node* ptr = shd_bld_add_instruction(bb, alloca);
                shd_set_debug_name(ptr, name);
                shd_bld_add_instruction(bb, store(a, (Store) { .ptr = ptr, .value = results.nodes[0], .mem = shd_bld_mem(bb) }));

                add_binding(ctx, true, name, ptr);
                shd_log_fmt(DEBUGV, "Bound mutable variable '%s'\n", name);
            }
            break;
        }
        case SlimFrontendOpsSlimBindContinuationsSHADY: {
            size_t names_count = (instr.operands.count ) / 2;
            const Node** names = &instr.operands.nodes[0];
            const Node** conts = &instr.operands.nodes[0 + names_count];
            LARRAY(Node*, bbs, names_count);
            for (size_t i = 0; i < names_count; i++) {
                String name = shd_get_string_literal(a, names[i]);
                Nodes nparams = shd_recreate_params(r, get_abstraction_params(conts[i]));
                bbs[i] = basic_block_helper(a, nparams);
                shd_set_debug_name(bbs[i], name);
                shd_register_processed(r, conts[i], bbs[i]);
                add_binding(ctx, false, name, bbs[i]);
                shd_log_fmt(DEBUGV, "Bound continuation '%s'\n", name);
            }
            for (size_t i = 0; i < names_count; i++) {
                Context cont_ctx = *ctx;
                Nodes bb_params = get_abstraction_params(bbs[i]);
                for (size_t j = 0; j < bb_params.count; j++) {
                    const Node* bb_param = bb_params.nodes[j];
                    assert(bb_param->tag == Param_TAG);
                    String param_name = shd_get_node_name_unsafe(bb_param);
                    if (param_name)
                        add_binding(&cont_ctx, false, param_name, bb_param);
                }
                shd_set_abstraction_body(bbs[i], shd_rewrite_node(&cont_ctx.rewriter, get_abstraction_body(conts[i])));
            }
        }
    }

    return shd_bld_to_instr_yield_values(bb, shd_empty(a));
}

static bool is_used_as_value(Context* ctx, const Node* node) {
    const Use* use = shd_get_first_use(ctx->uses, node);
    for (;use;use = use->next_use) {
        if (use->operand_class != NcMem) {
            if (use->user->tag == ExtInstr_TAG && strcmp(use->user->payload.ext_instr.set, "shady.frontend") == 0) {
                if (use->user->payload.ext_instr.opcode == SlimFrontendOpsSlimAssignSHADY && use->operand_index == 0)
                    continue;
                if (use->user->payload.ext_instr.opcode == SlimFrontendOpsSlimSubscriptSHADY && use->operand_index == 0) {
                    const Node* ptr = get_node_address_maybe(ctx, node);
                    if (ptr)
                        continue;
                }
            }
            return true;
        }
    }
    return false;
}

static const Node* bind_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    // in case the node is an l-value, we load it
    // const Node* lhs = get_node_address_safe(ctx, node);
    // if (lhs) {
    //     return load(a, (Load) { lhs, .mem = rewrite_node() });
    // }

    if (shd_get_node_name_unsafe(node) && strcmp(shd_get_node_name_unsafe(node), "_top_level_bindings") == 0)
        return NULL;

    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.rewriter = shd_create_children_rewriter(&ctx->rewriter);
            Node* bound = shd_recreate_node_head(&fn_ctx.rewriter, node);
            assert(!ctx->local_variables);
            shd_register_processed(r, node, bound);
            Nodes new_fn_params = get_abstraction_params(bound);
            for (size_t i = 0; i < new_fn_params.count; i++) {
                String param_name = shd_get_node_name_unsafe(node->payload.fun.params.nodes[i]);
                if (param_name)
                    add_binding(&fn_ctx, false, param_name, new_fn_params.nodes[i]);
            }
            shd_register_processed_list(&fn_ctx.rewriter, node->payload.fun.params, new_fn_params);

            if (node->payload.fun.body) {
                shd_set_abstraction_body(bound, shd_rewrite_node(&fn_ctx.rewriter, node->payload.fun.body));
            }
            shd_destroy_rewriter(&fn_ctx.rewriter);
            return bound;
        }
        case Param_TAG: shd_error("the binders should be handled such that this node is never reached");
        case BasicBlock_TAG: {
            assert(is_basic_block(node));
            Nodes new_params = shd_recreate_params(&ctx->rewriter, node->payload.basic_block.params);
            String name = shd_get_node_name_unsafe(node);
            Node* new_bb = basic_block_helper(a, new_params);
            Context bb_ctx = *ctx;
            ctx = &bb_ctx;
            if (name)
                add_binding(ctx, false, name, new_bb);
            for (size_t i = 0; i < new_params.count; i++) {
                String param_name = shd_get_node_name_unsafe(new_params.nodes[i]);
                if (param_name)
                    add_binding(ctx, false, param_name, new_params.nodes[i]);
            }
            shd_register_processed(&ctx->rewriter, node, new_bb);
            shd_register_processed_list(&ctx->rewriter, node->payload.basic_block.params, new_params);
            shd_set_abstraction_body(new_bb, shd_rewrite_node(&ctx->rewriter, node->payload.basic_block.body));
            return new_bb;
        }
        case ExtInstr_TAG: {
            ExtInstr payload = node->payload.ext_instr;
            if (strcmp("shady.frontend", payload.set) == 0) {
                switch ((enum SlimFrontendOpsInstructions) payload.opcode) {
                    case SlimFrontendOpsSlimDereferenceSHADY:
                        if (!is_used_as_value(ctx, node))
                            return shd_rewrite_node(r, payload.mem);
                        return load(a, (Load) {
                            .ptr = shd_rewrite_node(r, shd_first(payload.operands)),
                            .mem = shd_rewrite_node(r, payload.mem),
                        });
                    case SlimFrontendOpsSlimAssignSHADY: {
                        const Node* target_ptr = get_node_address(ctx, payload.operands.nodes[0]);
                        assert(target_ptr);
                        const Node* value = shd_rewrite_node(r, payload.operands.nodes[1]);
                        return store(a, (Store) { .ptr = target_ptr, .value = value, .mem = shd_rewrite_node(r, payload.mem) });
                    }
                    case SlimFrontendOpsSlimAddrOfSHADY: {
                        const Node* target_ptr = get_node_address(ctx, payload.operands.nodes[0]);
                        return mem_and_value(a, (MemAndValue) { .value = target_ptr, .mem = shd_rewrite_node(r, payload.mem) });
                    }
                    case SlimFrontendOpsSlimSubscriptSHADY: {
                        const Node* ptr = get_node_address_maybe(ctx, node);
                        if (ptr)
                            return load(a, (Load) {
                                .ptr = ptr,
                                .mem = shd_rewrite_node(r, payload.mem)
                            });
                        return mem_and_value(a, (MemAndValue) {
                            .value = prim_op(a, (PrimOp) {
                                .op = extract_op,
                                .operands = mk_nodes(a, shd_rewrite_node(r, payload.operands.nodes[0]), shd_rewrite_node(r, payload.operands.nodes[1]))
                            }),
                            .mem = shd_rewrite_node(r, payload.mem) }
                        );
                    }
                    case SlimFrontendOpsSlimUnboundSHADY: {
                        const Node* mem = NULL;
                        if (payload.mem) {
                            if (!is_used_as_value(ctx, node))
                                return shd_rewrite_node(r, payload.mem);
                            mem = shd_rewrite_node(r, payload.mem);
                        }
                        Resolved entry = resolve_using_name(ctx, shd_get_string_literal(a, shd_first(payload.operands)));
                        if (entry.is_var) {
                            return load(a, (Load) { .ptr = entry.node, .mem = mem });
                        } else if (mem) {
                            return mem_and_value(a, (MemAndValue) { .value = entry.node, .mem = mem });
                        }
                        return entry.node;
                    }
                    default: return desugar_bind_identifiers(ctx, payload);
                }
            }
            break;
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

Module* slim_pass_bind(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    assert(!src->arena->config.name_bound);
    aconfig.name_bound = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) bind_node),
        .local_variables = NULL,
        .uses = shd_new_uses_map_module(src, 0),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_uses_map(ctx.uses);
    return dst;
}
