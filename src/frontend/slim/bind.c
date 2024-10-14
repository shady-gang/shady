#include "shady/pass.h"
#include "shady/fe/slim.h"

#include "../shady/ir_private.h"
#include "../shady/analysis/uses.h"

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

    Nodes new_decls = shd_module_get_declarations(ctx->rewriter.dst_module);
    for (size_t i = 0; i < new_decls.count; i++) {
        const Node* decl = new_decls.nodes[i];
        if (strcmp(get_declaration_name(decl), name) == 0) {
            return (Resolved) {
                .is_var = decl->tag == GlobalVariable_TAG,
                .node = decl
            };
        }
    }

    Nodes old_decls = shd_module_get_declarations(ctx->rewriter.src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* old_decl = old_decls.nodes[i];
        if (strcmp(get_declaration_name(old_decl), name) == 0) {
            Context top_ctx = *ctx;
            top_ctx.current_function = NULL;
            top_ctx.local_variables = NULL;
            const Node* decl = shd_rewrite_node(&top_ctx.rewriter, old_decl);
            return (Resolved) {
                .is_var = decl->tag == GlobalVariable_TAG,
                .node = decl
            };
        }
    }

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
                if (payload.opcode == SlimOpSubscript) {
                    assert(payload.operands.count == 2);
                    const Node* src_ptr = get_node_address_maybe(ctx, shd_first(payload.operands));
                    if (src_ptr == NULL)
                        return NULL;
                    const Node* index = shd_rewrite_node(&ctx->rewriter, payload.operands.nodes[1]);
                    return mem_and_value(a, (MemAndValue) {
                        .mem = shd_rewrite_node(r, payload.mem),
                        .value = ptr_composite_element(a, (PtrCompositeElement) { .ptr = src_ptr, .index = index }),
                    });
                } else if (payload.opcode == SlimOpDereference) {
                    assert(payload.operands.count == 1);
                    return mem_and_value(a, (MemAndValue) {
                        .mem = shd_rewrite_node(r, payload.mem),
                        .value = shd_rewrite_node(&ctx->rewriter, shd_first(payload.operands)),
                    });
                } else if (payload.opcode == SlimOpUnbound) {
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
        case SlimOpBindVal: {
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
        case SlimOpBindVar: {
            size_t names_count = (instr.operands.count - 1) / 2;
            const Node** names = &instr.operands.nodes[1];
            const Node** types = &instr.operands.nodes[1 + names_count];
            const Node* value = shd_rewrite_node(r, shd_first(instr.operands));
            Nodes results = shd_deconstruct_composite(a, value, names_count);
            for (size_t i = 0; i < names_count; i++) {
                String name = shd_get_string_literal(a, names[i]);
                const Type* type_annotation = types[i];
                assert(type_annotation);
                const Node* alloca = stack_alloc(a, (StackAlloc) { .type = shd_rewrite_node(&ctx->rewriter, type_annotation), .mem = shd_bb_mem(bb) });
                const Node* ptr = shd_bld_add_instruction_extract_count(bb, alloca, 1).nodes[0];
                shd_set_value_name(ptr, name);
                shd_bld_add_instruction_extract_count(bb, store(a, (Store) { .ptr = ptr, .value = results.nodes[0], .mem = shd_bb_mem(bb) }), 0);

                add_binding(ctx, true, name, ptr);
                shd_log_fmt(DEBUGV, "Bound mutable variable '%s'\n", name);
            }
            break;
        }
        case SlimOpBindContinuations: {
            size_t names_count = (instr.operands.count ) / 2;
            const Node** names = &instr.operands.nodes[0];
            const Node** conts = &instr.operands.nodes[0 + names_count];
            LARRAY(Node*, bbs, names_count);
            for (size_t i = 0; i < names_count; i++) {
                String name = shd_get_string_literal(a, names[i]);
                Nodes nparams = shd_recreate_params(r, get_abstraction_params(conts[i]));
                bbs[i] = basic_block(a, nparams, shd_get_abstraction_name_unsafe(conts[i]));
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
                    String param_name = bb_param->payload.param.name;
                    if (param_name)
                        add_binding(&cont_ctx, false, param_name, bb_param);
                }
                shd_set_abstraction_body(bbs[i], shd_rewrite_node(&cont_ctx.rewriter, get_abstraction_body(conts[i])));
            }
        }
    }

    return shd_bld_to_instr_yield_values(bb, shd_empty(a));
}

static const Node* rewrite_decl(Context* ctx, const Node* decl) {
    assert(is_declaration(decl));
    switch (decl->tag) {
        case GlobalVariable_TAG: {
            const GlobalVariable* ogvar = &decl->payload.global_variable;
            Node* bound = global_var(ctx->rewriter.dst_module, shd_rewrite_nodes(&ctx->rewriter, ogvar->annotations), shd_rewrite_node(&ctx->rewriter, ogvar->type), ogvar->name, ogvar->address_space);
            shd_register_processed(&ctx->rewriter, decl, bound);
            bound->payload.global_variable.init = shd_rewrite_node(&ctx->rewriter, decl->payload.global_variable.init);
            return bound;
        }
        case Constant_TAG: {
            const Constant* cnst = &decl->payload.constant;
            Node* bound = constant(ctx->rewriter.dst_module, shd_rewrite_nodes(&ctx->rewriter, cnst->annotations), shd_rewrite_node(&ctx->rewriter, decl->payload.constant.type_hint), cnst->name);
            shd_register_processed(&ctx->rewriter, decl, bound);
            bound->payload.constant.value = shd_rewrite_node(&ctx->rewriter, decl->payload.constant.value);
            return bound;
        }
        case Function_TAG: {
            Nodes new_fn_params = shd_recreate_params(&ctx->rewriter, decl->payload.fun.params);
            Node* bound = function(ctx->rewriter.dst_module, new_fn_params, decl->payload.fun.name, shd_rewrite_nodes(&ctx->rewriter, decl->payload.fun.annotations), shd_rewrite_nodes(&ctx->rewriter, decl->payload.fun.return_types));
            shd_register_processed(&ctx->rewriter, decl, bound);
            Context fn_ctx = *ctx;
            for (size_t i = 0; i < new_fn_params.count; i++) {
                add_binding(&fn_ctx, false, decl->payload.fun.params.nodes[i]->payload.param.name, new_fn_params.nodes[i]);
            }
            shd_register_processed_list(&ctx->rewriter, decl->payload.fun.params, new_fn_params);

            if (decl->payload.fun.body) {
                fn_ctx.current_function = bound;
                shd_set_abstraction_body(bound, shd_rewrite_node(&fn_ctx.rewriter, decl->payload.fun.body));
            }
            return bound;
        }
        case NominalType_TAG: {
            Node* bound = nominal_type(ctx->rewriter.dst_module, shd_rewrite_nodes(&ctx->rewriter, decl->payload.nom_type.annotations), decl->payload.nom_type.name);
            shd_register_processed(&ctx->rewriter, decl, bound);
            bound->payload.nom_type.body = shd_rewrite_node(&ctx->rewriter, decl->payload.nom_type.body);
            return bound;
        }
        default: shd_error("unknown declaration kind");
    }

    shd_error("unreachable")
    //register_processed(&ctx->rewriter, decl, bound);
    //return bound;
}

static bool is_used_as_value(Context* ctx, const Node* node) {
    const Use* use = shd_get_first_use(ctx->uses, node);
    for (;use;use = use->next_use) {
        if (use->operand_class != NcMem) {
            if (use->user->tag == ExtInstr_TAG && strcmp(use->user->payload.ext_instr.set, "shady.frontend") == 0) {
                if (use->user->payload.ext_instr.opcode == SlimOpAssign && use->operand_index == 0)
                    continue;
                if (use->user->payload.ext_instr.opcode == SlimOpSubscript && use->operand_index == 0) {
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

    switch (node->tag) {
        case Function_TAG:
        case Constant_TAG:
        case GlobalVariable_TAG:
        case NominalType_TAG: {
            assert(is_declaration(node));
            return rewrite_decl(ctx, node);
        }
        case Param_TAG: shd_error("the binders should be handled such that this node is never reached");
        case BasicBlock_TAG: {
            assert(is_basic_block(node));
            Nodes new_params = shd_recreate_params(&ctx->rewriter, node->payload.basic_block.params);
            String name = node->payload.basic_block.name;
            Node* new_bb = basic_block(a, new_params, name);
            Context bb_ctx = *ctx;
            ctx = &bb_ctx;
            if (name)
                add_binding(ctx, false, name, new_bb);
            for (size_t i = 0; i < new_params.count; i++) {
                String param_name = new_params.nodes[i]->payload.param.name;
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
                switch ((SlimFrontEndOpCodes) payload.opcode) {
                    case SlimOpDereference:
                        if (!is_used_as_value(ctx, node))
                            return shd_rewrite_node(r, payload.mem);
                        return load(a, (Load) {
                            .ptr = shd_rewrite_node(r, shd_first(payload.operands)),
                            .mem = shd_rewrite_node(r, payload.mem),
                        });
                    case SlimOpAssign: {
                        const Node* target_ptr = get_node_address(ctx, payload.operands.nodes[0]);
                        assert(target_ptr);
                        const Node* value = shd_rewrite_node(r, payload.operands.nodes[1]);
                        return store(a, (Store) { .ptr = target_ptr, .value = value, .mem = shd_rewrite_node(r, payload.mem) });
                    }
                    case SlimOpAddrOf: {
                        const Node* target_ptr = get_node_address(ctx, payload.operands.nodes[0]);
                        return mem_and_value(a, (MemAndValue) { .value = target_ptr, .mem = shd_rewrite_node(r, payload.mem) });
                    }
                    case SlimOpSubscript: {
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
                    case SlimOpUnbound: {
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

Module* slim_pass_bind(SHADY_UNUSED const CompilerConfig* compiler_config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    assert(!src->arena->config.name_bound);
    aconfig.name_bound = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) bind_node),
        .local_variables = NULL,
        .current_function = NULL,
        .uses = shd_new_uses_map_module(src, 0),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_uses_map(ctx.uses);
    return dst;
}
