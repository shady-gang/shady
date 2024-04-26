#include "passes.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "util.h"

#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"
#include "../analysis/uses.h"
#include "../analysis/leak.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const UsesMap* uses;
    const CompilerConfig* config;
    Arena* arena;
    struct Dict* alloca_info;
    bool todo;
} Context;

typedef struct {
    const Type* type;
    bool leaks;
    bool read_from;
    bool non_logical_use;
    const Node* bound;
} AllocaInfo;

typedef struct {
    AllocaInfo* src_alloca;
} PtrSourceKnowledge;

static void visit_ptr_uses(const Node* ptr_value, const Type* slice_type, AllocaInfo* k, const UsesMap* map) {
    const Type* ptr_type = ptr_value->type;
    bool ptr_u = deconstruct_qualified_type(&ptr_type);
    assert(ptr_type->tag == PtrType_TAG);

    const Use* use = get_first_use(map, ptr_value);
    for (;use; use = use->next_use) {
        if (is_abstraction(use->user) && use->operand_class == NcVariable)
            continue;
        else if (use->user->tag == Let_TAG && use->operand_class == NcInstruction) {
            Nodes vars = get_abstraction_params(get_let_tail(use->user));
            for (size_t i = 0; i < vars.count; i++) {
                debugv_print("demote_alloca leak analysis: following let-bound variable: ");
                log_node(DEBUGV, vars.nodes[i]);
                debugv_print(".\n");
                visit_ptr_uses(vars.nodes[i], slice_type, k, map);
            }
        } else if (use->user->tag == PrimOp_TAG) {
            PrimOp payload = use->user->payload.prim_op;
            switch (payload.op) {
                case load_op: {
                    //if (get_pointer_type_element(ptr_type) != slice_type)
                    //    k->reinterpreted = true;
                    k->read_from = true;
                    continue; // loads don't leak the address.
                }
                case store_op: {
                    //if (get_pointer_type_element(ptr_type) != slice_type)
                    //    k->reinterpreted = true;
                    // stores leak the value if it's stored
                    if (ptr_value == payload.operands.nodes[1])
                        k->leaks = true;
                    continue;
                }
                case reinterpret_op: {
                    k->non_logical_use = true;
                    continue;
                }
                case convert_op: {
                    if (first(payload.type_arguments)->tag == PtrType_TAG) {
                        k->non_logical_use = true;
                    } else {
                        k->leaks = true;
                    }
                    continue;
                }
                /*case reinterpret_op: {
                    debugvv_print("demote_alloca leak analysis: following reinterpret instr: ");
                    log_node(DEBUGVV, use->user);
                    debugvv_print(".\n");
                    visit_ptr_uses(use->user, slice_type, k, map);
                    continue;
                }
                case convert_op: {
                    if (first(payload.type_arguments)->tag == PtrType_TAG) {
                        // this is a ptr-ptr conversion, which means it's a Generic-non generic conversion
                        // these are fine, just track them
                        debugvv_print("demote_alloca leak analysis: following conversion instr: ");
                        log_node(DEBUGVV, use->user);
                        debugvv_print(".\n");
                        visit_ptr_uses(use->user, slice_type, k, map);
                        continue;
                    }
                    k->leaks = true;
                    continue;
                }*/
                case lea_op: {
                    // TODO: follow where those derived pointers are used and establish whether they leak themselves
                    // use slice_type to keep track of the expected type for the relevant sub-object
                    k->leaks = true;
                    continue;
                } default: break;
            }
            if (has_primop_got_side_effects(payload.op))
                k->leaks = true;
        } else if (use->user->tag == Composite_TAG) {
            // todo...
            // note: if a composite literal containing our POI (pointer-of-interest) is extracted from, folding ops simplify this to the original POI
            // so we don't need to be so clever here I think
            k->leaks = true;
        } else {
            k->leaks = true;
        }
    }
}

PtrSourceKnowledge get_ptr_source_knowledge(Context* ctx, const Node* ptr) {
    PtrSourceKnowledge k = { 0 };
    while (ptr) {
        assert(is_value(ptr));
        if (ptr->tag == Variable_TAG && ctx->uses) {
            const Node* instr = get_var_instruction(ctx->uses, ptr);
            if (instr) {
                PrimOp payload = instr->payload.prim_op;
                switch (payload.op) {
                    case alloca_logical_op:
                    case alloca_op: {
                        k.src_alloca = *find_value_dict(const Node*, AllocaInfo*, ctx->alloca_info, instr);
                        return k;
                    }
                    case convert_op:
                    case reinterpret_op: {
                        ptr = first(payload.operands);
                        continue;
                    }
                        // TODO: lea and co
                    default:
                        break;
                }
            }
        }

        ptr = NULL;
    }
    return k;
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (old->tag) {
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, old);
            Context fun_ctx = *ctx;
            fun_ctx.uses = create_uses_map(old, (NcDeclaration | NcType));
            fun_ctx.disable_lowering = lookup_annotation_with_string_payload(old, "DisableOpt", "demote_alloca");
            if (old->payload.fun.body)
                fun->payload.fun.body = rewrite_node(&fun_ctx.rewriter, old->payload.fun.body);
            destroy_uses_map(fun_ctx.uses);
            return fun;
        }
        case Constant_TAG: {
            Context fun_ctx = *ctx;
            fun_ctx.uses = NULL;
            return recreate_node_identity(&fun_ctx.rewriter, old);
        }
        case Let_TAG: {
            const Node* oinstruction = get_let_instruction(old);
            const Node* otail = get_let_tail(old);
            const Node* ninstruction = rewrite_node(r, oinstruction);
            AllocaInfo** found_info = find_value_dict(const Node*, AllocaInfo*, ctx->alloca_info, oinstruction);
            AllocaInfo* info = NULL;
            if (found_info) {
                const Node* ovar = first(get_abstraction_params(otail));
                info = *found_info;
                insert_dict(const Node*, AllocaInfo*, ctx->alloca_info, ovar, info);
            }
            Nodes oparams = otail->payload.case_.params;
            Nodes ntypes = unwrap_multiple_yield_types(r->dst_arena, ninstruction->type);
            assert(ntypes.count == oparams.count);
            LARRAY(const Node*, new_params, oparams.count);
            for (size_t i = 0; i < oparams.count; i++) {
                new_params[i] = var(r->dst_arena, ntypes.nodes[i], oparams.nodes[i]->payload.var.name);
                register_processed(r, oparams.nodes[i], new_params[i]);
            }
            if (info)
                info->bound = new_params[0];
            const Node* nbody = rewrite_node(r, otail->payload.case_.body);
            const Node* tail = case_(r->dst_arena, nodes(r->dst_arena, oparams.count, new_params), nbody);
            return let(a, ninstruction, tail);
        }
        case PrimOp_TAG: {
            PrimOp payload = old->payload.prim_op;
            switch (payload.op) {
                case alloca_op:
                case alloca_logical_op: {
                    AllocaInfo* k = arena_alloc(ctx->arena, sizeof(AllocaInfo));
                    *k = (AllocaInfo) { .type = rewrite_node(r, first(payload.type_arguments)) };
                    assert(ctx->uses);
                    visit_ptr_uses(old, first(payload.type_arguments), k, ctx->uses);
                    insert_dict(const Node*, AllocaInfo*, ctx->alloca_info, old, k);
                    debugv_print("demote_alloca: uses analysis results for ");
                    log_node(DEBUGV, old);
                    debugv_print(": leaks=%d read_from=%d non_logical_use=%d\n", k->leaks, k->read_from, k->non_logical_use);
                    if (!k->leaks) {
                        if (!k->read_from && !k->non_logical_use/* this should include killing dead stores! */) {
                            ctx->todo |= true;
                            return quote_helper(a, singleton(undef(a, (Undef) {.type = get_unqualified_type(rewrite_node(r, old->type))})));
                        }
                        if (!k->non_logical_use && get_arena_config(a).optimisations.weaken_non_leaking_allocas) {
                            ctx->todo |= true;
                            return prim_op_helper(a, alloca_logical_op, rewrite_nodes(&ctx->rewriter, payload.type_arguments), rewrite_nodes(r, payload.operands));
                        }
                    }
                    break;
                }
                case load_op: {
                    PtrSourceKnowledge k = get_ptr_source_knowledge(ctx, first(payload.operands));
                    if (k.src_alloca) {
                        const Type* access_type = get_pointer_type_element(get_unqualified_type(rewrite_node(r, payload.operands.nodes[0]->type)));
                        if (is_reinterpret_cast_legal(access_type, k.src_alloca->type)) {
                            if (k.src_alloca->bound == rewrite_node(r, first(payload.operands)))
                                break;
                            ctx->todo |= true;
                            BodyBuilder* bb = begin_body(a);
                            const Node* data = gen_load(bb, k.src_alloca->bound);
                            data = gen_reinterpret_cast(bb, access_type, data);
                            return yield_values_and_wrap_in_block(bb, singleton(data));
                        }
                    }
                    break;
                }
                case store_op: {
                    PtrSourceKnowledge k = get_ptr_source_knowledge(ctx, first(payload.operands));
                    if (k.src_alloca) {
                        const Type* access_type = get_pointer_type_element(get_unqualified_type(rewrite_node(r, payload.operands.nodes[0]->type)));
                        if (is_reinterpret_cast_legal(access_type, k.src_alloca->type)) {
                            if (k.src_alloca->bound == rewrite_node(r, first(payload.operands)))
                                break;
                            ctx->todo |= true;
                            BodyBuilder* bb = begin_body(a);
                            const Node* data = gen_reinterpret_cast(bb, access_type, rewrite_node(r, payload.operands.nodes[1]));
                            gen_store(bb, k.src_alloca->bound, data);
                            return yield_values_and_wrap_in_block(bb, empty(a));
                        }
                    }
                    break;
                }
                default:
                    break;
            }
            break;
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, old);
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

bool opt_demote_alloca(SHADY_UNUSED const CompilerConfig* config, Module** m) {
    Module* src = *m;
    IrArena* a = get_module_arena(src);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .arena = new_arena(),
        .alloca_info = new_dict(const Node*, AllocaInfo*, (HashFn) hash_node, (CmpFn) compare_node),
        .todo = false
    };
    ctx.rewriter.config.rebind_let = true;
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.alloca_info);
    destroy_arena(ctx.arena);
    *m = dst;
    return ctx.todo;
}
