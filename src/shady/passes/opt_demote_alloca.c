#include "passes.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "util.h"

#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"
#include "../analysis/uses.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const UsesMap* scope_uses;
    const CompilerConfig* config;
} Context;

typedef struct {
    bool leaks;
    bool read_from;
    bool reinterpreted;
} PtrUsageKnowledge;

static void visit_ptr_uses(const Node* ptr_value, const Type* slice_type, PtrUsageKnowledge* k, const UsesMap* map) {
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
                debugv_print("mem2reg leak analysis: following let-bound variable: ");
                log_node(DEBUGV, vars.nodes[i]);
                debugv_print(".\n");
                visit_ptr_uses(vars.nodes[i], slice_type, k, map);
            }
        } else if (use->user->tag == PrimOp_TAG) {
            PrimOp payload = use->user->payload.prim_op;
            switch (payload.op) {
                case load_op: {
                    if (get_pointer_type_element(ptr_type) != slice_type)
                        k->reinterpreted = true;
                    k->read_from = true;
                    continue; // loads don't leak the address.
                }
                case store_op: {
                    if (get_pointer_type_element(ptr_type) != slice_type)
                        k->reinterpreted = true;
                    // stores leak the value if it's stored
                    if (ptr_value == payload.operands.nodes[1])
                        k->leaks = true;
                    continue;
                }
                case reinterpret_op: {
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
                }
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

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, old);
            Context fun_ctx = *ctx;
            fun_ctx.scope_uses = create_uses_map(old, (NcDeclaration | NcType));
            fun_ctx.disable_lowering = lookup_annotation_with_string_payload(old, "DisableOpt", "demote_alloca");
            if (old->payload.fun.body)
                fun->payload.fun.body = rewrite_node(&fun_ctx.rewriter, old->payload.fun.body);
            destroy_uses_map(fun_ctx.scope_uses);
            return fun;
        }
        case PrimOp_TAG: {
            PrimOp payload = old->payload.prim_op;
            switch (payload.op) {
                case alloca_op:
                case alloca_logical_op: {
                    PtrUsageKnowledge k = { 0 };
                    assert(ctx->scope_uses);
                    visit_ptr_uses(old, first(payload.type_arguments), &k, ctx->scope_uses);
                    debugv_print("demote_alloca: uses analysis results for ");
                    log_node(DEBUGV, old);
                    debugv_print(": leaks=%d read_from=%d reinterpreted=%d\n", k.leaks, k.read_from, k.reinterpreted);
                    if (!k.leaks) {
                        if (!k.read_from /* this should include killing dead stores! */)
                            return quote_helper(a, singleton(undef(a, (Undef) {.type = get_unqualified_type(rewrite_node(&ctx->rewriter, old->type))})));
                        if (!k.reinterpreted && get_arena_config(a).optimisations.weaken_non_leaking_allocas)
                            return prim_op_helper(a, alloca_logical_op, rewrite_nodes(&ctx->rewriter, payload.type_arguments), rewrite_nodes(&ctx->rewriter, payload.operands));
                    }
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

Module* opt_demote_alloca(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };
    ctx.rewriter.config.rebind_let = true;
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
