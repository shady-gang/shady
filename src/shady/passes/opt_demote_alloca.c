#include "shady/pass.h"

#include "shady/visit.h"

#include "../ir_private.h"
#include "../check.h"
#include "../transform/ir_gen_helpers.h"
#include "../analysis/uses.h"
#include "../analysis/leak.h"

#include "log.h"
#include "portability.h"
#include "dict.h"
#include "util.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const UsesMap* uses;
    const CompilerConfig* config;
    Arena* arena;
    struct Dict* alloca_info;
    bool* todo;
} Context;

typedef struct {
    const Type* type;
    /// Set when the alloca is used in a way the analysis cannot follow
    /// Allocation must be left alone in such cases!
    bool leaks;
    /// Set when the alloca is read from.
    bool read_from;
    /// Set when the alloca is used in a manner forbidden by logical pointer rules
    bool non_logical_use;
    const Node* new;
} AllocaInfo;

typedef struct {
    AllocaInfo* src_alloca;
} PtrSourceKnowledge;

static void visit_ptr_uses(const Node* ptr_value, const Type* slice_type, AllocaInfo* k, const UsesMap* map) {
    const Type* ptr_type = ptr_value->type;
    bool ptr_u = shd_deconstruct_qualified_type(&ptr_type);
    assert(ptr_type->tag == PtrType_TAG);

    const Use* use = get_first_use(map, ptr_value);
    for (;use; use = use->next_use) {
        if (is_abstraction(use->user) && use->operand_class == NcParam)
            continue;
        if (use->operand_class == NcMem)
            continue;
        else if (use->user->tag == Load_TAG) {
            //if (get_pointer_type_element(ptr_type) != slice_type)
            //    k->reinterpreted = true;
            k->read_from = true;
            continue; // loads don't leak the address.
        } else if (use->user->tag == Store_TAG) {
            //if (get_pointer_type_element(ptr_type) != slice_type)
            //    k->reinterpreted = true;
            // stores leak the value if it's stored
            if (ptr_value == use->user->payload.store.value)
                k->leaks = true;
            continue;
        } else if (use->user->tag == PrimOp_TAG) {
            PrimOp payload = use->user->payload.prim_op;
            switch (payload.op) {
                case reinterpret_op: {
                    k->non_logical_use = true;
                    visit_ptr_uses(use->user, slice_type, k, map);
                    continue;
                }
                case convert_op: {
                    if (shd_first(payload.type_arguments)->tag == PtrType_TAG) {
                        k->non_logical_use = true;
                        visit_ptr_uses(use->user, slice_type, k, map);
                    } else {
                        k->leaks = true;
                    }
                    continue;
                }
                default: break;
            }
            if (shd_has_primop_got_side_effects(payload.op))
                k->leaks = true;
        } /*else if (use->user->tag == Lea_TAG) {
            // TODO: follow where those derived pointers are used and establish whether they leak themselves
            // use slice_type to keep track of the expected type for the relevant sub-object
            k->leaks = true;
            continue;
        } */else if (use->user->tag == Composite_TAG) {
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
        const Node* instr = ptr;
        switch (instr->tag) {
            case StackAlloc_TAG:
            case LocalAlloc_TAG: {
                k.src_alloca = *shd_dict_find_value(const Node*, AllocaInfo*, ctx->alloca_info, instr);
                return k;
            }
            case PrimOp_TAG: {
                PrimOp payload = instr->payload.prim_op;
                switch (payload.op) {
                    case convert_op:
                    case reinterpret_op: {
                        ptr = shd_first(payload.operands);
                        continue;
                    }
                    // TODO: lea and co
                    default:
                        break;
                }
            }
            default: break;
        }

        ptr = NULL;
    }
    return k;
}

static const Node* handle_alloc(Context* ctx, const Node* old, const Type* old_type) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    const Node* nmem = shd_rewrite_node(r, old->tag == StackAlloc_TAG ? old->payload.stack_alloc.mem : old->payload.local_alloc.mem);

    AllocaInfo* k = shd_arena_alloc(ctx->arena, sizeof(AllocaInfo));
    *k = (AllocaInfo) { .type = shd_rewrite_node(r, old_type) };
    assert(ctx->uses);
    visit_ptr_uses(old, old_type, k, ctx->uses);
    shd_dict_insert(const Node*, AllocaInfo*, ctx->alloca_info, old, k);
    // debugv_print("demote_alloca: uses analysis results for ");
    // log_node(DEBUGV, old);
    // debugv_print(": leaks=%d read_from=%d non_logical_use=%d\n", k->leaks, k->read_from, k->non_logical_use);
    if (!k->leaks) {
        if (!k->read_from/* this should include killing dead stores! */) {
            *ctx->todo |= true;
            const Node* new = undef(a, (Undef) { .type = shd_get_unqualified_type(shd_rewrite_node(r, old->type)) });
            new = mem_and_value(a, (MemAndValue) { .value = new, .mem = nmem });
            k->new = new;
            return new;
        }
        if (!k->non_logical_use && shd_get_arena_config(a)->optimisations.weaken_non_leaking_allocas) {
            *ctx->todo |= old->tag != LocalAlloc_TAG;
            const Node* new = local_alloc(a, (LocalAlloc) { .type = shd_rewrite_node(r, old_type), .mem = nmem });
            k->new = new;
            return new;
        }
    }
    const Node* new = shd_recreate_node(r, old);
    k->new = new;
    return new;
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case Function_TAG: {
            Node* fun = shd_recreate_node_head(&ctx->rewriter, old);
            Context fun_ctx = *ctx;
            fun_ctx.uses = create_fn_uses_map(old, (NcDeclaration | NcType));
            fun_ctx.disable_lowering = shd_lookup_annotation_with_string_payload(old, "DisableOpt", "demote_alloca");
            if (old->payload.fun.body)
                shd_set_abstraction_body(fun, shd_rewrite_node(&fun_ctx.rewriter, old->payload.fun.body));
            destroy_uses_map(fun_ctx.uses);
            return fun;
        }
        case Constant_TAG: {
            Context fun_ctx = *ctx;
            fun_ctx.uses = NULL;
            return shd_recreate_node(&fun_ctx.rewriter, old);
        }
        case Load_TAG: {
            Load payload = old->payload.load;
            shd_rewrite_node(r, payload.mem);
            PtrSourceKnowledge k = get_ptr_source_knowledge(ctx, payload.ptr);
            if (k.src_alloca) {
                const Type* access_type = get_pointer_type_element(shd_get_unqualified_type(shd_rewrite_node(r, payload.ptr->type)));
                if (shd_is_reinterpret_cast_legal(access_type, k.src_alloca->type)) {
                    if (k.src_alloca->new == shd_rewrite_node(r, payload.ptr))
                        break;
                    *ctx->todo |= true;
                    BodyBuilder* bb = begin_body_with_mem(a, shd_rewrite_node(r, payload.mem));
                    const Node* data = gen_load(bb, k.src_alloca->new);
                    data = gen_reinterpret_cast(bb, access_type, data);
                    return yield_value_and_wrap_in_block(bb, data);
                }
            }
            break;
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            shd_rewrite_node(r, payload.mem);
            PtrSourceKnowledge k = get_ptr_source_knowledge(ctx, payload.ptr);
            if (k.src_alloca) {
                const Type* access_type = get_pointer_type_element(shd_get_unqualified_type(shd_rewrite_node(r, payload.ptr->type)));
                if (shd_is_reinterpret_cast_legal(access_type, k.src_alloca->type)) {
                    if (k.src_alloca->new == shd_rewrite_node(r, payload.ptr))
                        break;
                    *ctx->todo |= true;
                    BodyBuilder* bb = begin_body_with_mem(a, shd_rewrite_node(r, payload.mem));
                    const Node* data = gen_reinterpret_cast(bb, access_type, shd_rewrite_node(r, payload.value));
                    gen_store(bb, k.src_alloca->new, data);
                    return yield_values_and_wrap_in_block(bb, shd_empty(a));
                }
            }
            break;
        }
        case LocalAlloc_TAG: return handle_alloc(ctx, old, old->payload.local_alloc.type);
        case StackAlloc_TAG: return handle_alloc(ctx, old, old->payload.stack_alloc.type);
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, old);
}

KeyHash shd_hash_node(const Node**);
bool shd_compare_node(const Node**, const Node**);

bool shd_opt_demote_alloca(SHADY_UNUSED const CompilerConfig* config, Module** m) {
    bool todo = false;
    Module* src = *m;
    IrArena* a = shd_module_get_arena(src);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .arena = shd_new_arena(),
        .alloca_info = shd_new_dict(const Node*, AllocaInfo*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .todo = &todo
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.alloca_info);
    shd_destroy_arena(ctx.arena);
    *m = dst;
    return todo;
}
