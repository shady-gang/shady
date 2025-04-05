#include "ir_private.h"

#include "shady/pass.h"
#include "shady/visit.h"
#include "shady/ir/cast.h"
#include "shady/analysis/uses.h"
#include "shady/print.h"

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
    shd_deconstruct_qualified_type(&ptr_type);
    assert(ptr_type->tag == PtrType_TAG);

    const Use* use = shd_get_first_use(map, ptr_value);
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
        } else if (use->user->tag == Conversion_TAG) {
            Conversion payload = use->user->payload.conversion;
            if (payload.type->tag == PtrType_TAG) {
                k->non_logical_use = true;
                visit_ptr_uses(use->user, slice_type, k, map);
            } else {
                k->leaks = true;
            }
            continue;
        } else if (use->user->tag == BitCast_TAG) {
            BitCast payload = use->user->payload.bit_cast;
            if (payload.type->tag == PtrType_TAG) {
                k->non_logical_use = true;
                visit_ptr_uses(use->user, slice_type, k, map);
            } else {
                k->leaks = true;
            }
        } else if (use->user->tag == PtrArrayElementOffset_TAG) {
            visit_ptr_uses(use->user, slice_type, k, map);
            k->non_logical_use = true;
        } else if (use->user->tag == PtrCompositeElement_TAG) {
            visit_ptr_uses(use->user, slice_type, k, map);
        } else {
            k->leaks = true;
        }
    }
}

static PtrSourceKnowledge get_ptr_source_knowledge(Context* ctx, const Node* ptr) {
    PtrSourceKnowledge k = { 0 };
    while (ptr) {
        assert(is_value(ptr));
        switch (ptr->tag) {
            case StackAlloc_TAG:
            case LocalAlloc_TAG: {
                k.src_alloca = *shd_dict_find_value(const Node*, AllocaInfo*, ctx->alloca_info, ptr);
                return k;
            }
            case GlobalVariable_TAG: {
                // if it's a global variable we gotta make sure to rewrite it first
                shd_rewrite_node(&ctx->rewriter, ptr);
                k.src_alloca = *shd_dict_find_value(const Node*, AllocaInfo*, ctx->alloca_info, ptr);
                return k;
            }
            case BitCast_TAG: {
                BitCast payload = ptr->payload.bit_cast;
                ptr = payload.src;
                continue;
            }
            case Conversion_TAG: {
                Conversion payload = ptr->payload.conversion;
                ptr = payload.src;
                continue;
            }
            default: break;
        }

        ptr = NULL;
    }
    return k;
}

static AllocaInfo* analyze_alloc(Context* ctx, const Node* old, const Type* old_type) {
    Rewriter* r = &ctx->rewriter;
    AllocaInfo* k = shd_arena_alloc(ctx->arena, sizeof(AllocaInfo));
    *k = (AllocaInfo) { .type = shd_rewrite_node(r, old_type) };

    switch (old->tag) {
        case GlobalVariable_TAG: {
            // GlobalVariable payload = old->payload.global_variable;
            if (shd_lookup_annotation(old, "Exported")) {
               k->leaks = true;
            }
            break;
        }
        default: break;
    }

    assert(ctx->uses);
    visit_ptr_uses(old, old_type, k, ctx->uses);
    shd_dict_insert(const Node*, AllocaInfo*, ctx->alloca_info, old, k);

    // shd_debugv_print("demote_alloca: uses analysis results for ");
    // NodePrintConfig config = *shd_default_node_print_config();
    // config.max_depth = 3;
    // shd_log_node_config(DEBUGV, old, &config);
    // shd_debugv_print(": leaks=%d read_from=%d non_logical_use=%d\n", k->leaks, k->read_from, k->non_logical_use);
    return k;
}

static const Node* handle_alloc(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    const Type* old_ptr_type = shd_get_unqualified_type(old->type);
    assert(old_ptr_type->tag == PtrType_TAG);
    bool was_ref = old_ptr_type->payload.ptr_type.is_reference;
    const Type* old_type = old_ptr_type->payload.ptr_type.pointed_type;

    const Node* omem = is_mem(old) ? shd_get_parent_mem(old) : NULL;
    AllocaInfo* k = analyze_alloc(ctx, old, old_type);
    if (!k->leaks && !k->non_logical_use) {
        if (!k->read_from/* this should include killing dead stores! */) {
            *ctx->todo |= true;
            const Node* new = undef(a, (Undef) { .type = shd_get_unqualified_type(shd_rewrite_node(r, old->type)) });

            const Node* nmem = shd_rewrite_node(r, omem);
            new = mem_and_value(a, (MemAndValue) { .value = new, .mem = nmem });
            k->new = new;
            return new;
        } else if (shd_get_arena_config(a)->optimisations.weaken_non_leaking_allocas) {
            const Node* new;
            switch (old->tag) {
                case LocalAlloc_TAG: {
                    new = local_alloc(a, (LocalAlloc) { .type = shd_rewrite_node(r, old_type), .mem = shd_rewrite_node(r, omem) });
                    break;
                }
                case StackAlloc_TAG: {
                    *ctx->todo |= true;
                    new = local_alloc(a, (LocalAlloc) { .type = shd_rewrite_node(r, old_type), .mem = shd_rewrite_node(r, omem) });
                    break;
                }
                case GlobalVariable_TAG: {
                    GlobalVariable payload = shd_rewrite_global_head_payload(r, old->payload.global_variable);
                    *ctx->todo |= !payload.is_ref;
                    payload.is_ref = true;
                    Node* g = shd_global_var(r->dst_module, payload);
                    shd_recreate_node_body(r, old, g);
                    new = g;
                    break;
                }
                default: shd_error("Unreachable");
            }
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
            fun_ctx.rewriter = shd_create_children_rewriter(&ctx->rewriter);
            fun_ctx.disable_lowering = shd_lookup_annotation_with_string_payload(old, "DisableOpt", "demote_alloca");
            if (old->payload.fun.body)
                shd_set_abstraction_body(fun, shd_rewrite_node(&fun_ctx.rewriter, old->payload.fun.body));
            shd_destroy_rewriter(&fun_ctx.rewriter);
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
                const Type* access_type = shd_get_pointer_type_element(shd_get_unqualified_type(shd_rewrite_node(r, payload.ptr->type)));
                if (shd_is_bitcast_legal(access_type, k.src_alloca->type)) {
                    if (k.src_alloca->new == shd_rewrite_node(r, payload.ptr))
                        break;
                    *ctx->todo |= true;
                    BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
                    const Node* data = shd_bld_load(bb, k.src_alloca->new);
                    data = shd_bld_bitcast(bb, access_type, data);
                    return shd_bld_to_instr_yield_value(bb, data);
                }
            }
            break;
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            shd_rewrite_node(r, payload.mem);
            PtrSourceKnowledge k = get_ptr_source_knowledge(ctx, payload.ptr);
            if (k.src_alloca) {
                const Type* access_type = shd_get_pointer_type_element(shd_get_unqualified_type(shd_rewrite_node(r, payload.ptr->type)));
                if (shd_is_bitcast_legal(access_type, k.src_alloca->type)) {
                    if (k.src_alloca->new == shd_rewrite_node(r, payload.ptr))
                        break;
                    *ctx->todo |= true;
                    BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
                    const Node* data = shd_bld_bitcast(bb, access_type, shd_rewrite_node(r, payload.value));
                    shd_bld_store(bb, k.src_alloca->new, data);
                    return shd_bld_to_instr_yield_values(bb, shd_empty(a));
                }
            }
            break;
        }
        case GlobalVariable_TAG:
        case LocalAlloc_TAG:
        case StackAlloc_TAG: return handle_alloc(ctx, old);
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
    ctx.uses = shd_new_uses_map_module(src, NcType);
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.alloca_info);
    shd_destroy_arena(ctx.arena);
    shd_destroy_uses_map(ctx.uses);
    *m = dst;
    return todo;
}
