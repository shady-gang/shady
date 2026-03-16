#include "shady/analysis/ptr.h"

#include "arena.h"
#include "dict.h"

struct PtrAnalysis_ {
    struct Dict* alloca_info;
    const UsesMap* uses_map;
    Arena* arena;
};

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
        } else if (use->user->tag == ScopeCast_TAG) {
            visit_ptr_uses(use->user, slice_type, k, map);
        } else {
            k->leaks = true;
        }
    }
}

PtrSourceKnowledge shd_get_ptr_source_knowledge(PtrAnalysis* ctx, const Node* ptr) {
    PtrSourceKnowledge k = { 0 };
    while (ptr) {
        assert(is_value(ptr));
        switch (ptr->tag) {
            case StackAlloc_TAG:
            case LocalAlloc_TAG: {
                AllocaInfo** found = shd_dict_find_value(const Node*, AllocaInfo*, ctx->alloca_info, ptr);
                if (found)
                    k.src_alloca = *found;
                return k;
            }
            case GlobalVariable_TAG: {
                // if it's a global variable we gotta make sure to rewrite it first
                // shd_rewrite_node(&ctx->rewriter, ptr);
                AllocaInfo** found = shd_dict_find_value(const Node*, AllocaInfo*, ctx->alloca_info, ptr);
                if (found)
                    k.src_alloca = *found;
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
            case ScopeCast_TAG: {
                ScopeCast payload = ptr->payload.scope_cast;
                ptr = payload.src;
                continue;
            }
            default: break;
        }

        ptr = NULL;
    }
    return k;
}

const AllocaInfo* shd_analyze_alloc(PtrAnalysis* ctx, const Node* old) {
    //Rewriter* r = &ctx->rewriter;
    AllocaInfo* k = shd_arena_alloc(ctx->arena, sizeof(AllocaInfo));

    const Type* old_ptr_type = shd_get_unqualified_type(old->type);
    assert(old_ptr_type->tag == PtrType_TAG);
    bool was_ref = old_ptr_type->payload.ptr_type.is_reference;
    const Type* old_type = old_ptr_type->payload.ptr_type.pointed_type;

    *k = (AllocaInfo) { .type = old_type };

    switch (old->tag) {
        case GlobalVariable_TAG: {
            // GlobalVariable payload = old->payload.global_variable;
            if (shd_lookup_annotation(old, "Exported")) {
               k->read_from = true;
            }
            break;
        }
        default: break;
    }

    if (shd_lookup_annotation(old, "DoNotDemoteToReference"))
        k->leaks = true;

    assert(ctx->uses_map);
    visit_ptr_uses(old, old_type, k, ctx->uses_map);
    shd_dict_insert(const Node*, AllocaInfo*, ctx->alloca_info, old, k);

    // shd_debugv_print("demote_alloca: uses analysis results for ");
    // NodePrintConfig config = *shd_default_node_print_config();
    // config.max_depth = 3;
    // shd_log_node_config(DEBUGV, old, &config);
    // shd_debugv_print(": leaks=%d read_from=%d non_logical_use=%d\n", k->leaks, k->read_from, k->non_logical_use);
    return k;
}

KeyHash shd_hash_node(const Node**);
bool shd_compare_node(const Node**, const Node**);

PtrAnalysis* shd_new_ptr_analysis(Module* module, const UsesMap* uses) {
    PtrAnalysis* analysis = calloc(sizeof(PtrAnalysis), 1);
    *analysis = (PtrAnalysis) {
        .arena = shd_new_arena(),
        .alloca_info = shd_new_dict(const Node*, AllocaInfo*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .uses_map = uses,
    };
    Nodes globals = shd_module_collect_reachable_globals(module);
    for (size_t i = 0; i < globals.count; i++) {
        shd_analyze_alloc(analysis, globals.nodes[i]);
    }
    return analysis;
}

void shd_destroy_ptr_analysis(PtrAnalysis* analysis) {
    shd_destroy_dict(analysis->alloca_info);
    shd_destroy_arena(analysis->arena);
    free(analysis);
}