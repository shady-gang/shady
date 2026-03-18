#include "shady/analysis/ptr.h"
#include "shady/visit.h"

#include "arena.h"
#include "dict.h"
#include "log.h"

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
        if (use->user->tag == Load_TAG) {
            k->read_from = true;
        } else if (use->user->tag == Store_TAG) {
            if (ptr_value == use->user->payload.store.value) {
                // stores leak the value if it's stored
                k->leaks = true;
                // storing a pointer is also not a legal operation for logical pointers
                k->non_logical_use = true;
            }
        } else if (use->user->tag == Conversion_TAG) {
            assert(false && "this is dead code right ?");
            Conversion payload = use->user->payload.conversion;
            if (payload.type->tag == PtrType_TAG) {
                visit_ptr_uses(use->user, slice_type, k, map);
            } else {
                k->leaks = true;
            }
        } else if (use->user->tag == BitCast_TAG) {
            BitCast payload = use->user->payload.bit_cast;
            // bitcasts are never legal logical uses
            k->non_logical_use = true;
            if (payload.type->tag == PtrType_TAG) {
                // though we need to track the result to know if this is also leaking the address
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
        } else if (use->user->tag == GenericPtrCast_TAG) {
            visit_ptr_uses(use->user, slice_type, k, map);
            k->non_logical_use = true;
        } else {
            k->non_logical_use = true;
            k->leaks = true;
        }
    }
}

const AllocaInfo* shd_get_memory_declaration_info(PtrAnalysis* ptr_analysis, const Node* ptr) {
    switch (ptr->tag) {
        case BuiltinRef_TAG:
        case LocalAlloc_TAG:
        case GlobalVariable_TAG: {
            AllocaInfo** found = shd_dict_find_value(const Node*, AllocaInfo*, ptr_analysis->alloca_info, ptr);
            if (found)
                return *found;
            return NULL;
        }
        default: break;
    }
    shd_error("Not memory declaration")
}

const AllocaInfo* shd_find_memory_declaration(PtrAnalysis* ptr_analysis, const Node* ptr, bool allow_non_logical_ops) {
    while (ptr) {
        assert(is_value(ptr));
        switch (ptr->tag) {
            case BuiltinRef_TAG:
            case LocalAlloc_TAG:
            case GlobalVariable_TAG: return shd_get_memory_declaration_info(ptr_analysis, ptr);

            case PtrArrayElementOffset_TAG: {
                if (allow_non_logical_ops) {
                    PtrArrayElementOffset payload = ptr->payload.ptr_array_element_offset;
                    ptr = payload.ptr;
                    continue;
                }
                break;
            }
            case PtrCompositeElement_TAG: {
                PtrCompositeElement payload = ptr->payload.ptr_composite_element;
                ptr = payload.ptr;
                continue;
            }
            case BitCast_TAG: {
                if (allow_non_logical_ops) {
                    BitCast payload = ptr->payload.bit_cast;
                    ptr = payload.src;
                    continue;
                }
                break;
            }
            case GenericPtrCast_TAG: {
                if (allow_non_logical_ops) {
                    GenericPtrCast payload = ptr->payload.generic_ptr_cast;
                    ptr = payload.src;
                    continue;
                }
                break;
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
    return NULL;
}

bool shd_is_logical_memory_declaration(PtrAnalysis* ptr_analysis, const Node* ptr) {
    const AllocaInfo* k = shd_find_memory_declaration(ptr_analysis, ptr, false);
    if (!k) return false;
    return !k->non_logical_use;
}

static const AllocaInfo* create_memory_declaration(PtrAnalysis* ctx, const Node* old) {
    AllocaInfo* k = shd_arena_alloc(ctx->arena, sizeof(AllocaInfo));

    const Type* old_ptr_type = shd_get_unqualified_type(old->type);
    assert(old_ptr_type->tag == PtrType_TAG);
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

typedef struct {
    Visitor v;
    PtrAnalysis* a;
} PtrAnalysisVisitor;

static void analyze_maybe_alloc(PtrAnalysisVisitor* v, const Node* node) {
    switch (node->tag) {
        case LocalAlloc_TAG:
        case BuiltinRef_TAG:
        case GlobalVariable_TAG: create_memory_declaration(v->a, node);
        default: break;
    }
}

static void analyze_fn(PtrAnalysis* ptr_analysis, const Node* fn) {
    if (!get_abstraction_body(fn))
        return;
    PtrAnalysisVisitor v = {
        .v = {
            .visit_node_fn = (VisitNodeFn) analyze_maybe_alloc,
        },
        .a = ptr_analysis,
    };
    shd_visit_function_cfg_mem_rpo((Visitor*) &v, fn);
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
        create_memory_declaration(analysis, globals.nodes[i]);
    }
    Nodes fns = shd_module_collect_reachable_functions(module);
    for (size_t i = 0; i < fns.count; i++) {
        analyze_fn(analysis, fns.nodes[i]);
    }
    return analysis;
}

void shd_destroy_ptr_analysis(PtrAnalysis* analysis) {
    shd_destroy_dict(analysis->alloca_info);
    shd_destroy_arena(analysis->arena);
    free(analysis);
}