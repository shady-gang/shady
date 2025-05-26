#include "shady/analysis/literal.h"

#include "shady/ir/int.h"
#include "shady/ir/type.h"

#include "portability.h"

#include <assert.h>

static bool is_zero(const Node* node) {
    const IntLiteral* lit = shd_resolve_to_int_literal(node);
    if (lit && shd_get_int_literal_value(*lit, false) == 0)
        return true;
    return false;
}

const Node* shd_chase_ptr_to_source(const Node* ptr, NodeResolveConfig config) {
    while (true) {
        ptr = shd_resolve_node_to_definition(ptr, config);
        switch (ptr->tag) {
            case PtrArrayElementOffset_TAG: break;
            case PtrCompositeElement_TAG: {
                PtrCompositeElement payload = ptr->payload.ptr_composite_element;
                if (!is_zero(payload.index))
                    break;
                ptr = payload.ptr;
                continue;
            }
            case BitCast_TAG: {
                // chase ptr casts to their source
                        // TODO: figure out round-trips through integer casts?
                if (ptr->payload.bit_cast.type->tag == PtrType_TAG) {
                    ptr = shd_first(ptr->payload.prim_op.operands);
                    continue;
                }
                break;
            }
            case Conversion_TAG: {
                Conversion payload = ptr->payload.conversion;
                // chase generic pointers to their source
                if (payload.type->tag == PtrType_TAG) {
                    ptr = payload.src;
                    continue;
                }
                break;
            }
            default: break;
        }
        break;
    }
    return ptr;
}

const Node* shd_resolve_ptr_to_value(const Node* ptr, NodeResolveConfig config) {
    while (ptr) {
        ptr = shd_resolve_node_to_definition(ptr, config);
        switch (ptr->tag) {
            case GenericPtrCast_TAG: {
                // allow address space conversions
                ptr = ptr->payload.generic_ptr_cast.src;
                continue;
            }
            case GlobalVariable_TAG:
                if (config.assume_globals_immutability)
                    return ptr->payload.global_variable.init;
                break;
            default: break;
        }
        ptr = NULL;
    }
    return NULL;
}

NodeResolveConfig shd_default_node_resolve_config(void) {
    return (NodeResolveConfig) {
        .enter_loads = true,
        .allow_incompatible_types = false,
        .assume_globals_immutability = false,
    };
}

const Node* shd_resolve_node_to_definition(const Node* node, NodeResolveConfig config) {
    while (node) {
        switch (node->tag) {
            case Constant_TAG:
                node = node->payload.constant.value;
                continue;
            case Load_TAG: {
                if (config.enter_loads) {
                    const Node* source = node->payload.load.ptr;
                    const Node* result = shd_resolve_ptr_to_value(source, config);
                    if (!result)
                        break;
                    node = result;
                    continue;
                }
                break;
            }
            case BitCast_TAG: {
                BitCast payload = node->payload.bit_cast;
                if (config.allow_incompatible_types) {
                    node = payload.src;
                    continue;
                }
                break;
            }
            case Conversion_TAG: {
                Conversion payload = node->payload.conversion;
                if (config.allow_incompatible_types) {
                    node = payload.src;
                    continue;
                }
                break;
            }
            case GenericPtrCast_TAG: {
                GenericPtrCast payload = node->payload.generic_ptr_cast;
                if (config.allow_incompatible_types) {
                    node = payload.src;
                    continue;
                }
                break;
            }
            default: break;
        }
        break;
    }
    return node;
}

const char* shd_get_string_literal(IrArena* arena, const Node* node) {
    if (!node)
        return NULL;
    if (node->type && shd_get_unqualified_type(node->type)->tag == PtrType_TAG) {
        NodeResolveConfig nrc = shd_default_node_resolve_config();
        const Node* ptr = shd_chase_ptr_to_source(node, nrc);
        const Node* value = shd_resolve_ptr_to_value(ptr, nrc);
        if (value)
            return shd_get_string_literal(arena, value);
    }
    switch (node->tag) {
        case GlobalVariable_TAG: {
            const Node* init = node->payload.global_variable.init;
            if (init) {
                return shd_get_string_literal(arena, init);
            }
            break;
        }
        case Constant_TAG: {
            return shd_get_string_literal(arena, node->payload.constant.value);
        }
        /*case Lea_TAG: {
            Lea lea = node->payload.lea;
            if (lea.indices.count == 3 && is_zero(lea.offset) && is_zero(first(lea.indices))) {
                const Node* ref = lea.ptr;
                if (ref->tag != RefDecl_TAG)
                    return NULL;
                const Node* decl = ref->payload.ref_decl.decl;
                if (decl->tag != GlobalVariable_TAG || !decl->payload.global_variable.init)
                    return NULL;
                return get_string_literal(arena, decl->payload.global_variable.init);
            }
            break;
        }*/
        case StringLiteral_TAG: return node->payload.string_lit.string;
        case Composite_TAG: {
            Nodes contents = node->payload.composite.contents;
            LARRAY(char, chars, contents.count);
            for (size_t i = 0; i < contents.count; i++) {
                const Node* value = contents.nodes[i];
                assert(value->tag == IntLiteral_TAG && value->payload.int_literal.width == IntTy8);
                chars[i] = (unsigned char) shd_get_int_literal_value(*shd_resolve_to_int_literal(value), false);
            }
            assert(chars[contents.count - 1] == 0);
            return shd_string(arena, chars);
        }
        default: break;
    }
    return NULL;
}
