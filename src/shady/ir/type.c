#include "shady/ir/type.h"
#include "shady/ir/function.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/module.h"

#include "../ir_private.h"

#include "log.h"
#include "portability.h"
#include "util.h"

#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

static bool are_types_identical(size_t num_types, const Type* types[]) {
    for (size_t i = 0; i < num_types; i++) {
        assert(types[i]);
        if (types[0] != types[i])
            return false;
    }
    return true;
}

bool shd_is_subtype(const Type* supertype, const Type* type) {
    assert(supertype && type);
    if (supertype->tag != type->tag)
        return false;
    if (type == supertype)
        return true;
    switch (is_type(supertype)) {
        case NotAType: shd_error("supplied not a type to is_subtype");
        case QualifiedType_TAG: {
            // uniform T <: varying T
            if (supertype->payload.qualified_type.is_uniform && !type->payload.qualified_type.is_uniform)
                return false;
            return shd_is_subtype(supertype->payload.qualified_type.type, type->payload.qualified_type.type);
        }
        case RecordType_TAG: {
            const Nodes* supermembers = &supertype->payload.record_type.members;
            const Nodes* members = &type->payload.record_type.members;
            if (supermembers->count != members->count)
                return false;
            for (size_t i = 0; i < members->count; i++) {
                if (!shd_is_subtype(supermembers->nodes[i], members->nodes[i]))
                    return false;
            }
            return supertype->payload.record_type.special == type->payload.record_type.special;
        }
        case JoinPointType_TAG: {
            const Nodes* superparams = &supertype->payload.join_point_type.yield_types;
            const Nodes* params = &type->payload.join_point_type.yield_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!shd_is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        }
        case FnType_TAG: {
            // check returns
            if (supertype->payload.fn_type.return_types.count != type->payload.fn_type.return_types.count)
                return false;
            for (size_t i = 0; i < type->payload.fn_type.return_types.count; i++)
                if (!shd_is_subtype(supertype->payload.fn_type.return_types.nodes[i], type->payload.fn_type.return_types.nodes[i]))
                    return false;
            // check params
            const Nodes* superparams = &supertype->payload.fn_type.param_types;
            const Nodes* params = &type->payload.fn_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!shd_is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        } case BBType_TAG: {
            // check params
            const Nodes* superparams = &supertype->payload.bb_type.param_types;
            const Nodes* params = &type->payload.bb_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!shd_is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        } case LamType_TAG: {
            // check params
            const Nodes* superparams = &supertype->payload.lam_type.param_types;
            const Nodes* params = &type->payload.lam_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!shd_is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        } case PtrType_TAG: {
            if (supertype->payload.ptr_type.address_space != type->payload.ptr_type.address_space)
                return false;
            if (!supertype->payload.ptr_type.is_reference && type->payload.ptr_type.is_reference)
                return false;
            return shd_is_subtype(supertype->payload.ptr_type.pointed_type, type->payload.ptr_type.pointed_type);
        }
        case Int_TAG: return supertype->payload.int_type.width == type->payload.int_type.width && supertype->payload.int_type.is_signed == type->payload.int_type.is_signed;
        case ArrType_TAG: {
            if (!shd_is_subtype(supertype->payload.arr_type.element_type, type->payload.arr_type.element_type))
                return false;
            // unsized arrays are supertypes of sized arrays (even though they're not datatypes...)
            // TODO: maybe change this so it's only valid when talking about to pointer-to-arrays
            const IntLiteral* size_literal = shd_resolve_to_int_literal(supertype->payload.arr_type.size);
            if (size_literal && size_literal->value == 0)
                return true;
            return supertype->payload.arr_type.size == type->payload.arr_type.size || !supertype->payload.arr_type.size;
        }
        case PackType_TAG: {
            if (!shd_is_subtype(supertype->payload.pack_type.element_type, type->payload.pack_type.element_type))
                return false;
            return supertype->payload.pack_type.width == type->payload.pack_type.width;
        }
        case NominalType_TAG: {
            return false;
        }
        case Type_ImageType_TAG: {
            if (!shd_is_subtype(supertype->payload.image_type.sampled_type, type->payload.image_type.sampled_type))
                return false;
            if (supertype->payload.image_type.depth != type->payload.image_type.depth)
                return false;
            if (supertype->payload.image_type.dim != type->payload.image_type.dim)
                return false;
            if (supertype->payload.image_type.arrayed != type->payload.image_type.arrayed)
                return false;
            if (supertype->payload.image_type.ms != type->payload.image_type.ms)
                return false;
            if (supertype->payload.image_type.sampled != type->payload.image_type.sampled)
                return false;
            if (supertype->payload.image_type.imageformat != type->payload.image_type.imageformat)
                return false;
            return true;
        }
        case Type_SampledImageType_TAG:
            return shd_is_subtype(supertype->payload.sampled_image_type.image_type, type->payload.sampled_image_type.image_type);
        default: break;
    }
    // Two types are always equal (and therefore subtypes of each other) if their payload matches
    return memcmp(&supertype->payload, &type->payload, sizeof(type->payload)) == 0;
}

void shd_check_subtype(const Type* supertype, const Type* type) {
    if (!shd_is_subtype(supertype, type)) {
        shd_log_node(ERROR, type);
        shd_error_print(" isn't a subtype of ");
        shd_log_node(ERROR, supertype);
        shd_error_print("\n");
        shd_error("failed check_subtype")
    }
}

/// Is this a type that a value in the language can have ?
bool shd_is_value_type(const Type* type) {
    if (type->tag != QualifiedType_TAG)
        return false;
    return shd_is_data_type(shd_get_unqualified_type(type));
}

bool shd_is_physical_data_type(const Type* type) {
    switch (is_type(type)) {
        case Type_MaskType_TAG:
        case Type_JoinPointType_TAG:
        case Type_Int_TAG:
        case Type_Float_TAG:
        case Type_Bool_TAG:
            return true;
        case Type_PtrType_TAG:
            return !type->payload.ptr_type.is_reference;
        case Type_ArrType_TAG:
            // array types _must_ be sized to be real data types
            return type->payload.arr_type.size != NULL;
        case Type_PackType_TAG:
            return shd_is_data_type(type->payload.pack_type.element_type);
        case Type_RecordType_TAG: {
            for (size_t i = 0; i < type->payload.record_type.members.count; i++)
                if (!shd_is_data_type(type->payload.record_type.members.nodes[i]))
                    return false;
            // multi-return record types are the results of instructions, but are not values themselves
            return type->payload.record_type.special == NotSpecial;
        }
        case NominalType_TAG:
            return !shd_get_nominal_type_body(type) || shd_is_data_type(shd_get_nominal_type_body(type));
            // qualified types are not data types because that information is only meant for values
        case Type_QualifiedType_TAG: return false;
            // values cannot contain abstractions
        case Type_FnType_TAG:
        case Type_BBType_TAG:
        case Type_LamType_TAG:
            return false;
            // this type has no values to begin with
        case Type_NoRet_TAG:
            return false;
        case NotAType:
            return false;
        // Image stuff is data (albeit opaque)
        case Type_SampledImageType_TAG:
        case Type_SamplerType_TAG:
        case Type_ImageType_TAG:
            return false;
    }
}

/// Is this a valid data type (for usage in other types and as type arguments) ?
bool shd_is_data_type(const Type* type) {
    switch (is_type(type)) {
        case Type_PtrType_TAG:
            return true;
        // Image stuff is data (albeit opaque)
        case Type_SampledImageType_TAG:
        case Type_SamplerType_TAG:
        case Type_ImageType_TAG:
            return true;
        default: return shd_is_physical_data_type(type);
    }
}

bool shd_is_arithm_type(const Type* t) {
    return t->tag == Int_TAG || t->tag == Float_TAG;
}

bool shd_is_shiftable_type(const Type* t) {
    return t->tag == Int_TAG || t->tag == MaskType_TAG;
}

bool shd_has_boolean_ops(const Type* t) {
    return t->tag == Int_TAG || t->tag == Bool_TAG || t->tag == MaskType_TAG;
}

bool shd_is_comparable_type(const Type* t) {
    return true; // TODO this is fine to allow, but we'll need to lower it for composite and native ptr types !
}

bool shd_is_ordered_type(const Type* t) {
    return shd_is_arithm_type(t);
}

bool shd_is_physical_ptr_type(const Type* t) {
    if (t->tag != PtrType_TAG)
        return false;
    return !t->payload.ptr_type.is_reference;
    // AddressSpace as = t->payload.ptr_type.address_space;
    // return t->shd_get_arena_config(arena)->address_spaces[as].physical;
}

bool shd_is_generic_ptr_type(const Type* t) {
    if (t->tag != PtrType_TAG)
        return false;
    AddressSpace as = t->payload.ptr_type.address_space;
    return as == AsGeneric;
}

ShdScope shd_get_addr_space_scope(AddressSpace as) {
    switch (as) {
        case AsGeneric:
        case AsPrivate:
        case AsFunction:
        case AsInput:
        case AsOutput: return ShdScopeInvocation;
        case AsSubgroup: return ShdScopeSubgroup;
        case AsShared: return ShdScopeWorkgroup;
        case AsGlobal:
        case AsUInput: return ShdScopeDevice;
        default: return ShdScopeTop;
    }
}

bool shd_is_addr_space_uniform(IrArena* arena, AddressSpace as) {
    return shd_get_addr_space_scope(as) <= ShdScopeWorkgroup;
}

const Type* shd_get_actual_mask_type(IrArena* arena) {
    switch (shd_get_arena_config(arena)->target.subgroup_mask_representation) {
        case SubgroupMaskAbstract: return mask_type(arena);
        case SubgroupMaskInt64: return shd_uint64_type(arena);
        default: assert(false);
    }
}

String shd_get_type_name(IrArena* arena, const Type* t) {
    switch (is_type(t)) {
        case NotAType: assert(false);
        case Type_MaskType_TAG: return "mask_t";
        case Type_JoinPointType_TAG: return "join_type_t";
        case Type_NoRet_TAG: return "no_ret";
        case Type_Int_TAG: {
            if (t->payload.int_type.is_signed)
                return shd_fmt_string_irarena(arena, "i%s", ((String[]) { "8", "16", "32", "64" })[t->payload.int_type.width]);
            else
                return shd_fmt_string_irarena(arena, "u%s", ((String[]) { "8", "16", "32", "64" })[t->payload.int_type.width]);
        }
        case Type_Float_TAG: return shd_fmt_string_irarena(arena, "f%s", ((String[]) { "16", "32", "64" })[t->payload.float_type.width]);
        case Type_Bool_TAG: return "bool";
        case NominalType_TAG: return t->payload.nom_type.name;
        default: break;
    }
    return shd_make_unique_name(arena, shd_get_node_tag_string(t->tag));
}

const Type* shd_maybe_multiple_return(IrArena* arena, Nodes types) {
    switch (types.count) {
        case 0: return empty_multiple_return_type(arena);
        case 1: return types.nodes[0];
        default: return record_type(arena, (RecordType) {
                .members = types,
                .names = shd_strings(arena, 0, NULL),
                .special = MultipleReturn,
            });
    }
    SHADY_UNREACHABLE;
}

Nodes shd_unwrap_multiple_yield_types(IrArena* arena, const Type* type) {
    switch (type->tag) {
        case RecordType_TAG:
            if (type->payload.record_type.special == MultipleReturn)
                return type->payload.record_type.members;
            // fallthrough
        default:
            assert(shd_is_value_type(type));
            return shd_singleton(type);
    }
}

const Type* shd_get_pointee_type(IrArena* arena, const Type* type) {
    bool qualified = false, uniform = false;
    if (shd_is_value_type(type)) {
        qualified = true;
        uniform = shd_is_qualified_type_uniform(type);
        type = shd_get_unqualified_type(type);
    }
    assert(type->tag == PtrType_TAG);
    uniform &= shd_is_addr_space_uniform(arena, type->payload.ptr_type.address_space);
    type = type->payload.ptr_type.pointed_type;

    if (qualified)
        type = qualified_type(arena, (QualifiedType) {
            .type = type,
            .is_uniform = uniform
        });
    return type;
}

Nodes shd_get_param_types(IrArena* arena, Nodes variables) {
    LARRAY(const Type*, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        assert(variables.nodes[i]->tag == Param_TAG);
        arr[i] = variables.nodes[i]->payload.param.type;
    }
    return shd_nodes(arena, variables.count, arr);
}

Nodes shd_get_values_types(IrArena* arena, Nodes values) {
    assert(shd_get_arena_config(arena)->check_types);
    LARRAY(const Type*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = values.nodes[i]->type;
    return shd_nodes(arena, values.count, arr);
}

bool shd_is_qualified_type_uniform(const Type* type) {
    const Type* result_type = type;
    bool is_uniform = shd_deconstruct_qualified_type(&result_type);
    return is_uniform;
}

const Type* shd_get_unqualified_type(const Type* type) {
    assert(is_type(type));
    const Type* result_type = type;
    shd_deconstruct_qualified_type(&result_type);
    return result_type;
}

bool shd_deconstruct_qualified_type(const Type** type_out) {
    const Type* type = *type_out;
    if (type->tag == QualifiedType_TAG) {
        *type_out = type->payload.qualified_type.type;
        return type->payload.qualified_type.is_uniform;
    } else shd_error("Expected a value type (annotated with qual_type)")
}

const Type* shd_as_qualified_type(const Type* type, bool uniform) {
    return qualified_type(type->arena, (QualifiedType) { .type = type, .is_uniform = uniform });
}

Nodes shd_strip_qualifiers(IrArena* arena, Nodes tys) {
    LARRAY(const Type*, arr, tys.count);
    for (size_t i = 0; i < tys.count; i++)
        arr[i] = shd_get_unqualified_type(tys.nodes[i]);
    return shd_nodes(arena, tys.count, arr);
}

Nodes shd_add_qualifiers(IrArena* arena, Nodes tys, bool uniform) {
    LARRAY(const Type*, arr, tys.count);
    for (size_t i = 0; i < tys.count; i++)
        arr[i] = shd_as_qualified_type(tys.nodes[i],
                                       uniform || !shd_get_arena_config(arena)->is_simt /* SIMD arenas ban varying value types */);
    return shd_nodes(arena, tys.count, arr);
}

const Type* shd_get_packed_type_element(const Type* type) {
    const Type* t = type;
    shd_deconstruct_packed_type(&t);
    return t;
}

size_t shd_get_packed_type_width(const Type* type) {
    const Type* t = type;
    return shd_deconstruct_packed_type(&t);
}

size_t shd_deconstruct_packed_type(const Type** type) {
    assert((*type)->tag == PackType_TAG);
    return shd_deconstruct_maybe_packed_type(type);
}

const Type* shd_get_maybe_packed_type_element(const Type* type) {
    const Type* t = type;
    shd_deconstruct_maybe_packed_type(&t);
    return t;
}

size_t shd_get_maybe_packed_type_width(const Type* type) {
    const Type* t = type;
    return shd_deconstruct_maybe_packed_type(&t);
}

size_t shd_deconstruct_maybe_packed_type(const Type** type) {
    const Type* t = *type;
    assert(shd_is_data_type(t));
    if (t->tag == PackType_TAG) {
        *type = t->payload.pack_type.element_type;
        return t->payload.pack_type.width;
    }
    return 1;
}

const Type* shd_maybe_packed_type_helper(const Type* type, size_t width) {
    assert(width > 0);
    if (width == 1)
        return type;
    return pack_type(type->arena, (PackType) {
        .width = width,
        .element_type = type,
    });
}

const Type* shd_get_pointer_type_element(const Type* type) {
    const Type* t = type;
    shd_deconstruct_pointer_type(&t);
    return t;
}

AddressSpace shd_deconstruct_pointer_type(const Type** type) {
    const Type* t = *type;
    assert(t->tag == PtrType_TAG);
    *type = t->payload.ptr_type.pointed_type;
    return t->payload.ptr_type.address_space;
}

const Type* shd_get_nominal_type_body(const Type* type) {
    assert(type->tag == NominalType_TAG);
    return type->payload.nom_type.body;
}

const Type* shd_get_maybe_nominal_type_body(const Type* type) {
    if (type->tag == NominalType_TAG)
        return type->payload.nom_type.body;
    return type;
}

Nodes shd_get_composite_type_element_types(const Type* type) {
    switch (is_type(type)) {
        case NominalType_TAG: {
            type = shd_get_nominal_type_body(type);
            assert(type->tag == RecordType_TAG);
            SHADY_FALLTHROUGH
        }
        case RecordType_TAG: {
            return type->payload.record_type.members;
        }
        case Type_ArrType_TAG:
        case Type_PackType_TAG: {
            size_t size = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_fill_type_size(type)), false);
            if (size >= 1024) {
                shd_warn_print("Potential performance issue: creating a really big array of composites of types (size=%d)!\n", size);
            }
            const Type* element_type = shd_get_fill_type_element_type(type);
            LARRAY(const Type*, types, size);
            for (size_t i = 0; i < size; i++) {
                types[i] = element_type;
            }
            return shd_nodes(type->arena, size, types);
        }
        default: shd_error("Not a composite type !")
    }
}

const Node* shd_get_fill_type_element_type(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.element_type;
        case PackType_TAG: return composite_t->payload.pack_type.element_type;
        default: shd_error("fill values need to be either array or pack types")
    }
}

const Node* shd_get_fill_type_size(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.size;
        case PackType_TAG: return shd_int32_literal(composite_t->arena, composite_t->payload.pack_type.width);
        default: shd_error("fill values need to be either array or pack types")
    }
}

Type* _shd_nominal_type(Module* mod, Nodes annotations, String name) {
    IrArena* arena = shd_module_get_arena(mod);
    NominalType payload = {
        .name = shd_string(arena, name),
        .annotations = annotations,
        .body = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = NULL,
        .tag = NominalType_TAG,
        .payload.nom_type = payload
    };
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    _shd_module_add_decl(mod, decl);
    return decl;
}

const Node* shd_get_default_value(IrArena* a, const Type* t) {
    switch (is_type(t)) {
        case NotAType: shd_error("")
        case Type_Int_TAG: return int_literal(a, (IntLiteral) { .width = t->payload.int_type.width, .is_signed = t->payload.int_type.is_signed, .value = 0 });
        case Type_Float_TAG: return float_literal(a, (FloatLiteral) { .width = t->payload.float_type.width, .value = 0 });
        case Type_Bool_TAG: return false_lit(a);
        case Type_PtrType_TAG: return null_ptr(a, (NullPtr) { .ptr_type = t });
        case Type_QualifiedType_TAG: return shd_get_default_value(a, t->payload.qualified_type.type);
        case Type_RecordType_TAG:
        case Type_ArrType_TAG:
        case Type_PackType_TAG:
        case NominalType_TAG: {
            Nodes elem_tys = shd_get_composite_type_element_types(t);
            if (elem_tys.count >= 1024) {
                shd_warn_print("Potential performance issue: creating a really composite full of zero/default values (size=%d)!\n", elem_tys.count);
            }
            LARRAY(const Node*, elems, elem_tys.count);
            for (size_t i = 0; i < elem_tys.count; i++)
                elems[i] = shd_get_default_value(a, elem_tys.nodes[i]);
            return composite_helper(a, t, shd_nodes(a, elem_tys.count, elems));
        }
        default: break;
    }
    return NULL;
}
