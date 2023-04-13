#include "type.h"

#include "log.h"
#include "ir_private.h"
#include "portability.h"

#include "dict.h"

#include <string.h>
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

bool is_subtype(const Type* supertype, const Type* type) {
    assert(supertype && type);
    if (supertype->tag != type->tag)
        return false;
    switch (is_type(supertype)) {
        case NotAType: error("supplied not a type to is_subtype");
        case QualifiedType_TAG: {
            // uniform T <: varying T
            if (supertype->payload.qualified_type.is_uniform && !type->payload.qualified_type.is_uniform)
                return false;
            return is_subtype(supertype->payload.qualified_type.type, type->payload.qualified_type.type);
        }
        case RecordType_TAG: {
            const Nodes* supermembers = &supertype->payload.record_type.members;
            const Nodes* members = &type->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                if (!is_subtype(supermembers->nodes[i], members->nodes[i]))
                    return false;
            }
            return true;
        }
        case JoinPointType_TAG: {
            const Nodes* superparams = &supertype->payload.join_point_type.yield_types;
            const Nodes* params = &type->payload.join_point_type.yield_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        }
        case FnType_TAG: {
            // check returns
            if (supertype->payload.fn_type.return_types.count != type->payload.fn_type.return_types.count)
                return false;
            for (size_t i = 0; i < type->payload.fn_type.return_types.count; i++)
                if (!is_subtype(supertype->payload.fn_type.return_types.nodes[i], type->payload.fn_type.return_types.nodes[i]))
                    return false;
            // check params
            const Nodes* superparams = &supertype->payload.fn_type.param_types;
            const Nodes* params = &type->payload.fn_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        } case BBType_TAG: {
            // check params
            const Nodes* superparams = &supertype->payload.bb_type.param_types;
            const Nodes* params = &type->payload.bb_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        } case LamType_TAG: {
            // check params
            const Nodes* superparams = &supertype->payload.lam_type.param_types;
            const Nodes* params = &type->payload.lam_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        } case PtrType_TAG: {
            if (supertype->payload.ptr_type.address_space != type->payload.ptr_type.address_space)
                return false;
            return is_subtype(supertype->payload.ptr_type.pointed_type, type->payload.ptr_type.pointed_type);
        }
        case Int_TAG: return supertype->payload.int_type.width == type->payload.int_type.width && supertype->payload.int_type.is_signed == type->payload.int_type.is_signed;
        case ArrType_TAG: {
            if (!is_subtype(supertype->payload.arr_type.element_type, type->payload.arr_type.element_type))
                return false;
            return supertype->payload.arr_type.size == type->payload.arr_type.size || !supertype->payload.arr_type.size || get_int_literal_value(supertype->payload.arr_type.size, false) == 0;
        }
        case PackType_TAG: {
            if (!is_subtype(supertype->payload.pack_type.element_type, type->payload.pack_type.element_type))
                return false;
            return supertype->payload.pack_type.width == type->payload.pack_type.width;
        }
        case Type_TypeDeclRef_TAG: {
            return supertype->payload.type_decl_ref.decl == type->payload.type_decl_ref.decl;
        }
        case NoRet_TAG:
        case Bool_TAG:
        case MaskType_TAG:
        case Float_TAG:
            return supertype->payload.float_type.width == type->payload.float_type.width;
    }
    SHADY_UNREACHABLE;
}

void check_subtype(const Type* supertype, const Type* type) {
    if (!is_subtype(supertype, type)) {
        log_node(ERROR, type);
        error_print(" isn't a subtype of ");
        log_node(ERROR, supertype);
        error_print("\n");
        error("failed check_subtype")
    }
}

size_t get_type_bitwidth(const Type* t) {
    switch (t->tag) {
        case Int_TAG: {
            switch (t->payload.int_type.width) {
                case IntTy8:  return 8;
                case IntTy16: return 16;
                case IntTy32: return 32;
                case IntTy64: return 64;
            }
        }
        case Float_TAG: {
            switch (t->payload.float_type.width) {
                case FloatTy16: return 16;
                case FloatTy32: return 32;
                case FloatTy64: return 64;
            }
        }
        case PtrType_TAG: {
            if (is_physical_as(t->payload.ptr_type.address_space))
                return 64;
        }
        default: break;
    }
    return SIZE_MAX;
}

bool is_addr_space_uniform(IrArena* arena, AddressSpace as) {
    switch (as) {
        case AsSPVFunctionLogical:
        case AsPrivateLogical:
        case AsPrivatePhysical:
        case AsInput:
            return !arena->config.is_simt;
        default:
            return true;
    }
}

const Type* get_actual_mask_type(IrArena* arena) {
    switch (arena->config.specializations.subgroup_mask_representation) {
        case SubgroupMaskAbstract: return mask_type(arena);
        case SubgroupMaskInt64: return uint64_type(arena);
        default: assert(false);
    }
}

String name_type_safe(IrArena* arena, const Type* t) {
    switch (is_type(t)) {
        case NotAType: assert(false);
        case Type_MaskType_TAG: return "mask_t";
        case Type_JoinPointType_TAG: return "join_type_t";
        case Type_NoRet_TAG: return "no_ret";
        case Type_Int_TAG:
            if (t->payload.int_type.is_signed)
                return format_string(arena, "i%s", ((String[]) { "8", "16", "32", "64" })[t->payload.int_type.width]);
            else
                return format_string(arena, "u%s", ((String[]) { "8", "16", "32", "64" })[t->payload.int_type.width]);
        case Type_Float_TAG:
            return format_string(arena, "f%s", ((String[]) { "16", "32", "64" })[t->payload.float_type.width]);
        case Type_Bool_TAG: return "bool";
        case Type_RecordType_TAG: break;
        case Type_FnType_TAG: break;
        case Type_BBType_TAG: break;
        case Type_LamType_TAG: break;
        case Type_PtrType_TAG: break;
        case Type_QualifiedType_TAG: break;
        case Type_ArrType_TAG: break;
        case Type_PackType_TAG: break;
        case Type_TypeDeclRef_TAG: return t->payload.type_decl_ref.decl->payload.nom_type.name;
    }
    return unique_name(arena, node_tags[t->tag]);
}

/// Is this a type that a value in the language can have ?
bool is_value_type(const Type* type) {
    //if (type->tag == RecordType_TAG && type->payload.record_type.special == MultipleReturn)
    //    return true;
    if (type->tag != QualifiedType_TAG)
        return false;
    return is_data_type(get_unqualified_type(type));
}

/// Is this a valid data type (for usage in other types and as type arguments) ?
bool is_data_type(const Type* type) {
    switch (is_type(type)) {
        case Type_MaskType_TAG:
        case Type_JoinPointType_TAG:
        case Type_Int_TAG:
        case Type_Float_TAG:
        case Type_Bool_TAG:
            return true;
        case Type_PtrType_TAG:
            return true;
        case Type_ArrType_TAG:
            // array types _must_ be sized to be real data types
            return type->payload.arr_type.size != NULL;
        case Type_PackType_TAG:
            return is_data_type(type->payload.pack_type.element_type);
        case Type_RecordType_TAG: {
            for (size_t i = 0; i < type->payload.record_type.members.count; i++)
                if (!is_data_type(type->payload.record_type.members.nodes[i]))
                    return false;
            // multi-return record types are the results of instructions, but are not values themselves
            return type->payload.record_type.special == NotSpecial;
        }
        case Type_TypeDeclRef_TAG:
            return !get_nominal_type_body(type) || is_data_type(get_nominal_type_body(type));
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
    }
}

bool is_arithm_type(const Type* t) {
    return t->tag == Int_TAG || t->tag == Float_TAG;
}

bool is_shiftable_type(const Type* t) {
    return t->tag == Int_TAG || t->tag == MaskType_TAG;
}

bool has_boolean_ops(const Type* t) {
    return t->tag == Int_TAG || t->tag == Bool_TAG || t->tag == MaskType_TAG;
}

bool is_comparable_type(const Type* t) {
    return true; // TODO this is fine to allow, but we'll need to lower it for composite and native ptr types !
}

bool is_ordered_type(const Type* t) {
    return is_arithm_type(t);
}

static bool is_transparent_pointer_type(const Type* t) {
    if (t->tag != PtrType_TAG)
        return false;
    AddressSpace as = t->payload.ptr_type.address_space;
    // it's illegal to reinterpret from/into a Generic pointer because they have a tag
    return is_physical_as(as);
}

/// Oracle of what casts are legal
static bool is_reinterpret_cast_legal(const Type* src_type, const Type* dst_type) {
    assert(is_data_type(src_type) && is_data_type(dst_type));
    assert(get_type_bitwidth(src_type) == get_type_bitwidth(dst_type));
    if (!(is_arithm_type(src_type) || src_type->tag == MaskType_TAG || is_transparent_pointer_type(src_type)))
        return false;
    if (!(is_arithm_type(dst_type) || dst_type->tag == MaskType_TAG || is_transparent_pointer_type(dst_type)))
        return false;
    if (is_transparent_pointer_type(src_type) && is_transparent_pointer_type(dst_type)) {
        AddressSpace src_as = src_type->payload.ptr_type.address_space;
        AddressSpace dst_as = dst_type->payload.ptr_type.address_space;
        // either both pointers need to be in the generic address space, and we're only casting the element type, OR neither can be
        if ((src_as == AsGeneric) != (dst_as == AsGeneric))
            return false;
    }
    return true;
}

/// Oracle of what casts are legal
static bool is_conversion_legal(const Type* src_type, const Type* dst_type) {
    assert(is_data_type(src_type) && is_data_type(dst_type));
    if (!(is_arithm_type(src_type) || is_transparent_pointer_type(src_type) && get_type_bitwidth(src_type) == get_type_bitwidth(dst_type)))
        return false;
    if (!(is_arithm_type(dst_type) || is_transparent_pointer_type(dst_type) && get_type_bitwidth(src_type) == get_type_bitwidth(dst_type)))
        return false;
    if (is_transparent_pointer_type(src_type) && is_transparent_pointer_type(dst_type)) {
        AddressSpace src_as = src_type->payload.ptr_type.address_space;
        AddressSpace dst_as = dst_type->payload.ptr_type.address_space;
        // exactly one of the pointers needs to be in the generic address space
        if ((src_as == AsGeneric) == (dst_as == AsGeneric))
            return false;
        // element types have to match (use reinterpret_cast for changing it)
        if (src_type->payload.ptr_type.pointed_type != dst_type->payload.ptr_type.pointed_type)
            return false;
    }
    return true;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

const Type* check_type_join_point_type(IrArena* arena, JoinPointType type) {
    for (size_t i = 0; i < type.yield_types.count; i++) {
        assert(is_data_type(type.yield_types.nodes[i]));
    }
    return NULL;
}

const Type* check_type_record_type(IrArena* arena, RecordType type) {
    assert(type.names.count == 0 || type.names.count == type.members.count);
    for (size_t i = 0; i < type.members.count; i++) {
        // member types are value types iff this is a return tuple
        assert((type.special == MultipleReturn) == is_value_type(type.members.nodes[i]));
    }
    return NULL;
}

const Type* check_type_qualified_type(IrArena* arena, QualifiedType qualified_type) {
    assert(is_data_type(qualified_type.type));
    assert(arena->config.is_simt || qualified_type.is_uniform);
    return NULL;
}

const Type* check_type_arr_type(IrArena* arena, ArrType type) {
    assert(is_data_type(type.element_type));
    return NULL;
}

const Type* check_type_pack_type(IrArena* arena, PackType pack_type) {
    assert(is_data_type(pack_type.element_type));
    return NULL;
}

const Type* check_type_var(IrArena* arena, Variable variable) {
    assert(is_value_type(variable.type));
    return variable.type;
}

const Type* check_type_untyped_number(IrArena* arena, UntypedNumber untyped) {
    error("should never happen");
}

const Type* check_type_int_literal(IrArena* arena, IntLiteral lit) {
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = int_type(arena, (Int) { .width = lit.width, .is_signed = lit.is_signed })
    });
}

const Type* check_type_float_literal(IrArena* arena, FloatLiteral lit) {
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = float_type(arena, (Float) { .width = lit.width })
    });
}

const Type* check_type_true_lit(IrArena* arena) { return qualified_type(arena, (QualifiedType) { .type = bool_type(arena), .is_uniform = true }); }
const Type* check_type_false_lit(IrArena* arena) { return qualified_type(arena, (QualifiedType) { .type = bool_type(arena), .is_uniform = true }); }

const Type* check_type_string_lit(IrArena* arena, StringLiteral str_lit) {
    const Type* t = arr_type(arena, (ArrType) {
        .element_type = int8_type(arena),
        .size = int32_literal(arena, strlen(str_lit.string))
    });
    return qualified_type(arena, (QualifiedType) {
        .type = t,
        .is_uniform = true,
    });
}

const Type* check_type_composite(IrArena* arena, Composite composite) {
    assert(is_data_type(composite.type));
    Nodes expected_member_types = get_composite_type_element_types(composite.type);
    bool is_uniform = true;
    assert(composite.contents.count == expected_member_types.count);
    for (size_t i = 0; i < composite.contents.count; i++) {
        const Type* element_type = composite.contents.nodes[i]->type;
        is_uniform &= deconstruct_qualified_type(&element_type);
        assert(is_subtype(expected_member_types.nodes[i], element_type));
    }
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = is_uniform,
        .type = composite.type
    });
}

const Type* check_type_fill(IrArena* arena, Fill payload) {
    assert(is_data_type(payload.type));
    const Node* element_t = get_fill_type_element_type(payload.type);
    const Node* value_t = payload.value->type;
    bool u = deconstruct_qualified_type(&value_t);
    assert(is_subtype(element_t, value_t));
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = u,
        .type = payload.type
    });
}

const Type* check_type_undef(IrArena* arena, Undef payload) {
    assert(is_data_type(payload.type));
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = payload.type
    });
}

const Type* check_type_fn_addr(IrArena* arena, FnAddr fn_addr) {
    assert(fn_addr.fn->type->tag == FnType_TAG);
    assert(fn_addr.fn->tag == Function_TAG);
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = ptr_type(arena, (PtrType) {
            .pointed_type = fn_addr.fn->type,
            .address_space = AsProgramCode,
        })
    });
}

const Type* check_type_ref_decl(IrArena* arena, RefDecl ref_decl) {
    const Type* t = ref_decl.decl->type;
    assert(t && "RefDecl needs to be applied on a decl with a non-null type. Did you forget to set 'type' on a constant ?");
    switch (ref_decl.decl->tag) {
        case GlobalVariable_TAG:
        case Constant_TAG: break;
        default: error("You can only use RefDecl on a global or a constant. See FnAddr for taking addresses of functions.")
    }
    assert(t->tag != QualifiedType_TAG && "decl types may not be qualified");
    return qualified_type(arena, (QualifiedType) {
        .type = t,
        .is_uniform = true,
    });
}

const Type* check_type_anti_quote(IrArena* arena, AntiQuote payload) {
    assert(is_instruction(payload.instruction));
    return payload.instruction->type;
}

const Type* check_type_prim_op(IrArena* arena, PrimOp prim_op) {
    for (size_t i = 0; i < prim_op.type_arguments.count; i++) {
        const Node* ta = prim_op.type_arguments.nodes[i];
        assert(ta && is_type(ta));
    }
    for (size_t i = 0; i < prim_op.operands.count; i++) {
        const Node* operand = prim_op.operands.nodes[i];
        assert(operand && is_value(operand));
    }

    bool extended = false;
    bool ordered = false;
    AddressSpace as;
    switch (prim_op.op) {
        case deref_op:
        case assign_op:
        case subscript_op: error("These ops are only allowed in untyped IR before desugaring. They don't type to anything.");
        case quote_op: {
            assert(prim_op.type_arguments.count == 0);
            return wrap_multiple_yield_types(arena, get_values_types(arena, prim_op.operands));
        }
        case neg_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);

            const Type* type = first(prim_op.operands)->type;
            assert(is_arithm_type(get_maybe_packed_type_element(get_unqualified_type(type))));
            return type;
        }
        case rshift_arithm_op:
        case rshift_logical_op:
        case lshift_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = first(prim_op.operands)->type;
            const Type* second_operand_type = prim_op.operands.nodes[1]->type;

            bool uniform_result = deconstruct_qualified_type(&first_operand_type);
            uniform_result &= deconstruct_qualified_type(&second_operand_type);

            size_t value_simd_width = deconstruct_maybe_packed_type(&first_operand_type);
            size_t shift_simd_width = deconstruct_maybe_packed_type(&second_operand_type);
            assert(value_simd_width == shift_simd_width);

            assert(first_operand_type->tag == Int_TAG);
            assert(second_operand_type->tag == Int_TAG);

            return qualified_type_helper(maybe_packed_type_helper(first_operand_type, value_simd_width), uniform_result);
        }
        case add_carry_op:
        case sub_borrow_op:
        case mul_extended_op: extended = true; SHADY_FALLTHROUGH;
        case add_op:
        case sub_op:
        case mul_op:
        case div_op:
        case mod_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = get_unqualified_type(first(prim_op.operands)->type);

            bool result_uniform = true;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                bool operand_uniform = deconstruct_qualified_type(&operand_type);

                assert(is_arithm_type(get_maybe_packed_type_element(operand_type)));
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_uniform &= operand_uniform;
            }

            const Type* result_t = first_operand_type;
            if (extended) {
                // TODO: assert unsigned
                result_t = record_type(arena, (RecordType) {.members = mk_nodes(arena, result_t, result_t)});
            }
            return qualified_type_helper(result_t, result_uniform);
        }

        case not_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);

            const Type* type = first(prim_op.operands)->type;
            assert(has_boolean_ops(get_maybe_packed_type_element(get_unqualified_type(type))));
            return type;
        }
        case or_op:
        case xor_op:
        case and_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = get_unqualified_type(first(prim_op.operands)->type);

            bool result_uniform = true;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                bool operand_uniform = deconstruct_qualified_type(&operand_type);

                assert(has_boolean_ops(get_maybe_packed_type_element(operand_type)));
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_uniform &= operand_uniform;
            }

            return qualified_type_helper(first_operand_type, result_uniform);
        }
        case lt_op:
        case lte_op:
        case gt_op:
        case gte_op: ordered = true; SHADY_FALLTHROUGH
        case eq_op:
        case neq_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = get_unqualified_type(first(prim_op.operands)->type);
            size_t first_operand_width = get_maybe_packed_type_width(first_operand_type);

            bool result_uniform = true;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                bool operand_uniform = deconstruct_qualified_type(&operand_type);

                assert((ordered ? is_ordered_type : is_comparable_type)(get_maybe_packed_type_element(operand_type)));
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_uniform &= operand_uniform;
            }

            return qualified_type_helper(maybe_packed_type_helper(bool_type(arena), first_operand_width), result_uniform);
        }
        case sqrt_op:
        case inv_sqrt_op:
        case floor_op:
        case ceil_op:
        case round_op:
        case fract_op:
        case sin_op:
        case cos_op:
        {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);
            const Node* src_type = first(prim_op.operands)->type;
            bool uniform = deconstruct_qualified_type(&src_type);
            size_t width = deconstruct_maybe_packed_type(&src_type);
            assert(src_type->tag == Float_TAG);
            return qualified_type_helper(maybe_packed_type_helper(src_type, width), uniform);
        }
        case abs_op:
        case sign_op:
        {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);
            const Node* src_type = first(prim_op.operands)->type;
            bool uniform = deconstruct_qualified_type(&src_type);
            size_t width = deconstruct_maybe_packed_type(&src_type);
            assert(src_type->tag == Float_TAG || src_type->tag == Int_TAG && src_type->payload.int_type.is_signed);
            return qualified_type_helper(maybe_packed_type_helper(src_type, width), uniform);
        }
        case load_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);

            const Node* ptr = first(prim_op.operands);
            const Node* ptr_type = ptr->type;
            bool ptr_uniform = deconstruct_qualified_type(&ptr_type);
            size_t width = deconstruct_maybe_packed_type(&ptr_type);

            assert(ptr_type->tag == PtrType_TAG);
            const PtrType* node_ptr_type_ = &ptr_type->payload.ptr_type;
            const Type* elem_type = node_ptr_type_->pointed_type;
            elem_type = maybe_packed_type_helper(elem_type, width);
            return qualified_type_helper(elem_type, ptr_uniform && is_addr_space_uniform(arena, ptr_type->payload.ptr_type.address_space));
        }
        case store_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 2);

            const Node* ptr = first(prim_op.operands);
            const Node* ptr_type = ptr->type;
            bool ptr_uniform = deconstruct_qualified_type(&ptr_type);
            size_t width = deconstruct_maybe_packed_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            const PtrType* ptr_type_payload = &ptr_type->payload.ptr_type;
            const Type* elem_type = ptr_type_payload->pointed_type;
            elem_type = maybe_packed_type_helper(elem_type, width);
            // we don't enforce uniform stores - but we care about storing the right thing :)
            const Type* val_expected_type = qualified_type(arena, (QualifiedType) {
                .is_uniform = !arena->config.is_simt,
                .type = elem_type
            });

            const Node* val = prim_op.operands.nodes[1];
            assert(is_subtype(val_expected_type, val->type));
            return unit_type(arena);
        }
        case alloca_logical_op:  as = AsSPVFunctionLogical; goto alloca_case;
        case alloca_subgroup_op: as = AsSubgroupPhysical; goto alloca_case;
        case alloca_op:          as = AsPrivatePhysical; goto alloca_case;
        alloca_case: {
            assert(prim_op.type_arguments.count == 1);
            assert(prim_op.operands.count == 0);
            const Type* elem_type = prim_op.type_arguments.nodes[0];
            assert(is_type(elem_type));
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = is_addr_space_uniform(arena, as),
                .type = ptr_type(arena, (PtrType) {
                    .pointed_type = elem_type,
                    .address_space = as,
                })
            });
        }
        case lea_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count >= 2);

            const Node* base = prim_op.operands.nodes[0];
            bool uniform = is_qualified_type_uniform(base->type);

            const Type* curr_ptr_type = get_unqualified_type(base->type);
            assert(curr_ptr_type->tag == PtrType_TAG && "lea expects a pointer as a base");

            const Node* offset = prim_op.operands.nodes[1];
            assert(offset);
            const Type* offset_type = offset->type;
            bool offset_uniform = deconstruct_qualified_type(&offset_type);
            assert(offset_type->tag == Int_TAG && "lea expects an integer offset");
            const Type* pointee_type = curr_ptr_type->payload.ptr_type.pointed_type;

            const IntLiteral* lit = resolve_to_literal(offset);
            bool offset_is_zero = lit && lit->value.i64 == 0;
            assert(offset_is_zero || pointee_type->tag == ArrType_TAG && "if an offset is used, the base pointer must point to an array");
            uniform &= offset_uniform;

            // enter N levels of pointers
            size_t i = 2;
            while (true) {
                assert(curr_ptr_type->tag == PtrType_TAG && "lea is supposed to work on, and yield pointers");
                if (i >= prim_op.operands.count) break;

                const Node* selector = prim_op.operands.nodes[i];
                const Type* selector_type = selector->type;
                bool selector_uniform = deconstruct_qualified_type(&selector_type);

                assert(selector_type->tag == Int_TAG && "selectors must be integers");
                uniform &= selector_uniform;
                pointee_type = curr_ptr_type->payload.ptr_type.pointed_type;
                switch (pointee_type->tag) {
                    case ArrType_TAG: {
                        curr_ptr_type = ptr_type(arena, (PtrType) {
                            .pointed_type = pointee_type->payload.arr_type.element_type,
                            .address_space = curr_ptr_type->payload.ptr_type.address_space
                        });
                        break;
                    }
                    case TypeDeclRef_TAG: {
                        pointee_type = get_nominal_type_body(pointee_type);
                        SHADY_FALLTHROUGH
                    }
                    case RecordType_TAG: {
                        assert(selector->tag == IntLiteral_TAG && "selectors when indexing into a record need to be constant");
                        size_t index = get_int_literal_value(selector, false);
                        assert(index < pointee_type->payload.record_type.members.count);
                        curr_ptr_type = ptr_type(arena, (PtrType) {
                            .pointed_type = pointee_type->payload.record_type.members.nodes[index],
                            .address_space = curr_ptr_type->payload.ptr_type.address_space
                        });
                        break;
                    }
                    // also remember to assert literals for the selectors !
                    default: error("lea selectors can only work on pointers to arrays or records")
                }
                i++;
            }

            return qualified_type(arena, (QualifiedType) {
                .is_uniform = uniform,
                .type = curr_ptr_type
            });
        }
        case memcpy_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 3);
            const Type* dst_t = prim_op.operands.nodes[0]->type;
            deconstruct_qualified_type(&dst_t);
            assert(dst_t->tag == PtrType_TAG);
            const Type* src_t = prim_op.operands.nodes[1]->type;
            deconstruct_qualified_type(&src_t);
            assert(src_t->tag == PtrType_TAG);
            const Type* cnt_t = prim_op.operands.nodes[2]->type;
            deconstruct_qualified_type(&cnt_t);
            assert(cnt_t->tag == Int_TAG);
            return unit_type(arena);
        }
        case align_of_op:
        case size_of_op: {
            assert(prim_op.type_arguments.count == 1);
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = int_type(arena, (Int) { .width = arena->config.memory.ptr_size, .is_signed = false })
            });
        }
        case offset_of_op: {
            assert(prim_op.type_arguments.count == 1);
            assert(prim_op.operands.count == 1);
            const Type* optype = first(prim_op.operands)->type;
            bool uniform = deconstruct_qualified_type(&optype);
            assert(uniform && optype->tag == Int_TAG);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = int_type(arena, (Int) { .width = arena->config.memory.ptr_size, .is_signed = false })
            });
        }
        case select_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 3);
            const Type* condition_type = prim_op.operands.nodes[0]->type;
            bool condition_uniform = deconstruct_qualified_type(&condition_type);
            size_t width = deconstruct_maybe_packed_type(&condition_type);

            const Type* alternatives_types[2];
            bool alternatives_all_uniform = true;
            for (size_t i = 0; i < 2; i++) {
                alternatives_types[i] = prim_op.operands.nodes[1 + i]->type;
                alternatives_all_uniform &= deconstruct_qualified_type(&alternatives_types[i]);
                size_t alternative_width = deconstruct_maybe_packed_type(&alternatives_types[i]);
                assert(alternative_width == width);
            }

            assert(is_subtype(bool_type(arena), condition_type));
            // todo find true supertype
            assert(are_types_identical(2, alternatives_types));

            return qualified_type_helper(maybe_packed_type_helper(alternatives_types[0], width), alternatives_all_uniform && condition_uniform);
        }
        case insert_op:
        case extract_dynamic_op:
        case extract_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count >= 2);
            const Node* source = first(prim_op.operands);

            const Type* current_type = source->type;
            bool is_uniform = deconstruct_qualified_type(&current_type);

            size_t indices_start = prim_op.op == insert_op ? 2 : 1;

            for (size_t i = indices_start; i < prim_op.operands.count; i++) {
                assert(is_data_type(current_type));

                // Check index is valid !
                const Node* ith_index = prim_op.operands.nodes[i];
                bool dynamic_index = prim_op.op == extract_dynamic_op;
                if (dynamic_index) {
                    const Type* index_type = ith_index->type;
                    bool index_uniform = deconstruct_qualified_type(&index_type);
                    is_uniform &= index_uniform;
                    assert(index_type->tag == Int_TAG && "extract_dynamic requires integers for the indices");
                } else {
                    assert(ith_index->tag == IntLiteral_TAG && "extract takes integer literals");
                }

                // Go down one level...
                try_again:
                switch(current_type->tag) {
                    case RecordType_TAG: {
                        assert(!dynamic_index);
                        size_t index_value = ith_index->payload.int_literal.value.i32;
                        assert(index_value < current_type->payload.record_type.members.count);
                        current_type = current_type->payload.record_type.members.nodes[index_value];
                        continue;
                    }
                    case ArrType_TAG: {
                        assert(!dynamic_index);
                        current_type = current_type->payload.arr_type.element_type;
                        continue;
                    }
                    case TypeDeclRef_TAG: {
                        assert(!dynamic_index);
                        const Node* nom_decl = current_type->payload.type_decl_ref.decl;
                        assert(nom_decl->tag == NominalType_TAG);
                        current_type = nom_decl->payload.nom_type.body;
                        goto try_again;
                    }
                    case PackType_TAG: {
                        current_type = current_type->payload.pack_type.element_type;
                        continue;
                    }
                    default: error("Not a valid type to extract from")
                }
            }

            if (prim_op.op == insert_op) {
                const Node* inserted_data = prim_op.operands.nodes[1];
                const Type* inserted_data_type = inserted_data->type;
                is_uniform &= deconstruct_qualified_type(&inserted_data_type);
                assert(is_subtype(current_type, inserted_data_type) && "inserting data into a composite, but it doesn't match the target and indices");
                return qualified_type(arena, (QualifiedType) {
                    .is_uniform = is_uniform,
                    .type = get_unqualified_type(source->type),
                });
            }

            return qualified_type(arena, (QualifiedType) {
                .is_uniform = is_uniform,
                .type = current_type
            });
        }
        case reinterpret_op: {
            assert(prim_op.type_arguments.count == 1);
            assert(prim_op.operands.count == 1);
            const Node* source = first(prim_op.operands);
            const Type* src_type = source->type;
            bool src_uniform = deconstruct_qualified_type(&src_type);

            const Type* dst_type = first(prim_op.type_arguments);
            assert(is_data_type(dst_type));
            assert(is_reinterpret_cast_legal(src_type, dst_type));

            return qualified_type(arena, (QualifiedType) {
                .is_uniform = src_uniform,
                .type = dst_type
            });
        }
        case convert_op: {
            assert(prim_op.type_arguments.count == 1);
            assert(prim_op.operands.count == 1);
            const Node* source = first(prim_op.operands);
            const Type* src_type = source->type;
            bool src_uniform = deconstruct_qualified_type(&src_type);

            const Type* dst_type = first(prim_op.type_arguments);
            assert(is_data_type(dst_type));
            assert(is_conversion_legal(src_type, dst_type));

            // TODO check the conversion is legal
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = src_uniform,
                .type = dst_type
            });
        }
        // Mask management
        case empty_mask_op: {
            assert(prim_op.type_arguments.count == 0 && prim_op.operands.count == 0);
            return qualified_type_helper(get_actual_mask_type(arena), true);
        }
        case mask_is_thread_active_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 2);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = is_qualified_type_uniform(prim_op.operands.nodes[0]->type) && is_qualified_type_uniform(prim_op.operands.nodes[1]->type),
                .type = bool_type(arena)
            });
        }
        // Subgroup ops
        case subgroup_active_mask_op: {
            assert(prim_op.type_arguments.count == 0 && prim_op.operands.count == 0);
            return qualified_type_helper(get_actual_mask_type(arena), true);
        }
        case subgroup_ballot_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = get_actual_mask_type(arena)
            });
        }
        case subgroup_elect_first_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = false,
                .type = bool_type(arena)
            });
        }
        case subgroup_broadcast_first_op:
        case subgroup_reduce_sum_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);
            const Type* operand_type = get_unqualified_type(prim_op.operands.nodes[0]->type);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = operand_type
            });
        }
        // Intermediary ops
        case create_joint_point_op: {
            assert(prim_op.operands.count == 1);
            const Node* join_point = first(prim_op.operands);
            assert(is_qualified_type_uniform(join_point->type));
            return qualified_type(arena, (QualifiedType) { .type = join_point_type(arena, (JoinPointType) { .yield_types = prim_op.type_arguments }), .is_uniform = true });
        }
        case default_join_point_op: {
            assert(prim_op.operands.count == 0);
            assert(prim_op.type_arguments.count == 0);
            return qualified_type(arena, (QualifiedType) { .type = join_point_type(arena, (JoinPointType) { .yield_types = empty(arena) }), .is_uniform = true });
        }
        // Invocation ID and compute kernel stuff
        case subgroup_local_id_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = false,
                .type = uint32_type(arena)
            });
        }
        case subgroup_id_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = uint32_type(arena)
            });
        }
        case workgroup_id_op:
        case workgroup_num_op:
        case workgroup_size_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = pack_type(arena, (PackType) { .element_type = uint32_type(arena), .width = 3 })
            });
        }
        case workgroup_local_id_op:
        case global_id_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = false,
                .type = pack_type(arena, (PackType) { .element_type = uint32_type(arena), .width = 3 })
            });
        }
        // Stack stuff
        case get_stack_pointer_op:
        case get_stack_pointer_uniform_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) { .is_uniform = prim_op.op == get_stack_pointer_uniform_op, .type = uint32_type(arena) });
        }
        case get_stack_base_op:
        case get_stack_base_uniform_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 0);
            const Node* ptr = ptr_type(arena, (PtrType) { .pointed_type = arr_type(arena, (ArrType) { .element_type = uint8_type(arena), .size = NULL }), .address_space = prim_op.op == get_stack_base_op ? AsPrivatePhysical : AsSubgroupPhysical});
            return qualified_type(arena, (QualifiedType) { .is_uniform = prim_op.op == get_stack_base_uniform_op, .type = ptr });
        }
        case set_stack_pointer_op:
        case set_stack_pointer_uniform_op: {
            assert(prim_op.type_arguments.count == 0);
            assert(prim_op.operands.count == 1);
            bool is_uniform = prim_op.op == set_stack_pointer_uniform_op;
            if (is_uniform)
                assert(is_qualified_type_uniform(prim_op.operands.nodes[0]->type));
            assert(get_unqualified_type(prim_op.operands.nodes[0]->type) == uint32_type(arena));
            return unit_type(arena);
        }
        case push_stack_uniform_op:
        case push_stack_op: {
            assert(prim_op.type_arguments.count == 1);
            assert(prim_op.operands.count == 1);
            const Type* element_type = first(prim_op.type_arguments);
            assert(is_data_type(element_type));
            const Type* qual_element_type = qualified_type(arena, (QualifiedType) {
                .is_uniform = prim_op.op == push_stack_uniform_op,
                .type = element_type
            });
            // the operand has to be a subtype of the annotated type
            assert(is_subtype(qual_element_type, first(prim_op.operands)->type));
            return unit_type(arena);
        }
        case pop_stack_op:
        case pop_stack_uniform_op: {
            assert(prim_op.operands.count == 0);
            assert(prim_op.type_arguments.count == 1);
            const Type* element_type = prim_op.type_arguments.nodes[0];
            assert(is_data_type(element_type));
            return qualified_type(arena, (QualifiedType) { .is_uniform = prim_op.op == pop_stack_uniform_op, .type = element_type});
        }
        // Debugging ops
        case debug_printf_op: {
            assert(prim_op.type_arguments.count == 0);
            // TODO ?
            return unit_type(arena);
        }
        case PRIMOPS_COUNT: assert(false);
    }
}

static void check_arguments_types_against_parameters_helper(Nodes param_types, Nodes arg_types) {
    if (param_types.count != arg_types.count)
        error("Mismatched number of arguments/parameters");
    for (size_t i = 0; i < param_types.count; i++)
        check_subtype(param_types.nodes[i], arg_types.nodes[i]);
}

/// Shared logic between indirect calls and tailcalls
static Nodes check_value_call(const Node* callee, Nodes argument_types) {
    assert(is_value(callee));

    const Type* callee_type = callee->type;
    SHADY_UNUSED bool callee_uniform = deconstruct_qualified_type(&callee_type);
    AddressSpace as = deconstruct_pointer_type(&callee_type);
    assert(as == AsProgramCode);

    assert(callee_type->tag == FnType_TAG);

    const FnType* fn_type = &callee_type->payload.fn_type;
    check_arguments_types_against_parameters_helper(fn_type->param_types, argument_types);
    // TODO force the return types to be varying if the callee is not uniform
    return fn_type->return_types;
}

const Type* check_type_leaf_call(IrArena* arena, LeafCall call) {
    Nodes args = call.args;
    for (size_t i = 0; i < args.count; i++) {
        const Node* argument = args.nodes[i];
        assert(is_value(argument));
    }
    Nodes argument_types = get_values_types(arena, args);
    assert(is_function(call.callee));
    assert(call.callee->type->tag == FnType_TAG);
    check_arguments_types_against_parameters_helper(call.callee->type->payload.fn_type.param_types, argument_types);
    return wrap_multiple_yield_types(arena, call.callee->payload.fun.return_types);
}

const Type* check_type_indirect_call(IrArena* arena, IndirectCall call) {
    Nodes args = call.args;
    for (size_t i = 0; i < args.count; i++) {
        const Node* argument = args.nodes[i];
        assert(is_value(argument));
    }
    Nodes argument_types = get_values_types(arena, args);
    return wrap_multiple_yield_types(arena, check_value_call(call.callee, argument_types));
}

static void ensure_types_are_data_types(const Nodes* yield_types) {
    for (size_t i = 0; i < yield_types->count; i++) {
        assert(is_data_type(yield_types->nodes[i]));
    }
}

static void ensure_types_are_value_types(const Nodes* yield_types) {
    for (size_t i = 0; i < yield_types->count; i++) {
        assert(is_value_type(yield_types->nodes[i]));
    }
}

const Type* check_type_if_instr(IrArena* arena, If if_instr) {
    ensure_types_are_data_types(&if_instr.yield_types);
    if (get_unqualified_type(if_instr.condition->type) != bool_type(arena))
        error("condition of an if should be bool");
    // TODO check the contained Merge instrs
    if (if_instr.yield_types.count > 0)
        assert(if_instr.if_false);

    return wrap_multiple_yield_types(arena, add_qualifiers(arena, if_instr.yield_types, false));
}

const Type* check_type_loop_instr(IrArena* arena, Loop loop_instr) {
    ensure_types_are_data_types(&loop_instr.yield_types);
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return wrap_multiple_yield_types(arena, add_qualifiers(arena, loop_instr.yield_types, false));
}

const Type* check_type_match_instr(IrArena* arena, Match match_instr) {
    ensure_types_are_data_types(&match_instr.yield_types);
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return wrap_multiple_yield_types(arena, add_qualifiers(arena, match_instr.yield_types, false));
}

const Type* check_type_control(IrArena* arena, Control control) {
    ensure_types_are_data_types(&control.yield_types);
    // TODO check it then !
    assert(is_anonymous_lambda(control.inside));
    const Node* join_point = first(control.inside->payload.anon_lam.params);

    const Type* join_point_type = join_point->type;
    bool join_point_uniform = deconstruct_qualified_type(&join_point_type);
    assert(join_point_uniform && join_point_type->tag == JoinPointType_TAG);

    Nodes join_point_yield_types = join_point_type->payload.join_point_type.yield_types;
    assert(join_point_yield_types.count == control.yield_types.count);
    for (size_t i = 0; i < control.yield_types.count; i++) {
        assert(is_subtype(control.yield_types.nodes[i], join_point_yield_types.nodes[i]));
    }

    return wrap_multiple_yield_types(arena, add_qualifiers(arena, join_point_yield_types, false));
}

const Type* check_type_block(IrArena* arena, Block payload) {
    ensure_types_are_value_types(&payload.yield_types);
    assert(is_anonymous_lambda(payload.inside));
    assert(payload.inside->payload.anon_lam.params.count == 0);

    /*const Node* lam = payload.inside;
    const Node* yield_instr = NULL;
    while (true) {
        assert(lam->tag == AnonLambda_TAG);
        const Node* terminator = lam->payload.anon_lam.body;
        switch (terminator->tag) {
            case Let_TAG: {
                lam = terminator->payload.let.tail;
                continue;
            }
            case Yield_TAG:
                yield_instr = terminator;
                break;
            default: assert(false);
        }
        break;
    }

    Nodes yield_values = yield_instr->payload.yield.args;*/
    return wrap_multiple_yield_types(arena, payload.yield_types);
}

const Type* check_type_comment(IrArena* arena, SHADY_UNUSED Comment payload) {
    return unit_type(arena);
}

const Type* check_type_let(IrArena* arena, Let let) {
    assert(is_instruction(let.instruction));
    assert(is_anonymous_lambda(let.tail));
    Nodes produced_types = unwrap_multiple_yield_types(arena, let.instruction->type);
    Nodes param_types = get_variables_types(arena, let.tail->payload.anon_lam.params);

    check_arguments_types_against_parameters_helper(param_types, produced_types);
    return noret_type(arena);
}

const Type* check_type_tail_call(IrArena* arena, TailCall tail_call) {
    Nodes args = tail_call.args;
    for (size_t i = 0; i < args.count; i++) {
        const Node* argument = args.nodes[i];
        assert(is_value(argument));
    }
    assert(check_value_call(tail_call.target, get_values_types(arena, tail_call.args)).count == 0);
    return noret_type(arena);
}

static void check_basic_block_call(const Node* block, Nodes argument_types) {
    assert(is_basic_block(block));
    assert(block->type->tag == BBType_TAG);
    BBType bb_type = block->type->payload.bb_type;
    check_arguments_types_against_parameters_helper(bb_type.param_types, argument_types);
}

const Type* check_type_jump(IrArena* arena, Jump jump) {
    for (size_t i = 0; i < jump.args.count; i++) {
        const Node* argument = jump.args.nodes[i];
        assert(is_value(argument));
    }

    check_basic_block_call(jump.target, get_values_types(arena, jump.args));
    return noret_type(arena);
}

const Type* check_type_branch(IrArena* arena, Branch branch) {
    for (size_t i = 0; i < branch.args.count; i++) {
        const Node* argument = branch.args.nodes[i];
        assert(is_value(argument));
    }

    const Type* condition_type = branch.branch_condition->type;
    bool uniform = deconstruct_qualified_type(&condition_type);
    assert(bool_type(arena) == condition_type);

    const Node* branches[2] = { branch.true_target, branch.false_target };
    for (size_t i = 0; i < 2; i++)
        check_basic_block_call(branches[i], get_values_types(arena, branch.args));

    return noret_type(arena);
}

const Type* check_type_br_switch(IrArena* arena, Switch br_switch) {
    for (size_t i = 0; i < br_switch.args.count; i++) {
        const Node* argument = br_switch.args.nodes[i];
        assert(is_value(argument));
    }

    error("TODO")

    return noret_type(arena);
}

const Type* check_type_join(IrArena* arena, Join join) {
    for (size_t i = 0; i < join.args.count; i++) {
        const Node* argument = join.args.nodes[i];
        assert(is_value(argument));
    }

    const Type* join_target_type = join.join_point->type;

    bool join_target_uniform = deconstruct_qualified_type(&join_target_type);
    assert(join_target_uniform);
    assert(join_target_type->tag == JoinPointType_TAG);

    Nodes join_point_param_types = join_target_type->payload.join_point_type.yield_types;
    join_point_param_types = add_qualifiers(arena, join_point_param_types, !arena->config.is_simt);

    check_arguments_types_against_parameters_helper(join_point_param_types, get_values_types(arena, join.args));

    return noret_type(arena);
}

const Type* check_type_unreachable(IrArena* arena) {
    return noret_type(arena);
}

const Type* check_type_merge_continue(IrArena* arena, MergeContinue mc) {
    // TODO check it
    return noret_type(arena);
}

const Type* check_type_merge_break(IrArena* arena, MergeBreak mc) {
    // TODO check it
    return noret_type(arena);
}

const Type* check_type_yield(IrArena* arena, SHADY_UNUSED Yield payload) {
    // TODO check it
    return noret_type(arena);
}

const Type* check_type_fn_ret(IrArena* arena, Return ret) {
    // assert(ret.fn);
    // TODO check it then !
    return noret_type(arena);
}

const Type* check_type_fun(IrArena* arena, Function fn) {
    for (size_t i = 0; i < fn.return_types.count; i++) {
        assert(is_value_type(fn.return_types.nodes[i]));
    }
    return fn_type(arena, (FnType) { .param_types = get_variables_types(arena, (&fn)->params), .return_types = (&fn)->return_types });
}

const Type* check_type_basic_block(IrArena* arena, BasicBlock bb) {
    return bb_type(arena, (BBType) { .param_types = get_variables_types(arena, (&bb)->params) });
}

const Type* check_type_anon_lam(IrArena* arena, AnonLambda lam) {
    return lam_type(arena, (LamType) { .param_types = get_variables_types(arena, (&lam)->params) });
}

const Type* check_type_global_variable(IrArena* arena, GlobalVariable global_variable) {
    assert(is_type(global_variable.type));
    return ptr_type(arena, (PtrType) {
        .pointed_type = global_variable.type,
        .address_space = global_variable.address_space
    });
}

const Type* check_type_constant(IrArena* arena, Constant cnst) {
    assert(is_data_type(cnst.type_hint));
    return cnst.type_hint;
}

#pragma GCC diagnostic pop
