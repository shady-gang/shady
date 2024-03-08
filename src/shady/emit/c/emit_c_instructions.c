#include "emit_c.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "util.h"

#include "../../type.h"
#include "../../ir_private.h"

#include <assert.h>
#include <stdlib.h>

#pragma GCC diagnostic error "-Wswitch"

static void emit_pack_code(Printer* p, Strings src, String dst) {
    for (size_t i = 0; i < src.count; i++) {
        print(p, "\n%s->_%d = %s", dst, src.strings[i], i);
    }
}

static void emit_unpack_code(Printer* p, String src, Strings dst) {
    for (size_t i = 0; i < dst.count; i++) {
        print(p, "\n%s = %s->_%d", dst.strings[i], src, i);
    }
}

static void emit_variable_declaration(Emitter* emitter, Printer* block_printer, const Type* t, String variable_name, bool mut, const CTerm* initializer) {
    assert((mut || initializer != NULL) && "unbound results are only allowed when creating a mutable local variable");

    String prefix = "";
    String center = variable_name;

    // add extra qualifiers if immutable
    if (!mut) switch (emitter->config.dialect) {
            case ISPC:
                center = format_string_arena(emitter->arena->arena, "const %s", center);
                break;
            case C:
                prefix = "register ";
                center = format_string_arena(emitter->arena->arena, "const %s", center);
                break;
            case GLSL:
                prefix = "const ";
                break;
        }

    String decl = c_emit_type(emitter, t, center);
    if (initializer)
        print(block_printer, "\n%s%s = %s;", prefix, decl, to_cvalue(emitter, *initializer));
    else
        print(block_printer, "\n%s%s;", prefix, decl);
}

static Strings emit_variable_declarations(Emitter* emitter, Printer* p, String given_name, Strings* given_names, Nodes types, bool mut, const Nodes* init_values) {
    if (given_names)
        assert(given_names->count == types.count);
    if (init_values)
        assert(init_values->count == types.count);
    LARRAY(String, names, types.count);
    for (size_t i = 0; i < types.count; i++) {
        String name = given_names ? given_names->strings[i] : given_name;
        assert(name);
        names[i] = unique_name(emitter->arena, name);
        if (init_values) {
            CTerm initializer = c_emit_value(emitter, p, init_values->nodes[i]);
            emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, &initializer);
        } else
            emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, NULL);
    }
    return strings(emitter->arena, types.count, names);
}

static CTerm bind_and_register_emitted_cterm(Emitter* emitter, Printer* p, const Node* instruction, CTerm term) {
    const Type* t = instruction->type;
    Nodes yield_types = unwrap_multiple_yield_types(emitter->arena, instruction->type);
    assert(yield_types.count > 0);

    String bind_to = unique_name(emitter->arena, "_"); // TODO: better names
    emit_variable_declaration(emitter, p, t, bind_to, false, &term);
    register_emitted_cterm(emitter, instruction, term_from_cvalue(bind_to));
    return term;
}

static CTerm pack_terms(Emitter* emitter, Printer* p, Nodes types, CTerm* terms) {
    size_t count = types.count;
    IrArena* arena = emitter->arena;
    if (count == 0)
        return empty_term();
    if (count == 1) {
        return terms[0];
    }
    String dst = unique_name(emitter->arena, "packed_values");
    const Type* t = wrap_multiple_yield_types(arena, types);
    emit_variable_declarations(emitter, p, dst, NULL, singleton(t), true, NULL);
    LARRAY(String, outputs, count);
    for (size_t i = 0; i < count; i++) {
        outputs[i] = to_cvalue(emitter, terms[i]);
    }
    emit_pack_code(p, strings(arena, count, outputs), dst);
    return term_from_cvalue(dst);
}

static const Type* get_first_op_scalar_type(Nodes ops) {
    const Type* t = first(ops)->type;
    deconstruct_qualified_type(&t);
    deconstruct_maybe_packed_type(&t);
    return t;
}

typedef enum {
    OsInfix, OsPrefix, OsCall,
} OpStyle;

typedef enum {
    IsNone, // empty entry
    IsMono,
    IsPoly
} ISelMechanism;

typedef struct {
    ISelMechanism isel_mechanism;
    OpStyle style;
    String op;
    String u_ops[4];
    String s_ops[4];
    String f_ops[3];
} ISelTableEntry;

static const ISelTableEntry isel_table[PRIMOPS_COUNT] = {
    [add_op] = { IsMono, OsInfix,  "+" },
    [sub_op] = { IsMono, OsInfix,  "-" },
    [mul_op] = { IsMono, OsInfix,  "*" },
    [div_op] = { IsMono, OsInfix,  "/" },
    [mod_op] = { IsMono, OsInfix,  "%" },
    [neg_op] = { IsMono, OsPrefix, "-" },
    [gt_op] =  { IsMono, OsInfix,  ">" },
    [gte_op] = { IsMono, OsInfix,  ">=" },
    [lt_op] =  { IsMono, OsInfix,  "<"  },
    [lte_op] = { IsMono, OsInfix,  "<=" },
    [eq_op] =  { IsMono, OsInfix,  "==" },
    [neq_op] = { IsMono, OsInfix,  "!=" },
    [and_op] = { IsMono, OsInfix,  "&" },
    [or_op]  = { IsMono, OsInfix,  "|" },
    [xor_op] = { IsMono, OsInfix,  "^" },
    [not_op] = { IsMono, OsPrefix, "!" },
    [rshift_arithm_op] = { IsMono, OsInfix,  ">>" },
    [rshift_logical_op] = { IsMono, OsInfix,  ">>" }, // TODO achieve desired right shift semantics through unsigned/signed casts
    [lshift_op] = { IsMono, OsInfix,  "<<" },
};

static const ISelTableEntry isel_table_c[PRIMOPS_COUNT] = {
    [abs_op] = { IsPoly, OsCall, .s_ops = { "abs", "abs", "abs", "llabs" }, .f_ops = {"fabsf", "fabsf", "fabs"}},

    [sin_op] = { IsPoly, OsCall, .f_ops = {"sinf", "sinf", "sin"}},
    [cos_op] = { IsPoly, OsCall, .f_ops = {"cosf", "cosf", "cos"}},
    [floor_op] = { IsPoly, OsCall, .f_ops = {"floorf", "floorf", "floor"}},
    [ceil_op] = { IsPoly, OsCall, .f_ops = {"ceilf", "ceilf", "ceil"}},
    [round_op] = { IsPoly, OsCall, .f_ops = {"roundf", "roundf", "round"}},

    [sqrt_op] = { IsPoly, OsCall, .f_ops = {"sqrtf", "sqrtf", "sqrt"}},
    [exp_op] = { IsPoly, OsCall, .f_ops = {"expf", "expf", "exp"}},
    [pow_op] = { IsPoly, OsCall, .f_ops = {"powf", "powf", "pow"}},
};

static const ISelTableEntry isel_table_glsl[PRIMOPS_COUNT] = { 0 };

static const ISelTableEntry isel_table_ispc[PRIMOPS_COUNT] = {
    [abs_op] = { IsMono, OsCall, "abs" },

    [sin_op] = { IsMono, OsCall, "sin" },
    [cos_op] = { IsMono, OsCall, "cos" },
    [floor_op] = { IsMono, OsCall, "floor" },
    [ceil_op] = { IsMono, OsCall, "ceil" },
    [round_op] = { IsMono, OsCall, "round" },

    [sqrt_op] = { IsMono, OsCall, "sqrt" },
    [exp_op] = { IsMono, OsCall, "exp" },
    [pow_op] = { IsMono, OsCall, "pow" },

    [subgroup_active_mask_op] = { IsMono, OsCall, "lanemask" },
    [subgroup_ballot_op] = { IsMono, OsCall, "packmask" },
    [subgroup_reduce_sum_op] = { IsMono, OsCall, "reduce_add" },
};

static bool emit_using_entry(CTerm* out, Emitter* emitter, Printer* p, const ISelTableEntry* entry, Nodes operands) {
    String operator_str = NULL;
    switch (entry->isel_mechanism) {
        case IsNone: return false;
        case IsMono: operator_str = entry->op; break;
        case IsPoly: {
            const Type* t = get_first_op_scalar_type(operands);
            if (t->tag == Float_TAG)
                operator_str = entry->f_ops[t->payload.float_type.width];
            else if (t->tag == Int_TAG && t->payload.int_type.is_signed)
                operator_str = entry->s_ops[t->payload.int_type.width];
            else if (t->tag == Int_TAG)
                operator_str = entry->u_ops[t->payload.int_type.width];
            break;
        }
    }

    if (!operator_str)
        return false;

    switch (entry->style) {
        case OsInfix: {
            CTerm a = c_emit_value(emitter, p, operands.nodes[0]);
            CTerm b = c_emit_value(emitter, p, operands.nodes[1]);
            *out = term_from_cvalue(format_string_arena(emitter->arena->arena, "%s %s %s", to_cvalue(emitter, a), operator_str, to_cvalue(emitter, b)));
            break;
        }
        case OsPrefix: {
            CTerm operand = c_emit_value(emitter, p, operands.nodes[0]);
            *out = term_from_cvalue(format_string_arena(emitter->arena->arena, "%s%s", operator_str, to_cvalue(emitter, operand)));
            break;
        }
        case OsCall: {
            LARRAY(CTerm, cops, operands.count);
            for (size_t i = 0; i < operands.count; i++)
                cops[i] = c_emit_value(emitter, p, operands.nodes[i]);
            if (operands.count == 1)
                *out = term_from_cvalue(format_string_arena(emitter->arena->arena, "%s(%s)", operator_str, to_cvalue(emitter, cops[0])));
            else {
                Growy* g = new_growy();
                growy_append_string(g, operator_str);
                growy_append_string_literal(g, "(");
                for (size_t i = 0; i < operands.count; i++) {
                    growy_append_string(g, to_cvalue(emitter, cops[i]));
                    if (i + 1 < operands.count)
                        growy_append_string_literal(g, ", ");
                }
                growy_append_string_literal(g, ")");
                *out = term_from_cvalue(growy_deconstruct(g));
            }
            break;
        }
    }
    return true;
}

static const ISelTableEntry* lookup_entry(Emitter* emitter, Op op) {
    const ISelTableEntry* isel_entry = NULL;
    switch (emitter->config.dialect) {
        case C: isel_entry = &isel_table_c[op]; break;
        case GLSL: isel_entry = &isel_table_glsl[op]; break;
        case ISPC: isel_entry = &isel_table_ispc[op]; break;
    }
    if (isel_entry->isel_mechanism == IsNone)
        isel_entry = &isel_table[op];
    return isel_entry;
}

static CTerm emit_primop(Emitter* emitter, Printer* p, const Node* node) {
    assert(node->tag == PrimOp_TAG);
    IrArena* arena = emitter->arena;
    const PrimOp* prim_op = &node->payload.prim_op;
    CTerm term = term_from_cvalue(format_string_interned(emitter->arena, "/* todo %s */", get_primop_name(prim_op->op)));
    const ISelTableEntry* isel_entry = lookup_entry(emitter, prim_op->op);
    switch (prim_op->op) {
        case deref_op:
        case assign_op:
        case subscript_op: assert(false);
        case quote_op: {
            Nodes values = prim_op->operands;
            LARRAY(CTerm, terms, values.count);
            for (size_t i = 0; i < values.count; i++)
                terms[i] = c_emit_value(emitter, p, values.nodes[i]);
            return register_emitted_cterm(emitter, node, pack_terms(emitter, p, unwrap_multiple_yield_types(arena, node->type), terms));
        }
        case add_carry_op:
        case sub_borrow_op:
        case mul_extended_op:
            error("TODO: implement extended arithm ops in C");
            break;
        // MATH OPS
        case fract_op: {
            CTerm floored;
            emit_using_entry(&floored, emitter, p, lookup_entry(emitter, floor_op), prim_op->operands);
            term = term_from_cvalue(format_string_arena(arena->arena, "1 - %s", to_cvalue(emitter, floored)));
            break;
        }
        case inv_sqrt_op: {
            CTerm floored;
            emit_using_entry(&floored, emitter, p, lookup_entry(emitter, sqrt_op), prim_op->operands);
            term = term_from_cvalue(format_string_arena(arena->arena, "1.0f / %s", to_cvalue(emitter, floored)));
            break;
        }
        case min_op: {
            CValue a = to_cvalue(emitter, c_emit_value(emitter, p, first(prim_op->operands)));
            CValue b = to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[1]));
            term = term_from_cvalue(format_string_arena(arena->arena, "(%s > %s ? %s : %s)", a, b, b, a));
            break;
        }
        case max_op: {
            CValue a = to_cvalue(emitter, c_emit_value(emitter, p, first(prim_op->operands)));
            CValue b = to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[1]));
            term = term_from_cvalue(format_string_arena(arena->arena, "(%s > %s ? %s : %s)", a, b, a, b));
            break;
        }
        case sign_op: {
            CValue src = to_cvalue(emitter, c_emit_value(emitter, p, first(prim_op->operands)));
            term = term_from_cvalue(format_string_arena(arena->arena, "(%s > 0 ? 1 : -1)", src));
            break;
        }
        case alloca_op:
        case alloca_logical_op: {
            String variable_name = unique_name(emitter->arena, "alloca");
            CTerm variable = (CTerm) { .value = NULL, .var = variable_name };
            emit_variable_declaration(emitter, p, first(prim_op->type_arguments), variable_name, true, NULL);
            if (emitter->config.dialect == ISPC) {
                variable = ispc_varying_ptr_helper(emitter, p, get_unqualified_type(node->type), variable);
            }
            return register_emitted_cterm(emitter, node, variable);
        }
        case load_op: {
            CAddr dereferenced = deref_cterm(emitter, c_emit_value(emitter, p, first(prim_op->operands)));
            term = term_from_cvalue(dereferenced);
            break;
            // return bind_and_register_emitted_cterm(emitter, p, node, term_from_cvalue(dereferenced));
        }
        case store_op: {
            const Node* addr = first(prim_op->operands);
            const Node* value = prim_op->operands.nodes[1];
            const Type* addr_type = addr->type;
            bool addr_uniform = deconstruct_qualified_type(&addr_type);
            bool value_uniform = is_qualified_type_uniform(value->type);
            assert(addr_type->tag == PtrType_TAG);
            CAddr dereferenced = deref_cterm(emitter, c_emit_value(emitter, p, addr));
            CValue cvalue = to_cvalue(emitter, c_emit_value(emitter, p, value));
            // ISPC lets you broadcast to a uniform address space iff the address is non-uniform, otherwise we need to do this
            if (emitter->config.dialect == ISPC && addr_uniform && is_addr_space_uniform(arena, addr_type->payload.ptr_type.address_space) && !value_uniform)
                cvalue = format_string_arena(emitter->arena->arena, "extract(%s, count_trailing_zeros(lanemask()))", cvalue);

            print(p, "\n%s = %s;", dereferenced, cvalue);
            return register_emitted_cterm(emitter, node, empty_term());
        } case lea_op: {
            CTerm acc = c_emit_value(emitter, p, prim_op->operands.nodes[0]);

            const Type* src_qtype = prim_op->operands.nodes[0]->type;
            bool uniform = is_qualified_type_uniform(src_qtype);
            const Type* curr_ptr_type = get_unqualified_type(src_qtype);
            assert(curr_ptr_type->tag == PtrType_TAG);

            const IntLiteral* offset_static_value = resolve_to_int_literal(prim_op->operands.nodes[1]);
            if (!offset_static_value || offset_static_value->value != 0) {
                CTerm offset = c_emit_value(emitter, p, prim_op->operands.nodes[1]);
                // we sadly need to drop to the value level (aka explicit pointer arithmetic) to do this
                // this means such code is never going to be legal in GLSL
                // also the cast is to account for our arrays-in-structs hack
                acc = term_from_cvalue(format_string_arena(arena->arena, "((%s) &(%s.arr[%s]))", c_emit_type(emitter, curr_ptr_type, NULL), deref_cterm(emitter, acc), to_cvalue(emitter, offset)));
                uniform &= is_qualified_type_uniform(prim_op->operands.nodes[1]->type);
            }

            //t = t->payload.ptr_type.pointed_type;
            for (size_t i = 2; i < prim_op->operands.count; i++) {
                const Type* pointee_type = get_pointee_type(arena, curr_ptr_type);
                const Node* selector = prim_op->operands.nodes[i];
                uniform &= is_qualified_type_uniform(selector->type);
                switch (is_type(pointee_type)) {
                    case ArrType_TAG: {
                        CTerm index = c_emit_value(emitter, p, selector);
                        if (emitter->config.dialect == GLSL)
                            acc = term_from_cvar(format_string_arena(arena->arena, "(%s.arr[int(%s)])", deref_cterm(emitter, acc), to_cvalue(emitter, index)));
                        else
                            acc = term_from_cvar(format_string_arena(arena->arena, "(%s.arr[%s])", deref_cterm(emitter, acc), to_cvalue(emitter, index)));
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
                        // yet another ISPC bug and workaround
                        // ISPC cannot deal with subscripting if you've done pointer arithmetic (!) inside the expression
                        // so hum we just need to introduce a temporary variable to hold the pointer expression so far, and go again from there
                        // See https://github.com/ispc/ispc/issues/2496
                        if (emitter->config.dialect == ISPC) {
                            String interm = unique_name(arena, "lea_intermediary_ptr_value");
                            print(p, "\n%s = %s;", c_emit_type(emitter, qualified_type_helper(curr_ptr_type, uniform), interm), to_cvalue(emitter, acc));
                            acc = term_from_cvalue(interm);
                        }

                        assert(selector->tag == IntLiteral_TAG && "selectors when indexing into a record need to be constant");
                        size_t static_index = get_int_literal_value(*resolve_to_int_literal(selector), false);
                        String field_name = c_get_record_field_name(pointee_type, static_index);
                        acc = term_from_cvar(format_string_arena(arena->arena, "(%s.%s)", deref_cterm(emitter, acc), field_name));
                        curr_ptr_type = ptr_type(arena, (PtrType) {
                                .pointed_type = pointee_type->payload.record_type.members.nodes[static_index],
                                .address_space = curr_ptr_type->payload.ptr_type.address_space
                        });
                        break;
                    }
                    default: error("lea can't work on this");
                }
            }
            if (emitter->config.dialect == ISPC) {
                term = acc;
                break;
            }
            return register_emitted_cterm(emitter, node, acc);
        }
        case memcpy_op: {
            print(p, "\nmemcpy(%s, %s, %s);", to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[0])), to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[1])), to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[2])));
            return register_emitted_cterm(emitter, node, empty_term());
        }
        case size_of_op:
            term = term_from_cvalue(format_string_arena(emitter->arena->arena, "sizeof(%s)", c_emit_type(emitter, first(prim_op->type_arguments), NULL)));
            break;
        case align_of_op:
            term = term_from_cvalue(format_string_arena(emitter->arena->arena, "alignof(%s)", c_emit_type(emitter, first(prim_op->type_arguments), NULL)));
            break;
        case offset_of_op: {
            const Type* t = first(prim_op->type_arguments);
            while (t->tag == TypeDeclRef_TAG) {
                t = get_nominal_type_body(t);
            }
            const Node* index = first(prim_op->operands);
            uint64_t index_literal = get_int_literal_value(*resolve_to_int_literal(index), false);
            String member_name = c_get_record_field_name(t, index_literal);
            term = term_from_cvalue(format_string_arena(emitter->arena->arena, "offsetof(%s, %s)", c_emit_type(emitter, t, NULL), member_name));
            break;
        } case select_op: {
            assert(prim_op->operands.count == 3);
            CValue condition = to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[0]));
            CValue l = to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[1]));
            CValue r = to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[2]));
            term = term_from_cvalue(format_string_arena(emitter->arena->arena, "(%s) ? (%s) : (%s)", condition, l, r));
            break;
        }
        case convert_op: {
            CTerm src = c_emit_value(emitter, p, first(prim_op->operands));
            const Type* src_type = get_unqualified_type(first(prim_op->operands)->type);
            const Type* dst_type = first(prim_op->type_arguments);
            if (emitter->config.dialect == GLSL) {
                if (is_glsl_scalar_type(src_type) && is_glsl_scalar_type(dst_type)) {
                    CType t = c_emit_type(emitter, dst_type, NULL);
                    term = term_from_cvalue(format_string_arena(emitter->arena->arena, "%s(%s)", t, to_cvalue(emitter, src)));
                } else
                    assert(false);
            } else {
                CType t = c_emit_type(emitter, dst_type, NULL);
                term = term_from_cvalue(format_string_arena(emitter->arena->arena, "((%s) %s)", t, to_cvalue(emitter, src)));
            }
            break;
        }
        case reinterpret_op: {
            CTerm src_value = c_emit_value(emitter, p, first(prim_op->operands));
            const Type* src_type = get_unqualified_type(first(prim_op->operands)->type);
            const Type* dst_type = first(prim_op->type_arguments);
            switch (emitter->config.dialect) {
                case C: {
                    String src = unique_name(arena, "bitcast_src");
                    String dst = unique_name(arena, "bitcast_result");
                    print(p, "\n%s = %s;", c_emit_type(emitter, src_type, src), to_cvalue(emitter, src_value));
                    print(p, "\n%s;", c_emit_type(emitter, dst_type, dst));
                    print(p, "\nmemcpy(&%s, &s, sizeof(%s));", dst, src, src);
                    return register_emitted_cterm(emitter, node, term_from_cvalue(dst));
                }
                case GLSL: {
                    String n = NULL;
                    if (dst_type->tag == Float_TAG) {
                        assert(src_type->tag == Int_TAG);
                        switch (dst_type->payload.float_type.width) {
                            case FloatTy16: break;
                            case FloatTy32: n = src_type->payload.int_type.is_signed ? "intBitsToFloat" : "uintBitsToFloat";
                                break;
                            case FloatTy64: break;
                        }
                    } else if (dst_type->tag == Int_TAG) {
                        if (src_type->tag == Int_TAG) {
                            return register_emitted_cterm(emitter, node, src_value);
                        }
                        assert(src_type->tag == Float_TAG);
                        switch (src_type->payload.float_type.width) {
                            case FloatTy16: break;
                            case FloatTy32: n = dst_type->payload.int_type.is_signed ? "floatBitsToInt" : "floatBitsToUint";
                                break;
                            case FloatTy64: break;
                        }
                    }
                    if (n) {
                        term = term_from_cvalue(format_string_arena(emitter->arena->arena, "%s(%s)", n, to_cvalue(emitter, src_value)));
                        break;
                    }
                    error_print("glsl: unsupported bit cast from ");
                    log_node(ERROR, src_type);
                    error_print(" to ");
                    log_node(ERROR, dst_type);
                    error_print(".\n");
                    error_die();
                }
                case ISPC: {
                    if (dst_type->tag == Float_TAG) {
                        assert(src_type->tag == Int_TAG);
                        String n;
                        switch (dst_type->payload.float_type.width) {
                            case FloatTy16: n = "float16bits";
                                break;
                            case FloatTy32: n = "floatbits";
                                break;
                            case FloatTy64: n = "doublebits";
                                break;
                        }
                        term = term_from_cvalue(format_string_arena(emitter->arena->arena, "%s(%s)", n, to_cvalue(emitter, src_value)));
                        break;
                    } else if (src_type->tag == Float_TAG) {
                        assert(dst_type->tag == Int_TAG);
                        term = term_from_cvalue(format_string_arena(emitter->arena->arena, "intbits(%s)", to_cvalue(emitter, src_value)));
                        break;
                    }

                    CType t = c_emit_type(emitter, dst_type, NULL);
                    term = term_from_cvalue(format_string_arena(emitter->arena->arena, "((%s) %s)", t, to_cvalue(emitter, src_value)));
                    break;
                }
            }
            break;
        }
        case insert_op:
        case extract_dynamic_op:
        case extract_op: {
            CValue acc = to_cvalue(emitter, c_emit_value(emitter, p, first(prim_op->operands)));
            bool insert = prim_op->op == insert_op;

            if (insert) {
                String dst = unique_name(arena, "modified");
                print(p, "\n%s = %s;", c_emit_type(emitter, node->type, dst), acc);
                acc = dst;
                term = term_from_cvalue(dst);
            }

            const Type* t = get_unqualified_type(first(prim_op->operands)->type);
            for (size_t i = (insert ? 2 : 1); i < prim_op->operands.count; i++) {
                const Node* index = prim_op->operands.nodes[i];
                const IntLiteral* static_index = resolve_to_int_literal(index);

                switch (is_type(t)) {
                    case Type_TypeDeclRef_TAG: {
                        const Node* decl = t->payload.type_decl_ref.decl;
                        assert(decl && decl->tag == NominalType_TAG);
                        t = decl->payload.nom_type.body;
                        SHADY_FALLTHROUGH
                    }
                    case Type_RecordType_TAG: {
                        assert(static_index);
                        Strings names = t->payload.record_type.names;
                        if (names.count == 0)
                            acc = format_string_arena(emitter->arena->arena, "(%s._%d)", acc, static_index->value);
                        else
                            acc = format_string_arena(emitter->arena->arena, "(%s.%s)", acc, names.strings[static_index->value]);
                        break;
                    }
                    case Type_PackType_TAG: {
                        assert(static_index);
                        assert(static_index->value < 4 && static_index->value < t->payload.pack_type.width);
                        String suffixes = "xyzw";
                        acc = format_string_arena(emitter->arena->arena, "(%s.%c)", acc, suffixes[static_index->value]);
                        break;
                    }
                    case Type_ArrType_TAG: {
                        if (emitter->config.dialect == GLSL)
                            acc = format_string_arena(emitter->arena->arena, "(%s.arr[int(%s)])", acc, to_cvalue(emitter, c_emit_value(emitter, p, index)));
                        else
                            acc = format_string_arena(emitter->arena->arena, "(%s.arr[%s])", acc, to_cvalue(emitter, c_emit_value(emitter, p, index)));
                        break;
                    }
                    default:
                    case NotAType: error("Must be a type");
                }
            }

            if (insert) {
                print(p, "\n%s = %s;", acc, to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[1])));
                break;
            }

            term = term_from_cvalue(acc);
            break;
        }
        case get_stack_base_op:
        case push_stack_op:
        case pop_stack_op:
        case get_stack_pointer_op:
        case set_stack_pointer_op: error("Stack operations need to be lowered.");
        case default_join_point_op:
        case create_joint_point_op: error("lowered in lower_tailcalls.c");
        case subgroup_elect_first_op: {
            switch (emitter->config.dialect) {
                case ISPC: term = term_from_cvalue(format_string_arena(emitter->arena->arena, "(programIndex == count_trailing_zeros(lanemask()))")); break;
                case C:
                case GLSL: error("TODO")
            }
            break;
        }
        case subgroup_assume_uniform_op:
        case subgroup_broadcast_first_op: {
            CValue value = to_cvalue(emitter, c_emit_value(emitter, p, first(prim_op->operands)));
            switch (emitter->config.dialect) {
                case ISPC: term = term_from_cvalue(format_string_arena(emitter->arena->arena, "extract(%s, count_trailing_zeros(lanemask()))", value)); break;
                case C:
                case GLSL: error("TODO")
            }
            break;
        }
        case empty_mask_op:
        case mask_is_thread_active_op: error("lower_me");
        case debug_printf_op: {
            String args_list = "";
            for (size_t i = 0; i < prim_op->operands.count; i++) {
                CValue str = to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[i]));

                if (emitter->config.dialect == ISPC && i > 0)
                    str = format_string_arena(emitter->arena->arena, "extract(%s, printf_thread_index)", str);

                if (i > 0)
                    args_list = format_string_arena(emitter->arena->arena, "%s, %s", args_list, str);
                else
                    args_list = str;
            }
            switch (emitter->config.dialect) {
                case ISPC:
                    print(p, "\nforeach_active(printf_thread_index) { print(%s); }", args_list);
                    break;
                case C:
                    print(p, "\nprintf(%s);", args_list);
                    break;
                case GLSL: warn_print("printf is not supported in GLSL");
                    break;
            }

            return register_emitted_cterm(emitter, node, empty_term());
        }
        default: break;
        case PRIMOPS_COUNT: assert(false); break;
    }

    if (isel_entry->isel_mechanism != IsNone)
        emit_using_entry(&term, emitter, p, isel_entry, prim_op->operands);

    return bind_and_register_emitted_cterm(emitter, p, node, term);
}

static CTerm emit_call(Emitter* emitter, Printer* p, const Node* call) {
    Nodes args;
    if (call->tag == Call_TAG)
        args = call->payload.call.args;
    else
        assert(false);

    Growy* g = new_growy();
    Printer* paramsp = open_growy_as_printer(g);
    for (size_t i = 0; i < args.count; i++) {
        print(paramsp, to_cvalue(emitter, c_emit_value(emitter, p, args.nodes[i])));
        if (i + 1 < args.count)
            print(paramsp, ", ");
    }

    CValue e_callee;
    const Node* callee = call->payload.call.callee;
    if (callee->tag == FnAddr_TAG)
        e_callee = get_declaration_name(callee->payload.fn_addr.fn);
    else
        e_callee = to_cvalue(emitter, c_emit_value(emitter, p, callee));

    String params = printer_growy_unwrap(paramsp);
    CTerm term = term_from_cvalue(format_string_arena(emitter->arena->arena, "%s(%s)", e_callee, params));
    free_tmp_str(params);
    return bind_and_register_emitted_cterm(emitter, p, call, term);
}

static void emit_if(Emitter* emitter, Printer* p, const Node* if_instr) {
    assert(if_instr->tag == If_TAG);
    const If* if_ = &if_instr->payload.structured_if;
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, p, "if_phi", NULL, if_->yield_types, true, NULL);
    sub_emiter.phis.selection = ephis;

    assert(get_abstraction_params(if_->if_true).count == 0);
    String true_body = c_emit_lambda_body(&sub_emiter, get_abstraction_body(if_->if_true), NULL);
    CValue condition = to_cvalue(emitter, c_emit_value(emitter, p, if_->condition));
    print(p, "\nif (%s) { %s}", condition, true_body);
    free_tmp_str(true_body);
    if (if_->if_false) {
        assert(get_abstraction_params(if_->if_false).count == 0);
        String false_body = c_emit_lambda_body(&sub_emiter, get_abstraction_body(if_->if_false), NULL);
        print(p, " else {%s}", false_body);
        free_tmp_str(false_body);
    }

    const Node* tail = if_->tail;
    assert(tail->tag == Case_TAG);
    Nodes tail_params = get_abstraction_params(tail);
    for (size_t i = 0; i < ephis.count; i++)
        register_emitted_cterm(emitter, tail_params.nodes[i], term_from_cvalue(ephis.strings[i]));
    c_emit_lambda_body_at(emitter, p, get_abstraction_body(tail), NULL);
}

static void emit_match(Emitter* emitter, Printer* p, const Node* match_instr) {
    assert(match_instr->tag == Match_TAG);
    const Match* match = &match_instr->payload.structured_match;
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, p, "match_phi", NULL, match->yield_types, true, NULL);
    sub_emiter.phis.selection = ephis;

    // Of course, the sensible thing to do here would be to emit a switch statement.
    // ...
    // Except that doesn't work, because C/GLSL have a baffling design wart: the `break` statement is overloaded,
    // meaning that if you enter a switch statement, which should be orthogonal to loops, you can't actually break
    // out of the outer loop anymore. Brilliant. So we do this terrible if-chain instead.
    //
    // We could do GOTO for C, but at the cost of arguably even more noise in the output, and two different codepaths.
    // I don't think it's quite worth it, just like it's not worth doing some data-flow based solution either.

    CValue inspectee = to_cvalue(emitter, c_emit_value(emitter, p, match->inspect));
    bool first = true;
    LARRAY(CValue, literals, match->cases.count);
    for (size_t i = 0; i < match->cases.count; i++) {
        literals[i] = to_cvalue(emitter, c_emit_value(emitter, p, match->literals.nodes[i]));
    }
    for (size_t i = 0; i < match->cases.count; i++) {
        String case_body = c_emit_lambda_body(&sub_emiter, get_abstraction_body(match->cases.nodes[i]), NULL);
        print(p, "\n");
        if (!first)
            print(p, "else ");
        print(p, "if (%s == %s) { %s}", inspectee, literals[i], case_body);
        free_tmp_str(case_body);
        first = false;
    }
    if (match->default_case) {
        String default_case_body = c_emit_lambda_body(&sub_emiter, get_abstraction_body(match->default_case), NULL);
        print(p, "\nelse { %s}", default_case_body);
        free_tmp_str(default_case_body);
    }

    const Node* tail = match->tail;
    assert(tail->tag == Case_TAG);
    Nodes tail_params = get_abstraction_params(tail);
    for (size_t i = 0; i < ephis.count; i++)
        register_emitted_cterm(emitter, tail_params.nodes[i], term_from_cvalue(ephis.strings[i]));
    c_emit_lambda_body_at(emitter, p, get_abstraction_body(tail), NULL);
}

static void emit_loop(Emitter* emitter, Printer* p, const Node* loop_instr) {
    assert(loop_instr->tag == Loop_TAG);
    const Loop* loop = &loop_instr->payload.structured_loop;

    Emitter sub_emiter = *emitter;
    Nodes params = get_abstraction_params(loop->body);
    Nodes variables = params;
    LARRAY(String, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        arr[i] = get_value_name(variables.nodes[i]);
        if (!arr[i])
            arr[i] = unique_name(emitter->arena, "phi");
    }
    Strings param_names = strings(emitter->arena, variables.count, arr);
    Strings eparams = emit_variable_declarations(emitter, p, NULL, &param_names, get_variables_types(emitter->arena, params), true, &loop->initial_args);
    for (size_t i = 0; i < params.count; i++)
        register_emitted_cterm(&sub_emiter, params.nodes[i], term_from_cvalue(eparams.strings[i]));

    sub_emiter.phis.loop_continue = eparams;
    Strings ephis = emit_variable_declarations(emitter, p, "loop_break_phi", NULL, loop->yield_types, true, NULL);
    sub_emiter.phis.loop_break = ephis;

    String body = c_emit_lambda_body(&sub_emiter, get_abstraction_body(loop->body), NULL);
    print(p, "\nwhile(true) { %s}", body);
    free_tmp_str(body);

    const Node* tail = loop->tail;
    assert(tail->tag == Case_TAG);
    Nodes tail_params = get_abstraction_params(tail);
    for (size_t i = 0; i < ephis.count; i++)
        register_emitted_cterm(emitter, tail_params.nodes[i], term_from_cvalue(ephis.strings[i]));
    c_emit_lambda_body_at(emitter, p, get_abstraction_body(tail), NULL);
}

static void emit_body(Emitter* emitter, Printer* p, const Node* body) {
    for (size_t i = 0; i < body->payload.body.instructions.count; i++) {
        const Node* instruction = body->payload.body.instructions.nodes[i];
        c_emit_instruction(emitter, p, instruction);
    }
    c_emit_terminator(emitter, p, body->payload.body.terminator);
}

void c_emit_terminator(Emitter* emitter, Printer* block_printer, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case InsertHelperEnd_TAG: assert(false);
        case NotATerminator: assert(false);
        case Control_TAG:
        case Join_TAG: error("control/join must be lowered elsewhere");
        case Jump_TAG:
        case Branch_TAG:
        case Switch_TAG:
        case TailCall_TAG: error("TODO");
        case If_TAG:
            emit_if(emitter, block_printer, terminator);
            return;
        case Match_TAG:
            emit_match(emitter, block_printer, terminator);
            return;
        case Loop_TAG:
            emit_loop(emitter, block_printer, terminator);
            return;
        case Body_TAG:
            emit_body(emitter, block_printer, terminator);
            return;
        case Terminator_Return_TAG: {
            Nodes args = terminator->payload.fn_ret.args;
            if (args.count == 0) {
                print(block_printer, "\nreturn;");
            } else if (args.count == 1) {
                print(block_printer, "\nreturn %s;", to_cvalue(emitter, c_emit_value(emitter, block_printer, args.nodes[0])));
            } else {
                String packed = unique_name(emitter->arena, "pack_return");
                LARRAY(CValue, values, args.count);
                for (size_t i = 0; i < args.count; i++)
                    values[i] = to_cvalue(emitter, c_emit_value(emitter, block_printer, args.nodes[i]));
                emit_pack_code(block_printer, strings(emitter->arena, args.count, values), packed);
                print(block_printer, "\nreturn %s;", packed);
            }
            break;
        }
        case Yield_TAG: {
            Nodes args = terminator->payload.yield.args;
            Phis phis = emitter->phis.selection;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, c_emit_value(emitter, block_printer, args.nodes[i])));

            break;
        }
        case MergeContinue_TAG: {
            Nodes args = terminator->payload.merge_continue.args;
            Phis phis = emitter->phis.loop_continue;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, c_emit_value(emitter, block_printer, args.nodes[i])));
            print(block_printer, "\ncontinue;");
            break;
        }
        case MergeBreak_TAG: {
            Nodes args = terminator->payload.merge_break.args;
            Phis phis = emitter->phis.loop_break;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, c_emit_value(emitter, block_printer, args.nodes[i])));
            print(block_printer, "\nbreak;");
            break;
        }
        case Unreachable_TAG: {
            switch (emitter->config.dialect) {
                case C:
                    print(block_printer, "\n__builtin_unreachable();");
                    break;
                case ISPC:
                    print(block_printer, "\nassert(false);");
                    break;
                case GLSL:
                    print(block_printer, "\n//unreachable");
                    break;
            }
            break;
        }
    }
}

CTerm c_emit_instruction(Emitter* emitter, Printer* p, const Node* instruction) {
    assert(is_instruction(instruction));

    switch (is_instruction(instruction)) {
        case NotAnInstruction: assert(false);
        case Instruction_InsertHelper_TAG: error("Cannot be emitted.")
        case Instruction_PrimOp_TAG:       return emit_primop(emitter, p, instruction);
        case Instruction_Call_TAG:         return emit_call(emitter, p, instruction);
        case Instruction_Comment_TAG:      print(p, "/* %s */", instruction->payload.comment.string); return empty_term();
    }
}
