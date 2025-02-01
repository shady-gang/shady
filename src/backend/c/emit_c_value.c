#include "emit_c.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "util.h"

#include "../shady/ir_private.h"
#include "../shady/analysis/scheduler.h"

#include <spirv/unified1/spirv.h>

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>

#pragma GCC diagnostic error "-Wswitch"

static CTerm emit_instruction(Emitter* emitter, FnEmitter* fn, Printer* p, const Node* instruction);

static enum { ObjectsList, StringLit, CharsLit } array_insides_helper(Emitter* e, FnEmitter* fn, Printer* p, Growy* g, const Node* t, Nodes c) {
    if (t->tag == Int_TAG && t->payload.int_type.width == 8) {
        uint8_t* tmp = malloc(sizeof(uint8_t) * c.count);
        bool ends_zero = false;
        for (size_t i = 0; i < c.count; i++) {
            tmp[i] = shd_get_int_literal_value(*shd_resolve_to_int_literal(c.nodes[i]), false);
            if (tmp[i] == 0) {
                if (i == c.count - 1)
                    ends_zero = true;
            }
        }
        bool is_stringy = ends_zero;
        for (size_t i = 0; i < c.count; i++) {
            // ignore the last char in a string
            if (is_stringy && i == c.count - 1)
                break;
            if (isprint(tmp[i]))
                shd_print(p, "%c", tmp[i]);
            else
                shd_print(p, "\\x%02x", tmp[i]);
        }
        free(tmp);
        return is_stringy ? StringLit : CharsLit;
    } else {
        for (size_t i = 0; i < c.count; i++) {
            shd_print(p, shd_c_to_ssa(e, shd_c_emit_value(e, fn, c.nodes[i])));
            if (i + 1 < c.count)
                shd_print(p, ", ");
        }
        shd_growy_append_bytes(g, 1, "\0");
        return ObjectsList;
    }
}

static CTerm c_emit_value_(Emitter* emitter, FnEmitter* fn, Printer* p, const Node* value) {
    if (is_instruction(value))
        return emit_instruction(emitter, fn, p, value);
    
    String emitted = NULL;

    switch (is_value(value)) {
        case NotAValue: assert(false);
        case Value_UntypedNumber_TAG: shd_error("lower me");
        case Param_TAG: shd_error("tried to emit a param: all params should be emitted by their binding abstraction !");
        default: {
            assert(!is_instruction(value));
            shd_error("Unhandled value for code generation: %s", shd_get_node_tag_string(value->tag));
        }
        case Value_IntLiteral_TAG: {
            if (value->payload.int_literal.is_signed)
                emitted = shd_format_string_arena(emitter->arena->arena, "%" PRIi64, value->payload.int_literal.value);
            else
                emitted = shd_format_string_arena(emitter->arena->arena, "%" PRIu64, value->payload.int_literal.value);

            bool is_long = value->payload.int_literal.width == IntTy64;
            bool is_signed = value->payload.int_literal.is_signed;
            if (emitter->config.dialect == CDialect_GLSL && emitter->config.glsl_version >= 130) {
                if (!is_signed)
                    emitted = shd_format_string_arena(emitter->arena->arena, "%sU", emitted);
                if (is_long)
                    emitted = shd_format_string_arena(emitter->arena->arena, "%sL", emitted);
            }

            break;
        }
        case Value_FloatLiteral_TAG: {
            uint64_t v = value->payload.float_literal.value;
            switch (value->payload.float_literal.width) {
                case FloatTy16:
                    assert(false);
                case FloatTy32: {
                    float f;
                    memcpy(&f, &v, sizeof(uint32_t));
                    double d = (double) f;
                    emitted = shd_format_string_arena(emitter->arena->arena, "%#.9gf", d); break;
                }
                case FloatTy64: {
                    double d;
                    memcpy(&d, &v, sizeof(uint64_t));
                    emitted = shd_format_string_arena(emitter->arena->arena, "%.17g", d); break;
                }
            }
            break;
        }
        case Value_True_TAG: return term_from_cvalue("true");
        case Value_False_TAG: return term_from_cvalue("false");
        case Value_Undef_TAG: {
            if (emitter->config.dialect == CDialect_GLSL)
                return shd_c_emit_value(emitter, fn, shd_get_default_value(emitter->arena, value->payload.undef.type));
            String name = shd_make_unique_name(emitter->arena, "undef");
            // c_emit_variable_declaration(emitter, block_printer, value->type, name, true, NULL);
            shd_c_emit_global_variable_definition(emitter, AsGlobal, name, value->payload.undef.type, true, NULL);
            emitted = name;
            break;
        }
        case Value_NullPtr_TAG: return term_from_cvalue("NULL");
        case Value_Composite_TAG: {
            const Type* type = value->payload.composite.type;
            Nodes elements = value->payload.composite.contents;

            Growy* g = shd_new_growy();
            Printer* p2 = p;
            Printer* p = shd_new_printer_from_growy(g);

            if (type->tag == ArrType_TAG) {
                switch (array_insides_helper(emitter, fn, p, g, type, elements)) {
                    case ObjectsList:
                        emitted = shd_growy_data(g);
                        break;
                    case StringLit:
                        emitted = shd_format_string_arena(emitter->arena->arena, "\"%s\"", shd_growy_data(g));
                        break;
                    case CharsLit:
                        emitted = shd_format_string_arena(emitter->arena->arena, "'%s'", shd_growy_data(g));
                        break;
                }
            } else {
                for (size_t i = 0; i < elements.count; i++) {
                    shd_print(p, "%s", shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, elements.nodes[i])));
                    if (i + 1 < elements.count)
                        shd_print(p, ", ");
                }
                emitted = shd_growy_data(g);
            }
            shd_growy_append_bytes(g, 1, "\0");

            switch (emitter->config.dialect) {
                no_compound_literals:
                case CDialect_ISPC: {
                    // arrays need double the brackets
                    if (type->tag == ArrType_TAG)
                        emitted = shd_format_string_arena(emitter->arena->arena, "{ %s }", emitted);

                    if (p2) {
                        String tmp = shd_make_unique_name(emitter->arena, "composite");
                        shd_print(p2, "\n%s = { %s };", shd_c_emit_type(emitter, value->type, tmp), emitted);
                        emitted = tmp;
                    } else {
                        // this requires us to end up in the initialisation side of a declaration
                        emitted = shd_format_string_arena(emitter->arena->arena, "{ %s }", emitted);
                    }
                    break;
                }
                case CDialect_CUDA:
                case CDialect_C11:
                    // If we're C89 (ew)
                    if (!emitter->config.allow_compound_literals)
                        goto no_compound_literals;
                    emitted = shd_format_string_arena(emitter->arena->arena, "((%s) { %s })", shd_c_emit_type(emitter, value->type, NULL), emitted);
                    break;
                case CDialect_GLSL:
                    if (type->tag != PackType_TAG)
                        goto no_compound_literals;
                    // GLSL doesn't have compound literals, but it does have constructor syntax for vectors
                    emitted = shd_format_string_arena(emitter->arena->arena, "%s(%s)", shd_c_emit_type(emitter, value->type, NULL), emitted);
                    break;
            }

            shd_destroy_growy(g);
            shd_destroy_printer(p);
            break;
        }
        case Value_Fill_TAG: shd_error("lower me")
        case Value_StringLiteral_TAG: {
            Growy* g = shd_new_growy();
            Printer* p = shd_new_printer_from_growy(g);

            String str = value->payload.string_lit.string;
            size_t len = strlen(str);
            for (size_t i = 0; i < len; i++) {
                char c = str[i];
                switch (c) {
                    case '\n': shd_print(p, "\\n");
                        break;
                    default:
                        shd_growy_append_bytes(g, 1, &c);
                }
            }
            shd_growy_append_bytes(g, 1, "\0");

            emitted = shd_format_string_arena(emitter->arena->arena, "\"%s\"", shd_growy_data(g));
            shd_destroy_growy(g);
            shd_destroy_printer(p);
            break;
        }
        case Value_FnAddr_TAG: {
            emitted = shd_c_legalize_identifier(emitter, shd_get_node_name_safe(value->payload.fn_addr.fn));
            emitted = shd_format_string_arena(emitter->arena->arena, "(&%s)", emitted);
            break;
        }
        case Value_GlobalVariable_TAG:
        case Value_Constant_TAG: {
            shd_c_emit_decl(emitter, value);
            return *shd_c_lookup_existing_term(emitter, NULL, value);
        }
        case BuiltinRef_TAG: {
            Builtin b = value->payload.builtin_ref.builtin;
            CTerm t = shd_c_emit_builtin(emitter, b);
            shd_c_register_emitted(emitter, NULL, value, t);
            return t;
        }
    }

    assert(emitted);
    return term_from_cvalue(emitted);
}

CTerm shd_c_bind_intermediary_result(Emitter* emitter, Printer* p, const Type* t, CTerm term) {
    if (is_term_empty(term))
        return term;
    if (t == empty_multiple_return_type(emitter->arena)) {
        shd_print(p, "%s;", shd_c_to_ssa(emitter, term));
        return empty_term();
    }
    String bind_to = shd_make_unique_name(emitter->arena, "");
    shd_c_emit_variable_declaration(emitter, p, t, bind_to, false, &term);
    return term_from_cvalue(bind_to);
}

static const Type* get_first_op_scalar_type(Nodes ops) {
    const Type* t = shd_first(ops)->type;
    shd_deconstruct_qualified_type(&t);
    shd_deconstruct_maybe_packed_type(&t);
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

static const ISelTableEntry isel_dummy = { IsNone };

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
    /*[rshift_arithm_op] = { IsMono, OsInfix,  ">>" },
    [rshift_logical_op] = { IsMono, OsInfix,  ">>" }, // TODO achieve desired right shift semantics through unsigned/signed casts
    [lshift_op] = { IsMono, OsInfix,  "<<" },*/
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

static const ISelTableEntry isel_table_glsl[PRIMOPS_COUNT] = {
    [abs_op] = { IsMono, OsCall, "abs" },

    [sin_op] = { IsMono, OsCall, "sin" },
    [cos_op] = { IsMono, OsCall, "cos" },
    [floor_op] = { IsMono, OsCall, "floor" },
    [ceil_op] = { IsMono, OsCall, "ceil" },
    [round_op] = { IsMono, OsCall, "round" },

    [sqrt_op] = { IsMono, OsCall, "sqrt" },
    [exp_op] = { IsMono, OsCall, "exp" },
    [pow_op] = { IsMono, OsCall, "pow" },
};

static const ISelTableEntry isel_table_glsl_120[PRIMOPS_COUNT] = {
    [mod_op] = { IsMono, OsCall,  "mod" },

    [and_op] = { IsMono, OsCall,  "and" },
    [ or_op] = { IsMono, OsCall,   "or" },
    [xor_op] = { IsMono, OsCall,  "xor" },
    [not_op] = { IsMono, OsCall,  "not" },
};

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
};

static CTerm emit_using_entry(Emitter* emitter, FnEmitter* fn, Printer* p, const ISelTableEntry* entry, Nodes operands) {
    String operator_str = NULL;
    switch (entry->isel_mechanism) {
        case IsNone: return empty_term();
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

    if (!operator_str) {
        shd_log_fmt(ERROR, "emit_c: Missing or unsupported operands for this entry");
        return empty_term();
    }

    switch (entry->style) {
        case OsInfix: {
            CTerm a = shd_c_emit_value(emitter, fn, operands.nodes[0]);
            CTerm b = shd_c_emit_value(emitter, fn, operands.nodes[1]);
            return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "%s %s %s", shd_c_to_ssa(emitter, a), operator_str, shd_c_to_ssa(emitter, b)));
        }
        case OsPrefix: {
            CTerm operand = shd_c_emit_value(emitter, fn, operands.nodes[0]);
            return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "%s%s", operator_str, shd_c_to_ssa(emitter, operand)));
        }
        case OsCall: {
            LARRAY(CTerm, cops, operands.count);
            for (size_t i = 0; i < operands.count; i++)
                cops[i] = shd_c_emit_value(emitter, fn, operands.nodes[i]);
            if (operands.count == 1)
                return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "%s(%s)", operator_str, shd_c_to_ssa(emitter, cops[0])));
            else {
                Growy* g = shd_new_growy();
                shd_growy_append_string(g, operator_str);
                shd_growy_append_string_literal(g, "(");
                for (size_t i = 0; i < operands.count; i++) {
                    shd_growy_append_string(g, shd_c_to_ssa(emitter, cops[i]));
                    if (i + 1 < operands.count)
                        shd_growy_append_string_literal(g, ", ");
                }
                shd_growy_append_string_literal(g, ")");
                String s = shd_growy_deconstruct(g);
                CValue value = shd_fmt_string_irarena(emitter->arena, "%s", s);
                free((void*) s);
                return term_from_cvalue(value);
            }
            break;
        }
    }

    SHADY_UNREACHABLE;
}

static const ISelTableEntry* lookup_entry(Emitter* emitter, Op op) {
    const ISelTableEntry* isel_entry = &isel_dummy;

    switch (emitter->config.dialect) {
        case CDialect_CUDA: /* TODO: do better than that */
        case CDialect_C11: isel_entry = &isel_table_c[op]; break;
        case CDialect_GLSL: isel_entry = &isel_table_glsl[op]; break;
        case CDialect_ISPC: isel_entry = &isel_table_ispc[op]; break;
    }

    if (emitter->config.dialect == CDialect_GLSL && emitter->config.glsl_version <= 120)
        isel_entry = &isel_table_glsl_120[op];

    if (isel_entry->isel_mechanism == IsNone)
        isel_entry = &isel_table[op];
    return isel_entry;
}

static String index_into_array(Emitter* emitter, const Type* arr_type, CTerm expr, CTerm index) {
    IrArena* arena = emitter->arena;

    String index2 = emitter->config.dialect == CDialect_GLSL ? shd_format_string_arena(arena->arena, "int(%s)", shd_c_to_ssa(emitter, index)) : shd_c_to_ssa(emitter, index);
    if (emitter->config.decay_unsized_arrays && !arr_type->payload.arr_type.size)
        return shd_format_string_arena(arena->arena, "((&%s)[%s])", shd_c_deref(emitter, expr), index2);
    else
        return shd_format_string_arena(arena->arena, "(%s.arr[%s])", shd_c_deref(emitter, expr), index2);
}

static CTerm broadcast_first(Emitter* emitter, CValue value, const Type* value_type) {
    switch (emitter->config.dialect) {
        case CDialect_ISPC: {
            const Type* t = shd_get_unqualified_type(value_type);
            return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "__shady_extract(%s, count_trailing_zeros(lanemask()))", value));
        }
        case CDialect_GLSL: {
            const Type* t = shd_get_unqualified_type(value_type);
            return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "subgroupBroadcastFirst(%s)", value));
        }
        default: term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "__shady_subgroup_first(%s)", value));
    }
}

static CTerm emit_primop(Emitter* emitter, FnEmitter* fn, Printer* p, const Node* node) {
    assert(node->tag == PrimOp_TAG);
    IrArena* arena = emitter->arena;
    const PrimOp* prim_op = &node->payload.prim_op;
    CTerm term = term_from_cvalue(shd_fmt_string_irarena(emitter->arena, "/* todo %s */", shd_get_primop_name(prim_op->op)));
    const ISelTableEntry* isel_entry = lookup_entry(emitter, prim_op->op);
    switch (prim_op->op) {
        case add_carry_op:
        case sub_borrow_op:
        case mul_extended_op:
            shd_error("TODO: implement extended arithm ops in C");
            break;
        // MATH OPS
        case fract_op: {
            CTerm floored = emit_using_entry(emitter, fn, p, lookup_entry(emitter, floor_op), prim_op->operands);
            term = term_from_cvalue(shd_format_string_arena(arena->arena, "1 - %s", shd_c_to_ssa(emitter, floored)));
            break;
        }
        case inv_sqrt_op: {
            CTerm floored = emit_using_entry(emitter, fn, p, lookup_entry(emitter, sqrt_op), prim_op->operands);
            term = term_from_cvalue(shd_format_string_arena(arena->arena, "1.0f / %s", shd_c_to_ssa(emitter, floored)));
            break;
        }
        case min_op: {
            CValue a = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, shd_first(prim_op->operands)));
            CValue b = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[1]));
            term = term_from_cvalue(shd_format_string_arena(arena->arena, "(%s > %s ? %s : %s)", a, b, b, a));
            break;
        }
        case max_op: {
            CValue a = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, shd_first(prim_op->operands)));
            CValue b = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[1]));
            term = term_from_cvalue(shd_format_string_arena(arena->arena, "(%s > %s ? %s : %s)", a, b, a, b));
            break;
        }
        case sign_op: {
            CValue src = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, shd_first(prim_op->operands)));
            term = term_from_cvalue(shd_format_string_arena(arena->arena, "(%s > 0 ? 1 : -1)", src));
            break;
        }
        case fma_op: {
            CValue a = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[0]));
            CValue b = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[1]));
            CValue c = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[2]));
            switch (emitter->config.dialect) {
                case CDialect_C11:
                case CDialect_CUDA: {
                    term = term_from_cvalue(shd_format_string_arena(arena->arena, "fmaf(%s, %s, %s)", a, b, c));
                    break;
                }
                default: {
                    term = term_from_cvalue(shd_format_string_arena(arena->arena, "(%s * %s) + %s", a, b, c));
                    break;
                }
            }
            break;
        }
        case lshift_op:
        case rshift_arithm_op:
        case rshift_logical_op: {
            CValue src = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, shd_first(prim_op->operands)));
            const Node* offset = prim_op->operands.nodes[1];
            CValue c_offset = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, offset));
            if (emitter->config.dialect == CDialect_GLSL) {
                if (shd_get_unqualified_type(offset->type)->payload.int_type.width == IntTy64)
                    c_offset = shd_format_string_arena(arena->arena, "int(%s)", c_offset);
            }
            term = term_from_cvalue(shd_format_string_arena(arena->arena, "(%s %s %s)", src, prim_op->op == lshift_op ? "<<" : ">>", c_offset));
            break;
        }
        case size_of_op:
            term = term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "sizeof(%s)", shd_c_emit_type(emitter, shd_first(prim_op->type_arguments), NULL)));
            break;
        case align_of_op:
            term = term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "alignof(%s)", shd_c_emit_type(emitter, shd_first(prim_op->type_arguments), NULL)));
            break;
        case offset_of_op: {
            const Type* t = shd_first(prim_op->type_arguments);
            while (t->tag == NominalType_TAG) {
                t = shd_get_nominal_type_body(t);
            }
            const Node* index = shd_first(prim_op->operands);
            uint64_t index_literal = shd_get_int_literal_value(*shd_resolve_to_int_literal(index), false);
            String member_name = shd_c_get_record_field_name(t, index_literal);
            term = term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "offsetof(%s, %s)", shd_c_emit_type(emitter, t, NULL), member_name));
            break;
        } case select_op: {
            assert(prim_op->operands.count == 3);
            CValue condition = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[0]));
            CValue l = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[1]));
            CValue r = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[2]));
            term = term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "(%s) ? (%s) : (%s)", condition, l, r));
            break;
        }
        case convert_op: {
            CTerm src = shd_c_emit_value(emitter, fn, shd_first(prim_op->operands));
            const Type* src_type = shd_get_unqualified_type(shd_first(prim_op->operands)->type);
            const Type* dst_type = shd_first(prim_op->type_arguments);
            if (emitter->config.dialect == CDialect_GLSL) {
                if (is_glsl_scalar_type(src_type) && is_glsl_scalar_type(dst_type)) {
                    CType t = shd_c_emit_type(emitter, dst_type, NULL);
                    term = term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "%s(%s)", t, shd_c_to_ssa(emitter, src)));
                } else
                    assert(false);
            } else {
                CType t = shd_c_emit_type(emitter, dst_type, NULL);
                term = term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "((%s) %s)", t, shd_c_to_ssa(emitter, src)));
            }
            break;
        }
        case reinterpret_op: {
            CTerm src_value = shd_c_emit_value(emitter, fn, shd_first(prim_op->operands));
            const Type* src_type = shd_get_unqualified_type(shd_first(prim_op->operands)->type);
            const Type* dst_type = shd_first(prim_op->type_arguments);
            switch (emitter->config.dialect) {
                case CDialect_CUDA:
                case CDialect_C11: {
                    String src = shd_make_unique_name(arena, "bitcast_src");
                    String dst = shd_make_unique_name(arena, "bitcast_result");
                    shd_print(p, "\n%s = %s;", shd_c_emit_type(emitter, src_type, src), shd_c_to_ssa(emitter, src_value));
                    shd_print(p, "\n%s;", shd_c_emit_type(emitter, dst_type, dst));
                    shd_print(p, "\nmemcpy(&%s, &%s, sizeof(%s));", dst, src, src);
                    return term_from_cvalue(dst);
                }
                // GLSL does not feature arbitrary casts, instead we need to run specialized conversion functions...
                case CDialect_GLSL: {
                    String conv_fn = NULL;
                    if (dst_type->tag == Float_TAG) {
                        assert(src_type->tag == Int_TAG);
                        switch (dst_type->payload.float_type.width) {
                            case FloatTy16: break;
                            case FloatTy32: conv_fn = src_type->payload.int_type.is_signed ? "intBitsToFloat" : "uintBitsToFloat";
                                break;
                            case FloatTy64: break;
                        }
                    } else if (dst_type->tag == Int_TAG) {
                        if (src_type->tag == Int_TAG) {
                            if (!emitter->config.explicitly_sized_types)
                                conv_fn = dst_type->payload.int_type.is_signed ? "int" : "uint";
                            else
                                conv_fn = shd_c_emit_type(emitter, dst_type, NULL);
                        } else {
                            assert(src_type->tag == Float_TAG);
                            switch (src_type->payload.float_type.width) {
                                case FloatTy16: break;
                                case FloatTy32: conv_fn = dst_type->payload.int_type.is_signed ? "floatBitsToInt" : "floatBitsToUint";
                                    break;
                                case FloatTy64: break;
                            }
                        }
                    }
                    if (conv_fn) {
                        return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "%s(%s)", conv_fn, shd_c_to_ssa(emitter, src_value)));
                    }
                    shd_error_print("glsl: unsupported bit cast from ");
                    shd_log_node(ERROR, src_type);
                    shd_error_print(" to ");
                    shd_log_node(ERROR, dst_type);
                    shd_error_print(".\n");
                    shd_error_die();
                }
                case CDialect_ISPC: {
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
                        return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "%s(%s)", n, shd_c_to_ssa(emitter, src_value)));
                    } else if (src_type->tag == Float_TAG) {
                        assert(dst_type->tag == Int_TAG);
                        return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "intbits(%s)", shd_c_to_ssa(emitter, src_value)));
                    }

                    CType t = shd_c_emit_type(emitter, dst_type, NULL);
                    return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "((%s) %s)", t, shd_c_to_ssa(emitter, src_value)));
                }
            }
            SHADY_UNREACHABLE;
        }
        case insert_op:
        case extract_dynamic_op:
        case extract_op: {
            CValue acc = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, shd_first(prim_op->operands)));
            bool insert = prim_op->op == insert_op;

            if (insert) {
                String dst = shd_make_unique_name(arena, "modified");
                shd_print(p, "\n%s = %s;", shd_c_emit_type(emitter, node->type, dst), acc);
                acc = dst;
                term = term_from_cvalue(dst);
            }

            const Type* t = shd_get_unqualified_type(shd_first(prim_op->operands)->type);
            for (size_t i = (insert ? 2 : 1); i < prim_op->operands.count; i++) {
                const Node* index = prim_op->operands.nodes[i];
                const IntLiteral* static_index = shd_resolve_to_int_literal(index);

                switch (is_type(t)) {
                    case NominalType_TAG: {
                        t = t->payload.nom_type.body;
                        SHADY_FALLTHROUGH
                    }
                    case Type_RecordType_TAG: {
                        assert(static_index);
                        Strings names = t->payload.record_type.names;
                        if (names.count == 0)
                            acc = shd_format_string_arena(emitter->arena->arena, "(%s._%d)", acc, static_index->value);
                        else
                            acc = shd_format_string_arena(emitter->arena->arena, "(%s.%s)", acc, names.strings[static_index->value]);
                        break;
                    }
                    case Type_PackType_TAG: {
                        assert(static_index);
                        assert(static_index->value < 4 && static_index->value < t->payload.pack_type.width);
                        String suffixes = "xyzw";
                        acc = shd_format_string_arena(emitter->arena->arena, "(%s.%c)", acc, suffixes[static_index->value]);
                        break;
                    }
                    case Type_ArrType_TAG: {
                        acc = index_into_array(emitter, t, term_from_cvar(acc), shd_c_emit_value(emitter, fn, index));
                        break;
                    }
                    default:
                    case NotAType: shd_error("Must be a type");
                }
            }

            if (insert) {
                shd_print(p, "\n%s = %s;", acc, shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[1])));
                break;
            }

            term = term_from_cvalue(acc);
            break;
        }
        case shuffle_op: {
            String dst = shd_make_unique_name(arena, "shuffled");
            const Node* lhs = prim_op->operands.nodes[0];
            const Node* rhs = prim_op->operands.nodes[1];
            String lhs_e = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[0]));
            String rhs_e = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[1]));
            const Type* lhs_t = lhs->type;
            const Type* rhs_t = rhs->type;
            shd_deconstruct_qualified_type(&lhs_t);
            shd_deconstruct_qualified_type(&rhs_t);
            size_t left_size = lhs_t->payload.pack_type.width;
            // size_t total_size = lhs_t->payload.pack_type.width + rhs_t->payload.pack_type.width;
            String suffixes = "xyzw";
            shd_print(p, "\n%s = vec%d(", shd_c_emit_type(emitter, node->type, dst), prim_op->operands.count - 2);
            for (size_t i = 2; i < prim_op->operands.count; i++) {
                const IntLiteral* selector = shd_resolve_to_int_literal(prim_op->operands.nodes[i]);
                if (selector->value < left_size)
                    shd_print(p, "%s.%c\n", lhs_e, suffixes[selector->value]);
                else
                    shd_print(p, "%s.%c\n", rhs_e, suffixes[selector->value - left_size]);
                if (i + 1 < prim_op->operands.count)
                    shd_print(p, ", ");
            }
            shd_print(p, ");\n");
            term = term_from_cvalue(dst);
            break;
        }
        case subgroup_assume_uniform_op: {
            if (emitter->config.dialect == CDialect_ISPC) {
                CValue value = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[0]));
                return broadcast_first(emitter, value, prim_op->operands.nodes[0]->type);
            }
            return shd_c_emit_value(emitter, fn, prim_op->operands.nodes[0]);
        }
        case sample_texture_op: {
            String sampler = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[0]));
            String coords = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, prim_op->operands.nodes[1]));

            String dst = shd_make_unique_name(arena, "sampled");
            String dim = "";
            if (emitter->config.glsl_version < 130) {
                const Type* t = prim_op->operands.nodes[0]->type;
                assert(t->tag == SampledImageType_TAG);
                t = t->payload.sampled_image_type.image_type;
                assert(t->tag == ImageType_TAG);
                dim = shd_c_emit_dim(t->payload.image_type.dim);
            }
            shd_print(p, "\n%s = texture%s(%s, %s);", shd_c_emit_type(emitter, node->type, dst), dim, sampler, coords);
            term = term_from_cvalue(dst);
            break;
        }
        case empty_mask_op:
        case mask_is_thread_active_op: shd_error("lower_me");
        default: break;
        case PRIMOPS_COUNT: assert(false); break;
    }

    if (isel_entry->isel_mechanism != IsNone)
        return emit_using_entry(emitter, fn, p, isel_entry, prim_op->operands);

    return term;
}

typedef struct {
    String set;
    SpvOp op;
    size_t prefix_len;
    uint32_t* prefix;
} ExtISelPattern;

#define mk_prefix(...) .prefix_len = sizeof((uint32_t[]) {__VA_ARGS__}) / sizeof(uint32_t), .prefix = (uint32_t[]) {__VA_ARGS__}
#define subgroup_reduction mk_prefix(SpvScopeSubgroup, SpvGroupOperationReduce)

typedef struct {
    ExtISelPattern match;
    ISelTableEntry payload;
} ExtISelEntry;

ExtISelEntry ext_isel_ispc_entries[] = {
    // reduce add
    {{ "spirv.core", SpvOpGroupIAdd, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_add" }},
    {{ "spirv.core", SpvOpGroupFAdd, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_add" }},
    {{ "spirv.core", SpvOpGroupNonUniformIAdd, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_add" }},
    {{ "spirv.core", SpvOpGroupNonUniformFAdd, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_add" }},
    // min
    {{ "spirv.core", SpvOpGroupSMin, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_min" }},
    {{ "spirv.core", SpvOpGroupUMin, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_min" }},
    {{ "spirv.core", SpvOpGroupFMin, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_min" }},
    {{ "spirv.core", SpvOpGroupNonUniformSMin, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_min" }},
    {{ "spirv.core", SpvOpGroupNonUniformUMin, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_min" }},
    {{ "spirv.core", SpvOpGroupNonUniformFMin, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_min" }},
    // max
    {{ "spirv.core", SpvOpGroupSMax, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_max" }},
    {{ "spirv.core", SpvOpGroupUMax, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_max" }},
    {{ "spirv.core", SpvOpGroupFMax, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_max" }},
    {{ "spirv.core", SpvOpGroupNonUniformSMax, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_max" }},
    {{ "spirv.core", SpvOpGroupNonUniformUMax, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_max" }},
    {{ "spirv.core", SpvOpGroupNonUniformFMax, subgroup_reduction }, { IsMono, OsCall, .op = "reduce_max" }},
    // rest
    {{ "spirv.core", SpvOpGroupNonUniformAllEqual, mk_prefix(SpvScopeSubgroup) }, { IsMono, OsCall, .op = "reduce_equal" }},
    {{ "spirv.core", SpvOpGroupNonUniformBallot, mk_prefix(SpvScopeSubgroup) }, { IsMono, OsCall, .op = "packmask" }},
};

ExtISelEntry ext_isel_glsl_entries[] = {
    // reduce add
    {{ "spirv.core", SpvOpGroupIAdd, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupAdd" }},
    {{ "spirv.core", SpvOpGroupFAdd, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupAdd" }},
    {{ "spirv.core", SpvOpGroupNonUniformIAdd, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupAdd" }},
    {{ "spirv.core", SpvOpGroupNonUniformFAdd, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupAdd" }},
    // min
    {{ "spirv.core", SpvOpGroupSMin, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMin" }},
    {{ "spirv.core", SpvOpGroupUMin, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMin" }},
    {{ "spirv.core", SpvOpGroupFMin, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMin" }},
    {{ "spirv.core", SpvOpGroupNonUniformSMin, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMin" }},
    {{ "spirv.core", SpvOpGroupNonUniformUMin, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMin" }},
    {{ "spirv.core", SpvOpGroupNonUniformFMin, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMin" }},
    // max
    {{ "spirv.core", SpvOpGroupSMax, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMax" }},
    {{ "spirv.core", SpvOpGroupUMax, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMax" }},
    {{ "spirv.core", SpvOpGroupFMax, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMax" }},
    {{ "spirv.core", SpvOpGroupNonUniformSMax, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMax" }},
    {{ "spirv.core", SpvOpGroupNonUniformUMax, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMax" }},
    {{ "spirv.core", SpvOpGroupNonUniformFMax, subgroup_reduction }, { IsMono, OsCall, .op = "subgroupMax" }},
    // rest
    {{ "spirv.core", SpvOpGroupNonUniformAllEqual, mk_prefix(SpvScopeSubgroup) }, { IsMono, OsCall, .op = "subgroupAllEqual" }},
    {{ "spirv.core", SpvOpGroupNonUniformBallot, mk_prefix(SpvScopeSubgroup) }, { IsMono, OsCall, .op = "subgroupBallot" }},

    {{ "spirv.core", SpvOpGroupNonUniformBroadcastFirst, mk_prefix(SpvScopeSubgroup) }, { IsMono, OsCall, .op = "subgroupBroadcastFirst" }},
    {{ "spirv.core", SpvOpGroupNonUniformElect, mk_prefix(SpvScopeSubgroup) }, { IsMono, OsCall, .op = "subgroupElect" }},

};

ExtISelEntry ext_isel_entries[] = {
    {{ "spirv.core", SpvOpGroupNonUniformElect, mk_prefix(SpvScopeSubgroup) }, { IsMono, OsCall, .op = "__shady_elect_first" }},
};

static bool check_ext_entry(const ExtISelPattern* entry, ExtInstr instr) {
    if (strcmp(entry->set, instr.set) != 0 || entry->op != instr.opcode)
        return false;
    for (size_t i = 0; i < entry->prefix_len; i++) {
        if (i >= instr.operands.count)
            return false;
        const IntLiteral* lit = shd_resolve_to_int_literal(instr.operands.nodes[i]);
        if (!lit)
            return false;
        if (shd_get_int_literal_value(*lit, false) != entry->prefix[i])
            return false;
    }
    return true;
}

static const ExtISelEntry* find_ext_entry_in_list(const ExtISelEntry table[], size_t size, ExtInstr instr) {
    for (size_t i = 0; i < size; i++) {
        if (check_ext_entry(&table[i].match, instr))
            return &table[i];
    }
    return NULL;
}

#define scan_entries(name) { const ExtISelEntry* f = find_ext_entry_in_list(name, sizeof(name) / sizeof(name[0]), instr); if (f) return f; }

static const ExtISelEntry* find_ext_entry(Emitter* e, ExtInstr instr) {
    switch (e->config.dialect) {
        case CDialect_ISPC: scan_entries(ext_isel_ispc_entries); break;
        case CDialect_GLSL: scan_entries(ext_isel_glsl_entries); break;
        default: break;
    }
    scan_entries(ext_isel_entries);
    return NULL;
}

static CTerm emit_ext_instruction(Emitter* emitter, FnEmitter* fn, Printer* p, ExtInstr instr) {
    shd_c_emit_mem(emitter, fn, instr.mem);
    if (strcmp(instr.set, "spirv.core") == 0) {
        switch (instr.opcode) {
            case SpvOpGroupNonUniformBroadcastFirst: {
                assert(instr.operands.count == 2);
                CValue value = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, instr.operands.nodes[1]));
                return broadcast_first(emitter, value, instr.operands.nodes[1]->type);
            }
            case SpvOpGroupNonUniformElect: {
                assert(instr.operands.count == 1);
                const IntLiteral* scope = shd_resolve_to_int_literal(shd_first(instr.operands));
                assert(scope && scope->value == SpvScopeSubgroup);
                switch (emitter->config.dialect) {
                    case CDialect_ISPC: return term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "(programIndex == count_trailing_zeros(lanemask()))"));
                    default: break;
                }
                break;
            }
            default: break;
        }
    }

    const ExtISelEntry* entry = find_ext_entry(emitter, instr);
    if (entry) {
        Nodes operands = instr.operands;
        if (entry->match.prefix_len > 0)
            operands = shd_nodes(emitter->arena, operands.count - entry->match.prefix_len, &operands.nodes[entry->match.prefix_len]);
        return emit_using_entry(emitter, fn, p, &entry->payload, operands);
    } else {
        shd_error("Unsupported extended instruction: (set = %s, opcode = %d )", instr.set, instr.opcode);
    }
}

static CTerm emit_call(Emitter* emitter, FnEmitter* fn, Printer* p, const Node* callee, Nodes args, const Type* result_type) {
    Growy* g = shd_new_growy();
    Printer* paramsp = shd_new_printer_from_growy(g);
    if (emitter->use_private_globals) {
        shd_print(paramsp, "__shady_private_globals");
        if (args.count > 0)
            shd_print(paramsp, ", ");
    }
    for (size_t i = 0; i < args.count; i++) {
        shd_print(paramsp, shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, args.nodes[i])));
        if (i + 1 < args.count)
            shd_print(paramsp, ", ");
    }

    CValue e_callee;
    if (callee->tag == Function_TAG)
        e_callee = shd_c_to_ssa(emitter, shd_c_emit_function(emitter, callee));
    else
        e_callee = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, callee));

    String params = shd_printer_growy_unwrap(paramsp);

    CTerm called = term_from_cvalue(shd_format_string_arena(emitter->arena->arena, "\n%s(%s)", e_callee, params));
    called = shd_c_bind_intermediary_result(emitter, p, result_type, called);

    free_tmp_str(params);
    return called;
}

static CTerm emit_ptr_composite_element(Emitter* emitter, FnEmitter* fn, Printer* p, PtrCompositeElement lea) {
    IrArena* arena = emitter->arena;
    CTerm acc = shd_c_emit_value(emitter, fn, lea.ptr);

    const Type* src_qtype = lea.ptr->type;
    ShdScope scope = shd_get_qualified_type_scope(src_qtype);
    const Type* curr_ptr_type = shd_get_unqualified_type(src_qtype);
    assert(curr_ptr_type->tag == PtrType_TAG);

    const Type* pointee_type = shd_get_pointer_type_element(curr_ptr_type);
    const Node* selector = lea.index;
    scope = shd_combine_scopes(scope, shd_get_qualified_type_scope(selector->type));
    switch (is_type(pointee_type)) {
        case ArrType_TAG: {
            CTerm index = shd_c_emit_value(emitter, fn, selector);
            acc = term_from_cvar(index_into_array(emitter, pointee_type, acc, index));
            curr_ptr_type = ptr_type(arena, (PtrType) {
                    .pointed_type = pointee_type->payload.arr_type.element_type,
                    .address_space = curr_ptr_type->payload.ptr_type.address_space
            });
            break;
        }
        case NominalType_TAG: {
            pointee_type = shd_get_nominal_type_body(pointee_type);
            SHADY_FALLTHROUGH
        }
        case RecordType_TAG: {
            // yet another ISPC bug and workaround
            // ISPC cannot deal with subscripting if you've done pointer arithmetic (!) inside the expression
            // so hum we just need to introduce a temporary variable to hold the pointer expression so far, and go again from there
            // See https://github.com/ispc/ispc/issues/2496
            if (emitter->config.dialect == CDialect_ISPC) {
                String interm = shd_make_unique_name(arena, "lea_intermediary_ptr_value");
                shd_print(p, "\n%s = %s;", shd_c_emit_type(emitter, qualified_type_helper(arena, scope, curr_ptr_type), interm), shd_c_to_ssa(emitter, acc));
                acc = term_from_cvalue(interm);
            }

            assert(selector->tag == IntLiteral_TAG && "selectors when indexing into a record need to be constant");
            size_t static_index = shd_get_int_literal_value(*shd_resolve_to_int_literal(selector), false);
            String field_name = shd_c_get_record_field_name(pointee_type, static_index);
            acc = term_from_cvar(shd_format_string_arena(arena->arena, "(%s.%s)", shd_c_deref(emitter, acc), field_name));
            curr_ptr_type = ptr_type(arena, (PtrType) {
                    .pointed_type = pointee_type->payload.record_type.members.nodes[static_index],
                    .address_space = curr_ptr_type->payload.ptr_type.address_space
            });
            break;
        }
        case Type_PackType_TAG: {
            size_t static_index = shd_get_int_literal_value(*shd_resolve_to_int_literal(selector), false);
            String suffixes = "xyzw";
            acc = term_from_cvar(shd_format_string_arena(emitter->arena->arena, "(%s.%c)", shd_c_deref(emitter, acc), suffixes[static_index]));
            curr_ptr_type = ptr_type(arena, (PtrType) {
                    .pointed_type = pointee_type->payload.pack_type.element_type,
                    .address_space = curr_ptr_type->payload.ptr_type.address_space
            });
            break;
        }
        default: shd_error("lea can't work on this");
    }

    // if (emitter->config.dialect == CDialect_ISPC)
    //     acc = c_bind_intermediary_result(emitter, p, curr_ptr_type, acc);

    return acc;
}

static CTerm emit_ptr_array_element_offset(Emitter* emitter, FnEmitter* fn, Printer* p, PtrArrayElementOffset lea) {
    IrArena* arena = emitter->arena;
    CTerm acc = shd_c_emit_value(emitter, fn, lea.ptr);

    const Type* src_qtype = lea.ptr->type;
    ShdScope scope = shd_get_qualified_type_scope(src_qtype);
    const Type* curr_ptr_type = shd_get_unqualified_type(src_qtype);
    assert(curr_ptr_type->tag == PtrType_TAG);

    const IntLiteral* offset_static_value = shd_resolve_to_int_literal(lea.offset);
    if (!offset_static_value || offset_static_value->value != 0) {
        CTerm offset = shd_c_emit_value(emitter, fn, lea.offset);
        // we sadly need to drop to the value level (aka explicit pointer arithmetic) to do this
        // this means such code is never going to be legal in GLSL
        // also the cast is to account for our arrays-in-structs hack
        const Type* pointee_type = shd_get_pointer_type_element(curr_ptr_type);
        acc = term_from_cvalue(shd_format_string_arena(arena->arena, "((%s) &(%s)[%s])", shd_c_emit_type(emitter, curr_ptr_type, NULL), shd_c_to_ssa(emitter, acc), shd_c_to_ssa(emitter, offset)));
        scope = shd_combine_scopes(scope, shd_get_qualified_type_scope(lea.offset->type));
    }

    if (emitter->config.dialect == CDialect_ISPC)
        acc = shd_c_bind_intermediary_result(emitter, p, qualified_type_helper(emitter->arena, scope, curr_ptr_type), acc);

    return acc;
}

static const Type* get_allocated_type(const Node* alloc) {
    switch (alloc->tag) {
        case Instruction_StackAlloc_TAG: return alloc->payload.stack_alloc.type;
        case Instruction_LocalAlloc_TAG: return alloc->payload.local_alloc.type;
        default: assert(false); return NULL;
    }
}

static CTerm emit_alloca(Emitter* emitter, Printer* p, const Type* instr) {
    String variable_name = shd_make_unique_name(emitter->arena, "alloca");
    CTerm variable = (CTerm) { .value = NULL, .var = variable_name };
    shd_c_emit_variable_declaration(emitter, p, get_allocated_type(instr), variable_name, true, NULL);
    const Type* ptr_type = instr->type;
    shd_deconstruct_qualified_type(&ptr_type);
    assert(ptr_type->tag == PtrType_TAG);
    return variable;
}

static CTerm emit_instruction(Emitter* emitter, FnEmitter* fn, Printer* p, const Node* instruction) {
    assert(is_instruction(instruction));
    IrArena* a = emitter->arena;

    switch (is_instruction(instruction)) {
        case NotAnInstruction: assert(false);
        case Instruction_PushStack_TAG:
        case Instruction_PopStack_TAG:
        case Instruction_GetStackSize_TAG:
        case Instruction_SetStackSize_TAG:
        case Instruction_GetStackBaseAddr_TAG: shd_error("Stack operations need to be lowered.");
        case Instruction_ExtInstr_TAG: return emit_ext_instruction(emitter, fn, p, instruction->payload.ext_instr);
        case Instruction_PrimOp_TAG: return shd_c_bind_intermediary_result(emitter, p, instruction->type, emit_primop(emitter, fn, p, instruction));
        case Instruction_Call_TAG: {
            Call payload = instruction->payload.call;
            shd_c_emit_mem(emitter, fn, payload.mem);
            return emit_call(emitter, fn, p, payload.callee, payload.args, instruction->type);
        } case Instruction_IndirectCall_TAG: {
            IndirectCall payload = instruction->payload.indirect_call;
            shd_c_emit_mem(emitter, fn, payload.mem);
            return emit_call(emitter, fn, p, payload.callee, payload.args, instruction->type);
        } case Instruction_Comment_TAG: shd_print(p, "/* %s */", instruction->payload.comment.string); return empty_term();
        case Instruction_StackAlloc_TAG: shd_c_emit_mem(emitter, fn, instruction->payload.local_alloc.mem); return emit_alloca(emitter, p, instruction);
        case Instruction_LocalAlloc_TAG: shd_c_emit_mem(emitter, fn, instruction->payload.local_alloc.mem); return emit_alloca(emitter, p, instruction);
        case Instruction_PtrArrayElementOffset_TAG: return emit_ptr_array_element_offset(emitter, fn, p, instruction->payload.ptr_array_element_offset);
        case Instruction_PtrCompositeElement_TAG: return emit_ptr_composite_element(emitter, fn, p, instruction->payload.ptr_composite_element);
        case Instruction_Load_TAG: {
            Load payload = instruction->payload.load;
            shd_c_emit_mem(emitter, fn, payload.mem);
            CAddr dereferenced = shd_c_deref(emitter, shd_c_emit_value(emitter, fn, payload.ptr));
            if (shd_get_unqualified_type(payload.ptr->type)->payload.ptr_type.address_space == AsUniformConstant)
                return term_from_cvalue(dereferenced);
            return shd_c_bind_intermediary_result(emitter, p, instruction->type, term_from_cvalue(dereferenced));
        }
        case Instruction_Store_TAG: {
            Store payload = instruction->payload.store;
            shd_c_emit_mem(emitter, fn, payload.mem);
            const Type* addr_type = payload.ptr->type;
            ShdScope as_scope = shd_get_addr_space_scope(addr_type->payload.ptr_type.address_space);
            ShdScope value_scope = shd_get_qualified_type_scope(payload.value->type);
            ShdScope addr_scope = shd_deconstruct_qualified_type(&addr_type);
            assert(addr_type->tag == PtrType_TAG);
            CAddr dereferenced = shd_c_deref(emitter, shd_c_emit_value(emitter, fn, payload.ptr));
            CValue cvalue = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, payload.value));
            // ISPC lets you broadcast to a uniform address space iff the address is non-uniform, otherwise we need to do this
            bool addr_uniform = addr_scope <= ShdScopeSubgroup;
            bool as_uniform = as_scope <= ShdScopeSubgroup;
            bool value_uniform = value_scope <= ShdScopeSubgroup;
            if (emitter->config.dialect == CDialect_ISPC && addr_uniform && as_uniform && !value_uniform)
                cvalue = shd_format_string_arena(emitter->arena->arena, "__shady_extract(%s, count_trailing_zeros(lanemask()))", cvalue);

            shd_print(p, "\n%s = %s;", dereferenced, cvalue);
            return empty_term();
        }
        case Instruction_CopyBytes_TAG: {
            CopyBytes payload = instruction->payload.copy_bytes;
            shd_c_emit_mem(emitter, fn, payload.mem);
            shd_print(p, "\nmemcpy(%s, %s, %s);", shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, payload.dst)), shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, payload.src)), shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, payload.count)));
            return empty_term();
        }
        case Instruction_FillBytes_TAG:{
            FillBytes payload = instruction->payload.fill_bytes;
            shd_c_emit_mem(emitter, fn, payload.mem);
            shd_print(p, "\nmemset(%s, %s, %s);", shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, payload.dst)), shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, payload.src)), shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, payload.count)));
            return empty_term();
        }
        case Instruction_DebugPrintf_TAG: {
            DebugPrintf payload = instruction->payload.debug_printf;
            shd_c_emit_mem(emitter, fn, payload.mem);
            Printer* args_printer = shd_new_printer_from_growy(shd_new_growy());
            shd_print(args_printer, "\"");
            shd_printer_unescape(args_printer, instruction->payload.debug_printf.string);
            shd_print(args_printer, "\"");
            for (size_t i = 0; i < instruction->payload.debug_printf.args.count; i++) {
                CValue str = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, instruction->payload.debug_printf.args.nodes[i]));

                // special casing for the special child
                if (emitter->config.dialect == CDialect_ISPC)
                    shd_print(args_printer, ", __shady_extract(%s, printf_thread_index)", str);
                else
                    shd_print(args_printer, ", %s", str);
            }
            String args_list = shd_printer_growy_unwrap(args_printer);
            switch (emitter->config.dialect) {
                case CDialect_ISPC:
                    shd_print(p, "\nforeach_active(printf_thread_index) { print(%s); }", args_list);
                    break;
                case CDialect_CUDA:
                case CDialect_C11:
                    shd_print(p, "\nprintf(%s);", args_list);
                    break;
                case CDialect_GLSL: {
                    if (emitter->config.glsl_version < 460) {
                        static bool flag = false;
                        shd_warn_print_once(flag, "Warning: printf is not supported in GLSL\n");
                    } else {
                        shd_print(p, "\ndebugPrintfEXT(%s);", args_list);
                    }
                    break;
                }
            }
            free((char*) args_list);

            return empty_term();
        }
    }
    
    SHADY_UNREACHABLE;
}

static bool can_appear_at_top_level(Emitter* emitter, const Node* node) {
    if (is_instruction(node))
        return false;
    // if (emitter->config.dialect == CDialect_ISPC) {
    //     if (node->tag == GlobalVariable_TAG)
    //         if (!shd_is_addr_space_uniform(emitter->arena, node->payload.global_variable.address_space))
    //             //if (is_value(node) && !is_qualified_type_uniform(node->type))
    //                 return false;
    // }
    return true;
}

CTerm shd_c_emit_value(Emitter* emitter, FnEmitter* fn_builder, const Node* node) {
    CTerm* found = shd_c_lookup_existing_term(emitter, fn_builder, node);
    if (found) return *found;

    CFNode* where = fn_builder ? shd_schedule_instruction(fn_builder->scheduler, node) : NULL;
    if (where) {
        CTerm emitted = c_emit_value_(emitter, fn_builder, fn_builder->instruction_printers[where->rpo_index], node);
        shd_c_register_emitted(emitter, fn_builder, node, emitted);
        return emitted;
    } else if (!can_appear_at_top_level(emitter, node)) {
        if (!fn_builder) {
            shd_log_node(ERROR, node);
            shd_log_fmt(ERROR, "cannot appear at top-level");
            exit(-1);
        }
        // Pick the entry block of the current fn
        CTerm emitted = c_emit_value_(emitter, fn_builder, fn_builder->instruction_printers[0], node);
        shd_c_register_emitted(emitter, fn_builder, node, emitted);
        return emitted;
    } else {
        assert(!is_mem(node));
        CTerm emitted = c_emit_value_(emitter, NULL, NULL, node);
        shd_c_register_emitted(emitter, NULL, node, emitted);
        return emitted;
    }
}

CTerm shd_c_emit_mem(Emitter* e, FnEmitter* b, const Node* mem) {
    assert(is_mem(mem));
    if (mem->tag == AbsMem_TAG)
        return empty_term();
    if (is_instruction(mem))
        return shd_c_emit_value(e, b, mem);
    shd_error("What sort of mem is this ?");
}
