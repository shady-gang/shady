#include "parser.h"
#include "token.h"
#include "SlimFrontendOps.h"

#include "shady/ir/ext.h"

#include "list.h"
#include "portability.h"
#include "log.h"
#include "util.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

static int max_precedence() {
    return 10;
}

static int get_precedence(InfixOperators op) {
    switch (op) {
#define INFIX_OPERATOR(name, token, primop_op, precedence) case Infix##name: return precedence;
INFIX_OPERATORS()
#undef INFIX_OPERATOR
        default: shd_error("unknown operator");
    }
}
static bool is_primop_op(InfixOperators op, Op* out) {
    switch (op) {
#define INFIX_OPERATOR(name, token, primop_op, precedence) case Infix##name: if (primop_op != -1) { *out = primop_op; return true; } else return false;
INFIX_OPERATORS()
#undef INFIX_OPERATOR
        default: shd_error("unknown operator");
    }
}

static bool is_infix_operator(TokenTag token_tag, InfixOperators* out) {
    switch (token_tag) {
#define INFIX_OPERATOR(name, token, primop_op, precedence) case token: { *out = Infix##name; return true; }
INFIX_OPERATORS()
#undef INFIX_OPERATOR
        default: return false;
    }
}

// to avoid some repetition
#define ctxparams SHADY_UNUSED const SlimParserConfig* config, SHADY_UNUSED const char* contents, SHADY_UNUSED Module* mod, SHADY_UNUSED IrArena* arena, SHADY_UNUSED Tokenizer* tokenizer
#define ctx config, contents, mod, arena, tokenizer

static void error_with_loc(ctxparams) {
    Loc loc = shd_current_loc(tokenizer);
    size_t startline = loc.line - 2;
    if (startline < 1) startline = 1;
    size_t endline = startline + 5;

    int numdigits = 1;
    int e = endline;
    while (e >= 10) {
        numdigits++;
        e /= 10;
    }
    LARRAY(char, digits, numdigits);
    // char* digits = malloc(sizeof(char) * numdigits);

    size_t line = 1;
    size_t len = strlen(contents);
    for (size_t i = 0; i < len; i++) {
        if (line >= startline && line <= endline) {
            shd_log_fmt(ERROR, "%c", contents[i]);
        }
        if (contents[i] == '\n') {
            if (line == loc.line) {
                for (size_t digit = 0; digit < numdigits; digit++) {
                    shd_log_fmt(ERROR, " ");
                }
                shd_log_fmt(ERROR, "  ");
                for (size_t j = 1; j < loc.column; j++) {
                    shd_log_fmt(ERROR, " ");
                }
                shd_log_fmt(ERROR, "^");
                shd_log_fmt(ERROR, "\n");
            }
            line++;

            if (line >= startline && line <= endline) {
                size_t l = line, digit;
                for (digit = 0; digit < numdigits; digit++) {
                    if (l == 0)
                        break;
                    digits[numdigits - 1 - digit] = (char) ('0' + (l % 10));
                    l /= 10;
                }
                for (; digit < numdigits; digit++) {
                    digits[numdigits - 1 - digit] = (char) ' ';
                }
                for (digit = 0; digit < numdigits; digit++) {
                    shd_log_fmt(ERROR, "%c", digits[numdigits - 1 - digit]);
                }
                shd_log_fmt(ERROR, ": ");
            }
        }
    }
    shd_log_fmt(ERROR, "At %d:%d, ", loc.line, loc.column);
}

#define syntax_error(condition) syntax_error_impl(ctx, condition)
#define syntax_error_fmt(condition, ...) syntax_error_impl(ctx, condition, __VA_ARGS__)
static void syntax_error_impl(ctxparams, const char* format, ...) {
    va_list args;
    va_start(args, format);
    error_with_loc(ctx);
    shd_log_fmt_va_list(ERROR, format, args);
    shd_log_fmt(ERROR, "\n");
    exit(-4);
    va_end(args);
}

#define expect(condition, format, ...) expect_impl(ctx, condition, format)
#define expect_fmt(condition, format, ...) expect_impl(ctx, condition, format, __VA_ARGS__)
static void expect_impl(ctxparams, bool condition, const char* format, ...) {
    if (!condition) {
        va_list args;
        va_start(args, format);
        error_with_loc(ctx);
        shd_log_fmt(ERROR, "expected ");
        shd_log_fmt_va_list(ERROR, format, args);
        shd_log_fmt(ERROR, "\n");
        exit(-4);
        va_end(args);
    }
}

static bool accept_token(ctxparams, TokenTag tag) {
    if (shd_curr_token(tokenizer).tag == tag) {
        shd_next_token(tokenizer);
        return true;
    }
    return false;
}

static const char* accept_identifier(ctxparams) {
    Token tok = shd_curr_token(tokenizer);
    if (tok.tag == identifier_tok) {
        shd_next_token(tokenizer);
        size_t size = tok.end - tok.start;
        return shd_string_sized(arena, (int) size, &contents[tok.start]);
    }
    return NULL;
}

static const Node* expect_body(ctxparams, const Node* mem, const Node* default_terminator(const Node*));
static const Node* accept_value(ctxparams, BodyBuilder*);
static const Type* accept_unqualified_type(ctxparams);
static const Node* accept_expr(ctxparams, BodyBuilder*, int);
static Nodes expect_operands(ctxparams, BodyBuilder*);
static const Node* expect_operand(ctxparams, BodyBuilder*);

static const Type* accept_numerical_type(ctxparams) {
    if (accept_token(ctx, i8_tok)) {
        return shd_int8_type(arena);
    } else if (accept_token(ctx, i16_tok)) {
        return shd_int16_type(arena);
    } else if (accept_token(ctx, i32_tok)) {
        return shd_int32_type(arena);
    } else if (accept_token(ctx, i64_tok)) {
        return shd_int64_type(arena);
    } else if (accept_token(ctx, u8_tok)) {
        return shd_uint8_type(arena);
    } else if (accept_token(ctx, u16_tok)) {
        return shd_uint16_type(arena);
    } else if (accept_token(ctx, u32_tok)) {
        return shd_uint32_type(arena);
    } else if (accept_token(ctx, u64_tok)) {
        return shd_uint64_type(arena);
    } else if (accept_token(ctx, f16_tok)) {
        return shd_fp16_type(arena);
    } else if (accept_token(ctx, f32_tok)) {
        return shd_fp32_type(arena);
    } else if (accept_token(ctx, f64_tok)) {
        return shd_fp64_type(arena);
    }
    return NULL;
}

static const Node* accept_numerical_literal(ctxparams) {
    const Type* num_type = accept_numerical_type(ctx);

    bool negate = accept_token(ctx, minus_tok);

    Token tok = shd_curr_token(tokenizer);
    size_t size = tok.end - tok.start;
    String str = shd_string_sized(arena, (int) size, &contents[tok.start]);

    switch (tok.tag) {
        case hex_lit_tok:
            if (negate)
                syntax_error("hexadecimal literals can't start with '-'");
        case dec_lit_tok: {
            shd_next_token(tokenizer);
            break;
        }
        default: {
            if (negate || num_type)
                syntax_error("expected numerical literal");
            return NULL;
        }
    }

    if (negate) // add back the - in front
        str = shd_fmt_string_irarena(arena, "-%s", str);

    if (num_type) {
        // TODO: share this logic in infer.c
        if (num_type->tag == Int_TAG) {
            Int payload = num_type->payload.int_type;
            char* endptr;
            int64_t i = strtoll(str, &endptr, 10);
            return int_literal_helper(arena, payload.width, payload.is_signed, i);
        } else if (num_type->tag == Float_TAG) {
            Float payload = num_type->payload.float_type;
            uint64_t v;
            switch (payload.width) {
                case ShdFloatFormat16:
                shd_error("TODO: implement fp16 parsing");
                case ShdFloatFormat32:
                    assert(sizeof(float) == sizeof(uint32_t));
                    float f = strtof(str, NULL);
                    memcpy(&v, &f, sizeof(uint32_t));
                    break;
                case ShdFloatFormat64:
                    assert(sizeof(double) == sizeof(uint64_t));
                    double d = strtod(str, NULL);
                    memcpy(&v, &d, sizeof(uint64_t));
                    break;
            }
            return float_literal(arena, (FloatLiteral) { .value = v, .width = payload.width });
        } else {
            syntax_error("illegal type literal");
            return NULL;
        }
    } else {
        return untyped_number(arena, (UntypedNumber) {
                .plaintext = str
        });
    }
}

static bool accept_scope(ctxparams, ShdScope* out) {
    if (accept_token(ctx, uniform_tok))
        *out = config->target_config->scopes.gang;
    else if (accept_token(ctx, varying_tok))
        *out = config->target_config->scopes.bottom;
    else
        return false;
    return true;
}

static const Type* accept_maybe_qualified_type(ctxparams) {
    ShdScope scope;
    bool scope_specified = accept_scope(ctx, &scope);
    const Type* unqualified = accept_unqualified_type(ctx);
    if (!unqualified && !scope_specified)
        return NULL;
    expect(unqualified, "unqualified type");
    if (scope_specified)
        return qualified_type(arena, (QualifiedType) { .scope = scope, .type = unqualified });
    else
        return unqualified;
}

static const Type* accept_qualified_type(ctxparams) {
    ShdScope scope;
    expect(accept_scope(ctx, &scope), "frequency qualifier");
    const Type* unqualified = accept_unqualified_type(ctx);
    expect(unqualified, "unqualified type");
    return qualified_type(arena, (QualifiedType) { .scope = scope, .type = unqualified });
}

static Nodes accept_type_arguments(ctxparams) {
    Nodes ty_args = shd_empty(arena);
    if (accept_token(ctx, lsbracket_tok)) {
        while (true) {
            const Type* t = accept_unqualified_type(ctx);
            expect(t, "unqualified type");
            ty_args = shd_nodes_append(arena, ty_args, t);
            if (accept_token(ctx, comma_tok))
                continue;
            if (accept_token(ctx, rsbracket_tok))
                break;
        }
    }
    return ty_args;
}

static const Node* make_unbound(IrArena* a, const Node* mem, String identifier) {
    return ext_instr(a, (ExtInstr) {
        .mem = mem,
        .set = "shady.frontend",
        .opcode = SlimFrontendOpsSlimUnboundSHADY,
        .result_t = unit_type(a),
        .operands = shd_singleton(string_lit_helper(a, identifier)),
    });
}

static const Node* accept_value(ctxparams, BodyBuilder* bb) {
    Token tok = shd_curr_token(tokenizer);
    size_t size = tok.end - tok.start;

    const Node* number = accept_numerical_literal(ctx);
    if (number)
        return number;

    switch (tok.tag) {
        case identifier_tok: {
            const char* id = shd_string_sized(arena, (int) size, &contents[tok.start]);
            shd_next_token(tokenizer);

            Op op = PRIMOPS_COUNT;
            for (size_t i = 0; i < PRIMOPS_COUNT; i++) {
                if (strcmp(id, shd_get_primop_name(i)) == 0) {
                    op = i;
                    break;
                }
            }

            if (op != PRIMOPS_COUNT) {
                if (!bb)
                    syntax_error("primops cannot be used outside of a function");
                return shd_bld_add_instruction(bb, prim_op(arena, (PrimOp) {
                    .op = op,
                    .operands = expect_operands(ctx, bb)
                }));
            } else if (strcmp(id, "ext_instr") == 0) {
                expect(accept_token(ctx, lsbracket_tok), "'['");
                const Node* set = accept_value(ctx, NULL);
                expect(set->tag == StringLiteral_TAG, "string literal");
                expect(accept_token(ctx, comma_tok), "','");
                const Node* opcode = accept_value(ctx, NULL);
                expect(opcode->tag == UntypedNumber_TAG, "number");
                expect(accept_token(ctx, comma_tok), "','");
                const Type* type = accept_qualified_type(ctx);
                expect(type, "type");
                expect(accept_token(ctx, rsbracket_tok), "]");
                Nodes ops = expect_operands(ctx, bb);
                return shd_bld_add_instruction(bb, ext_instr(arena, (ExtInstr) {
                    .result_t = type,
                    .set = set->payload.string_lit.string,
                    .opcode = strtoll(opcode->payload.untyped_number.plaintext, NULL, 10),
                    .mem = shd_bld_mem(bb),
                    .operands = ops,
                }));
            } else if (strcmp(id, "alloca") == 0) {
                const Node* type = shd_first(accept_type_arguments(ctx));
                Nodes ops = expect_operands(ctx, bb);
                expect(ops.count == 0, "no operands");
                return shd_bld_add_instruction(bb, stack_alloc(arena, (StackAlloc) {
                    .type = type,
                    .mem = shd_bld_mem(bb),
                }));
            } else if (strcmp(id, "bitcast") == 0) {
                const Node* type = shd_first(accept_type_arguments(ctx));
                Nodes ops = expect_operands(ctx, bb);
                expect(ops.count == 1, "one operands");
                return bit_cast_helper(arena, type, shd_first(ops));
            } else if (strcmp(id, "convert") == 0) {
                const Node* type = shd_first(accept_type_arguments(ctx));
                Nodes ops = expect_operands(ctx, bb);
                expect(ops.count == 1, "one operands");
                return conversion_helper(arena, type, shd_first(ops));
            } else if (strcmp(id, "generic_ptr_cast") == 0) {
                Nodes ops = expect_operands(ctx, bb);
                expect(ops.count == 1, "one operands");
                return generic_ptr_cast_helper(arena, shd_first(ops));
            } else if (strcmp(id, "ptr_array_offset") == 0) {
                Nodes ops = expect_operands(ctx, bb);
                expect(ops.count == 2, "one operands");
                return ptr_array_element_offset_helper(arena, shd_first(ops), ops.nodes[1]);
            } else if (strcmp(id, "ptr_composite_offset") == 0) {
                Nodes ops = expect_operands(ctx, bb);
                expect(ops.count == 2, "one operands");
                return ptr_composite_element_helper(arena, shd_first(ops), ops.nodes[1]);
            } else if (strcmp(id, "debug_printf") == 0) {
                Nodes ops = expect_operands(ctx, bb);
                return shd_bld_add_instruction(bb, debug_printf(arena, (DebugPrintf) {
                    .string = shd_get_string_literal(arena, shd_first(ops)),
                    .args = shd_nodes(arena, ops.count - 1, &ops.nodes[1]),
                    .mem = shd_bld_mem(bb),
                }));
            }

            if (bb)
                return shd_bld_add_instruction(bb, make_unbound(arena, shd_bld_mem(bb), id));
            return make_unbound(arena, NULL, id);
        }
        case hex_lit_tok:
        case dec_lit_tok: {
            shd_next_token(tokenizer);
            return untyped_number(arena, (UntypedNumber) {
                .plaintext = shd_string_sized(arena, (int) size, &contents[tok.start])
            });
        }
        case string_lit_tok: {
            shd_next_token(tokenizer);
            char* unescaped = calloc(size + 1, 1);
            size_t j = shd_apply_escape_codes(&contents[tok.start], size, unescaped);
            const Node* lit = string_lit(arena, (StringLiteral) {.string = shd_string_sized(arena, (int) j, unescaped) });
            free(unescaped);
            return lit;
        }
        case true_tok:
            shd_next_token(tokenizer); return true_lit(arena);
        case false_tok:
            shd_next_token(tokenizer); return false_lit(arena);
        case lpar_tok: {
            shd_next_token(tokenizer);
            if (accept_token(ctx, rpar_tok)) {
                return shd_tuple_helper(arena, shd_empty(arena));
            }
            const Node* atom = expect_operand(ctx, bb);
            if (shd_curr_token(tokenizer).tag == rpar_tok) {
                shd_next_token(tokenizer);
            } else {
                struct List* elements = shd_new_list(const Node*);
                shd_list_append(const Node*, elements, atom);

                while (!accept_token(ctx, rpar_tok)) {
                    expect(accept_token(ctx, comma_tok), "','");
                    const Node* element = expect_operand(ctx, bb);
                    shd_list_append(const Node*, elements, element);
                }

                Nodes tcontents = shd_nodes(arena, shd_list_count(elements), shd_read_list(const Node*, elements));
                shd_destroy_list(elements);
                atom = shd_tuple_helper(arena, tcontents);
            }
            return atom;
        }
        case composite_tok: {
            shd_next_token(tokenizer);
            const Type* elem_type = accept_unqualified_type(ctx);
            expect(elem_type, "composite data type");
            Nodes elems = expect_operands(ctx, bb);
            return composite_helper(arena, elem_type, elems);
        }
        default: return NULL;
    }
}

static AddressSpace accept_address_space(ctxparams) {
    switch (shd_curr_token(tokenizer).tag) {
        case global_tok:
            shd_next_token(tokenizer); return AsGlobal;
        case private_tok:
            shd_next_token(tokenizer); return AsPrivate;
        case shared_tok:
            shd_next_token(tokenizer); return AsShared;
        case subgroup_tok:
            shd_next_token(tokenizer); return AsSubgroup;
        case generic_tok:
            shd_next_token(tokenizer); return AsGeneric;
        case input_tok:
            shd_next_token(tokenizer); return AsInput;
        case output_tok:
            shd_next_token(tokenizer); return AsOutput;
        case extern_tok:
            shd_next_token(tokenizer); return AsExternal;
        default:
            break;
    }
    return NumAddressSpaces;
}

static const Type* accept_unqualified_type(ctxparams) {
    const Type* prim_type = accept_numerical_type(ctx);
    if (prim_type) return prim_type;
    else if (accept_token(ctx, bool_tok)) {
        return bool_type(arena);
    } else if (accept_token(ctx, ptr_tok)) {
        AddressSpace as = accept_address_space(ctx);
        expect(as != NumAddressSpaces, "address space");
        const Type* elem_type = accept_unqualified_type(ctx);
        expect(elem_type, "data type");
        return ptr_type(arena, (PtrType) {
           .address_space = as,
           .pointed_type = elem_type,
        });
    } else if (accept_token(ctx, ref_tok)) {
        AddressSpace as = accept_address_space(ctx);
        expect(as != NumAddressSpaces, "address space");
        const Type* elem_type = accept_unqualified_type(ctx);
        expect(elem_type, "data type");
        return ptr_type(arena, (PtrType) {
           .address_space = as,
           .pointed_type = elem_type,
           .is_reference = true,
        });
    } else if (config->front_end && accept_token(ctx, lsbracket_tok)) {
        const Type* elem_type = accept_unqualified_type(ctx);
        expect(elem_type, "type");
        const Node* size = NULL;
        if (accept_token(ctx, semi_tok)) {
            size = accept_value(ctx, NULL);
            expect(size, "value");
        }
        expect(accept_token(ctx, rsbracket_tok), "']'");
        return arr_type(arena, (ArrType) {
            .element_type = elem_type,
            .size = size
        });
    } else if (accept_token(ctx, vec_tok)) {
        expect(accept_token(ctx, lsbracket_tok), "'['");
        const Type* elem_type = accept_unqualified_type(ctx);
        expect(elem_type, "packed element type");
        const Node* size = NULL;
        expect(accept_token(ctx, semi_tok), "';'");
        size = accept_numerical_literal(ctx);
        expect(size && size->tag == UntypedNumber_TAG, "number");
        expect(accept_token(ctx, rsbracket_tok), "']'");
        return vector_type(arena, (VectorType) {
            .element_type = elem_type,
            .width = strtoll(size->payload.untyped_number.plaintext, NULL, 10)
        });
    } else {
        String id = accept_identifier(ctx);
        if (id)
            return make_unbound(arena, NULL, id);

        return NULL;
    }
}

static const Node* accept_operand(ctxparams, BodyBuilder* bb) {
    return config->front_end ? accept_expr(ctx, bb, max_precedence()) : accept_value(ctx, bb);
}

static const Node* expect_operand(ctxparams, BodyBuilder* bb) {
    const Node* operand = accept_operand(ctx, bb);
    expect(operand, "value operand");
    return operand;
}

static void expect_parameters(ctxparams, Nodes* parameters, Nodes* default_values, BodyBuilder* bb) {
    expect(accept_token(ctx, lpar_tok), "'('");
    struct List* params = shd_new_list(Node*);
    struct List* default_vals = default_values ? shd_new_list(Node*) : NULL;

    while (true) {
        if (accept_token(ctx, rpar_tok))
            break;

        next: {
            const Type* qtype = accept_maybe_qualified_type(ctx);
            expect(qtype, "qualified type");
            const char* id = accept_identifier(ctx);
            expect(id, "parameter name");

            const Node* node = param_helper(arena, qtype);
            shd_set_debug_name(node, id);
            shd_list_append(Node*, params, node);

            if (default_values) {
                expect(accept_token(ctx, equal_tok), "'='");
                const Node* default_val = accept_operand(ctx, bb);
                shd_list_append(const Node*, default_vals, default_val);
            }

            if (accept_token(ctx, comma_tok))
                goto next;
        }
    }

    size_t count = shd_list_count(params);
    *parameters = shd_nodes(arena, count, shd_read_list(const Node*, params));
    shd_destroy_list(params);
    if (default_values) {
        *default_values = shd_nodes(arena, count, shd_read_list(const Node*, default_vals));
        shd_destroy_list(default_vals);
    }
}

typedef enum { MustQualified, MaybeQualified, NeverQualified } Qualified;

static Nodes accept_types(ctxparams, TokenTag separator, Qualified qualified) {
    struct List* tmp = shd_new_list(Type*);
    while (true) {
        const Type* type;
        switch (qualified) {
            case MustQualified:  type = accept_qualified_type(ctx);       break;
            case MaybeQualified: type = accept_maybe_qualified_type(ctx); break;
            case NeverQualified: type = accept_unqualified_type(ctx);     break;
        }
        if (!type)
            break;

        shd_list_append(Type*, tmp, type);

        if (separator != 0)
            accept_token(ctx, separator);
    }

    Nodes types2 = shd_nodes(arena, tmp->elements_count, (const Type**) tmp->alloc);
    shd_destroy_list(tmp);
    return types2;
}

static const Node* accept_primary_expr(ctxparams, BodyBuilder* bb) {
    assert(bb);
    if (accept_token(ctx, minus_tok)) {
        const Node* expr = accept_primary_expr(ctx, bb);
        expect(expr, "expression");
        if (expr->tag == IntLiteral_TAG) {
            return int_literal(arena, (IntLiteral) {
                // We always treat that value like an signed integer, because it makes no sense to negate an unsigned number !
                .value = -shd_get_int_literal_value(*shd_resolve_to_int_literal(expr), true)
            });
        } else {
            return shd_bld_add_instruction(bb, prim_op(arena, (PrimOp) {
                .op = neg_op,
                .operands = shd_nodes(arena, 1, (const Node* []) { expr })
            }));
        }
    } else if (accept_token(ctx, unary_excl_tok)) {
        const Node* expr = accept_primary_expr(ctx, bb);
        expect(expr, "expression");
        return shd_bld_add_instruction(bb, prim_op(arena, (PrimOp) {
            .op = not_op,
            .operands = shd_singleton(expr),
        }));
    } else if (accept_token(ctx, star_tok)) {
        const Node* expr = accept_primary_expr(ctx, bb);
        expect(expr, "expression");
        return shd_bld_add_instruction(bb, ext_instr(arena, (ExtInstr) { .set = "shady.frontend", .result_t = unit_type(arena), .opcode = SlimFrontendOpsSlimDereferenceSHADY, .operands = shd_singleton(expr), .mem = shd_bld_mem(bb) }));
    } else if (accept_token(ctx, infix_and_tok)) {
        const Node* expr = accept_primary_expr(ctx, bb);
        expect(expr, "expression");
        return shd_bld_add_instruction(bb, ext_instr(arena, (ExtInstr) {
            .set = "shady.frontend",
            .result_t = unit_type(arena),
            .opcode = SlimFrontendOpsSlimAddrOfSHADY,
            .operands = shd_singleton(expr),
            .mem = shd_bld_mem(bb),
        }));
    }

    return accept_value(ctx, bb);
}

static const Node* accept_expr(ctxparams, BodyBuilder* bb, int outer_precedence) {
    assert(bb);
    const Node* expr = accept_primary_expr(ctx, bb);
    while (expr) {
        InfixOperators infix;
        if (is_infix_operator(shd_curr_token(tokenizer).tag, &infix)) {
            int precedence = get_precedence(infix);
            if (precedence > outer_precedence) break;
            shd_next_token(tokenizer);

            const Node* rhs = accept_expr(ctx, bb, precedence - 1);
            expect(rhs, "expression");
            Op primop_op;
            if (is_primop_op(infix, &primop_op)) {
                expr = shd_bld_add_instruction(bb, prim_op(arena, (PrimOp) {
                    .op = primop_op,
                    .operands = shd_nodes(arena, 2, (const Node* []) { expr, rhs })
                }));
            } else switch (infix) {
                case InfixAss: {
                    expr = shd_bld_add_instruction(bb, ext_instr(arena, (ExtInstr) {
                        .set = "shady.frontend",
                        .opcode = SlimFrontendOpsSlimAssignSHADY,
                        .result_t = unit_type(arena),
                        .operands = shd_nodes(arena, 2, (const Node* []) { expr, rhs }),
                        .mem = shd_bld_mem(bb),
                    }));
                    break;
                }
                case InfixSbs: {
                    expr = shd_bld_add_instruction(bb, ext_instr(arena, (ExtInstr) {
                        .set = "shady.frontend",
                        .opcode = SlimFrontendOpsSlimSubscriptSHADY,
                        .result_t = unit_type(arena),
                        .operands = shd_nodes(arena, 2, (const Node* []) { expr, rhs }),
                        .mem = shd_bld_mem(bb),
                    }));
                    break;
                }
                default: syntax_error("unknown infix operator");
            }
            continue;
        }

        switch (shd_curr_token(tokenizer).tag) {
            case lpar_tok: {
                Nodes ops = expect_operands(ctx, bb);
                expr = shd_bld_add_instruction(bb, indirect_call(arena, (IndirectCall) {
                    .callee = expr,
                    .args = ops,
                    .mem = shd_bld_mem(bb),
                }));
                continue;
            }
            default:
                break;
        }

        break;
    }
    return expr;
}

static Nodes expect_operands(ctxparams, BodyBuilder* bb) {
    expect(accept_token(ctx, lpar_tok), "'('");

    struct List* list = shd_new_list(Node*);

    bool expect = false;
    while (true) {
        const Node* val = accept_operand(ctx, bb);
        if (!val) {
            if (expect)
                syntax_error("expected value but got none");
            else if (accept_token(ctx, rpar_tok))
                break;
            else
                syntax_error("Expected value or ')'");
        }

        shd_list_append(Node*, list, val);

        if (accept_token(ctx, comma_tok))
            expect = true;
        else if (accept_token(ctx, rpar_tok))
            break;
        else
            syntax_error("Expected ',' or ')'");
    }

    Nodes final = shd_nodes(arena, list->elements_count, (const Node**) list->alloc);
    shd_destroy_list(list);
    return final;
}

static const Node* make_selection_merge(const Node* mem) {
    IrArena* a = mem->arena;
    return merge_selection(a, (MergeSelection) { .args = shd_nodes(a, 0, NULL), .mem = mem });
}

static const Node* make_loop_continue(const Node* mem) {
    IrArena* a = mem->arena;
    return merge_continue(a, (MergeContinue) { .args = shd_nodes(a, 0, NULL), .mem = mem });
}

static const Node* accept_control_flow_instruction(ctxparams, BodyBuilder* bb) {
    Token current_token = shd_curr_token(tokenizer);
    switch (current_token.tag) {
        case if_tok: {
            shd_next_token(tokenizer);
            Nodes yield_types = accept_types(ctx, 0, NeverQualified);
            expect(accept_token(ctx, lpar_tok), "'('");
            const Node* condition = accept_operand(ctx, bb);
            expect(condition, "condition value");
            expect(accept_token(ctx, rpar_tok), "')'");
            const Node* (*merge)(const Node*) = config->front_end ? make_selection_merge : NULL;

            Node* true_case = basic_block_helper(arena, shd_nodes(arena, 0, NULL));
            shd_set_abstraction_body(true_case, expect_body(ctx, shd_get_abstraction_mem(true_case), merge));

            // else defaults to an empty body
            bool has_else = accept_token(ctx, else_tok);
            Node* false_case = NULL;
            if (has_else) {
                false_case = basic_block_helper(arena, shd_nodes(arena, 0, NULL));
                shd_set_abstraction_body(false_case, expect_body(ctx, shd_get_abstraction_mem(false_case), merge));
            }
            return shd_maybe_tuple_helper(arena, shd_bld_if(bb, yield_types, condition, true_case, false_case));
        }
        case loop_tok: {
            shd_next_token(tokenizer);
            Nodes yield_types = accept_types(ctx, 0, NeverQualified);
            Nodes parameters;
            Nodes initial_arguments;
            expect_parameters(ctx, &parameters, &initial_arguments, bb);
            // by default loops continue forever
            const Node* (*default_loop_end_behaviour)(const Node*) = config->front_end ? make_loop_continue : NULL;
            Node* loop_case = basic_block_helper(arena, parameters);
            shd_set_abstraction_body(loop_case, expect_body(ctx, shd_get_abstraction_mem(loop_case), default_loop_end_behaviour));
            return shd_maybe_tuple_helper(arena, shd_bld_loop(bb, yield_types, initial_arguments, loop_case));
        }
        case control_tok: {
            shd_next_token(tokenizer);
            Nodes yield_types = accept_types(ctx, 0, NeverQualified);
            expect(accept_token(ctx, lpar_tok), "'('");
            String str = accept_identifier(ctx);
            expect(str, "control parameter name");
            const Node* jp = param_helper(arena, join_point_type(arena, (JoinPointType) {
                .yield_types = yield_types,
            }));
            shd_set_debug_name(jp, str);
            expect(accept_token(ctx, rpar_tok), "')'");
            Node* control_case = basic_block_helper(arena, shd_singleton(jp));
            shd_set_abstraction_body(control_case, expect_body(ctx, shd_get_abstraction_mem(control_case), NULL));
            return shd_maybe_tuple_helper(arena, shd_bld_control(bb, yield_types, control_case));
        }
        default: break;
    }
    return NULL;
}

static const Node* accept_instruction(ctxparams, BodyBuilder* bb) {
    const Node* instr = accept_expr(ctx, bb, max_precedence());

    switch(shd_curr_token(tokenizer).tag) {
        case call_tok: {
            shd_next_token(tokenizer);
            expect(accept_token(ctx, lpar_tok), "'('");
            const Node* callee = accept_operand(ctx, bb);
            expect(accept_token(ctx, rpar_tok), "')'");
            Nodes args = expect_operands(ctx, bb);
            return indirect_call(arena, (IndirectCall) {
                    .callee = callee,
                    .args = args,
                    .mem = shd_bld_mem(bb)
            });
        }
        default: break;
    }

    if (instr)
        expect(accept_token(ctx, semi_tok), "';'");

    if (!instr) instr = accept_control_flow_instruction(ctx, bb);
    return instr;
}

static void expect_identifiers(ctxparams, Strings* out_strings) {
    struct List* list = shd_new_list(const char*);
    while (true) {
        const char* id = accept_identifier(ctx);
        expect(id, "identifier");

        shd_list_append(const char*, list, id);

        if (accept_token(ctx, comma_tok))
            continue;
        else
            break;
    }

    *out_strings = shd_strings(arena, list->elements_count, (const char**) list->alloc);
    shd_destroy_list(list);
}

static void expect_types_and_identifiers(ctxparams, Strings* out_strings, Nodes* out_types) {
    struct List* slist = shd_new_list(const char*);
    struct List* tlist = shd_new_list(const char*);

    while (true) {
        const Type* type = accept_unqualified_type(ctx);
        expect(type, "type");
        const char* id = accept_identifier(ctx);
        expect(id, "identifier");

        shd_list_append(const char*, tlist, type);
        shd_list_append(const char*, slist, id);

        if (accept_token(ctx, comma_tok))
            continue;
        else
            break;
    }

    *out_strings = shd_strings(arena, slist->elements_count, (const char**) slist->alloc);
    *out_types = shd_nodes(arena, tlist->elements_count, (const Node**) tlist->alloc);
    shd_destroy_list(slist);
    shd_destroy_list(tlist);
}

static Nodes strings2nodes(IrArena* a, Strings strings) {
    LARRAY(const Node*, arr, strings.count);
    for (size_t i = 0; i < strings.count; i++)
        arr[i] = string_lit_helper(a, strings.strings[i]);
    return shd_nodes(a, strings.count, arr);
}

static bool accept_statement(ctxparams, BodyBuilder* bb) {
    Strings ids;
    if (accept_token(ctx, val_tok)) {
        expect_identifiers(ctx, &ids);
        expect(accept_token(ctx, equal_tok), "'='");
        const Node* instruction = accept_instruction(ctx, bb);
        shd_bld_ext_instruction(bb, "shady.frontend", SlimFrontendOpsSlimBindValSHADY, unit_type(arena), shd_nodes_prepend(arena, strings2nodes(arena, ids), instruction));
    } else if (accept_token(ctx, var_tok)) {
        Nodes types;
        expect_types_and_identifiers(ctx, &ids, &types);
        expect(accept_token(ctx, equal_tok), "'='");
        const Node* instruction = accept_instruction(ctx, bb);
        shd_bld_ext_instruction(bb, "shady.frontend", SlimFrontendOpsSlimBindVarSHADY, unit_type(arena), shd_nodes_prepend(arena, shd_concat_nodes(arena, strings2nodes(arena, ids), types), instruction));
    } else {
        const Node* instr = accept_instruction(ctx, bb);
        if (!instr) return false;
        //bind_instruction_outputs_count(bb, instr, 0);
    }
    return true;
}

static const Node* expect_jump(ctxparams, BodyBuilder* bb) {
    String target = accept_identifier(ctx);
    expect(target, "jump target name");
    Nodes args = expect_operands(ctx, bb);
    const Node* tgt = make_unbound(arena, shd_bld_mem(bb), target);
    shd_bld_add_instruction(bb, tgt);
    return jump(arena, (Jump) {
        .target = tgt,
        .args = args,
        .mem = shd_bld_mem(bb)
    });
}

static const Node* accept_terminator(ctxparams, BodyBuilder* bb) {
    TokenTag tag = shd_curr_token(tokenizer).tag;
    switch (tag) {
        case jump_tok: {
            shd_next_token(tokenizer);
            return expect_jump(ctx, bb);
        }
        case branch_tok: {
            shd_next_token(tokenizer);

            expect(accept_token(ctx, lpar_tok), "'('");
            const Node* condition = accept_value(ctx, bb);
            expect(condition, "branch condition value");
            expect(accept_token(ctx, comma_tok), "','");
            const Node* true_target = expect_jump(ctx, bb);
            expect(accept_token(ctx, comma_tok), "','");
            const Node* false_target = expect_jump(ctx, bb);
            expect(accept_token(ctx, rpar_tok), "')'");

            return branch(arena, (Branch) {
                .condition = condition,
                .true_jump = true_target,
                .false_jump = false_target,
                .mem = shd_bld_mem(bb)
            });
        }
        case switch_tok: {
            shd_next_token(tokenizer);

            expect(accept_token(ctx, lpar_tok), "'('");
            const Node* inspectee = accept_value(ctx, bb);
            expect(inspectee, "value");
            expect(accept_token(ctx, comma_tok), "','");
            Nodes values = shd_empty(arena);
            Nodes cases = shd_empty(arena);
            const Node* default_jump;
            while (true) {
                if (accept_token(ctx, default_tok)) {
                    default_jump = expect_jump(ctx, bb);
                    break;
                }
                expect(accept_token(ctx, case_tok), "'case'");
                const Node* value = accept_value(ctx, bb);
                expect(value, "case value");
                expect(accept_token(ctx, comma_tok), "','");
                const Node* j = expect_jump(ctx, bb);
                expect(accept_token(ctx, comma_tok), "','");
                values = shd_nodes_append(arena, values, value);
                cases = shd_nodes_append(arena, cases, j);
            }
            expect(accept_token(ctx, rpar_tok), "')'");

            return br_switch(arena, (Switch) {
                .switch_value = shd_first(values),
                .case_values = values,
                .case_jumps = cases,
                .default_jump = default_jump,
                .mem = shd_bld_mem(bb)
            });
        }
        case return_tok: {
            shd_next_token(tokenizer);
            Nodes args = expect_operands(ctx, bb);
            return fn_ret(arena, (Return) {
                .args = args,
                .mem = shd_bld_mem(bb)
            });
        }
        case merge_selection_tok: {
            shd_next_token(tokenizer);
            Nodes args = shd_curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx, bb) : shd_nodes(arena, 0, NULL);
            return merge_selection(arena, (MergeSelection) {
                .args = args,
                .mem = shd_bld_mem(bb)
            });
        }
        case continue_tok: {
            shd_next_token(tokenizer);
            Nodes args = shd_curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx, bb) : shd_nodes(arena, 0, NULL);
            return merge_continue(arena, (MergeContinue) {
                .args = args,
                .mem = shd_bld_mem(bb)
            });
        }
        case break_tok: {
            shd_next_token(tokenizer);
            Nodes args = shd_curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx, bb) : shd_nodes(arena, 0, NULL);
            return merge_break(arena, (MergeBreak) {
                .args = args,
                .mem = shd_bld_mem(bb)
            });
        }
        case join_tok: {
            shd_next_token(tokenizer);
            expect(accept_token(ctx, lpar_tok), "'('");
            const Node* jp = accept_operand(ctx, bb);
            expect(accept_token(ctx, rpar_tok), "')'");
            Nodes args = expect_operands(ctx, bb);
            return join(arena, (Join) {
                .join_point = jp,
                .args = args,
                .mem = shd_bld_mem(bb)
            });
        }
        case tailcall_tok: {
            shd_next_token(tokenizer);
            expect(accept_token(ctx, lpar_tok), "'('");
            const Node* callee = accept_operand(ctx, bb);
            expect(accept_token(ctx, rpar_tok), "')'");
            Nodes args = expect_operands(ctx, bb);
            return indirect_tail_call(arena, (IndirectTailCall) {
                .callee = callee,
                .args = args,
                .mem = shd_bld_mem(bb)
            });
        }
        case unreachable_tok: {
            shd_next_token(tokenizer);
            expect(accept_token(ctx, lpar_tok), "'('");
            expect(accept_token(ctx, rpar_tok), "')'");
            return unreachable(arena, (Unreachable) { .mem = shd_bld_mem(bb) });
        }
        default: break;
    }
    return NULL;
}

static const Node* expect_body(ctxparams, const Node* mem, const Node* default_terminator(const Node*)) {
    expect(accept_token(ctx, lbracket_tok), "'['");
    BodyBuilder* bb = shd_bld_begin(arena, mem);

    while (true) {
        if (!accept_statement(ctx, bb))
            break;
    }

    Node* terminator_case = basic_block_helper(arena, shd_empty(arena));
    BodyBuilder* terminator_bb = shd_bld_begin(arena, shd_get_abstraction_mem(terminator_case));
    const Node* terminator = accept_terminator(ctx, terminator_bb);

    if (terminator)
        expect(accept_token(ctx, semi_tok), "';'");

    if (!terminator) {
        if (default_terminator)
            terminator = default_terminator(shd_bld_mem(terminator_bb));
        else
            syntax_error("expected terminator: return, jump, branch ...");
    }

    shd_set_abstraction_body(terminator_case, shd_bld_finish(terminator_bb, terminator));

    Node* cont_wrapper_case = basic_block_helper(arena, shd_empty(arena));
    BodyBuilder* cont_wrapper_bb = shd_bld_begin(arena, shd_get_abstraction_mem(cont_wrapper_case));

    Nodes ids = shd_empty(arena);
    Nodes conts = shd_empty(arena);
    if (shd_curr_token(tokenizer).tag == cont_tok) {
        while (true) {
            if (!accept_token(ctx, cont_tok))
                break;
            const char* name = accept_identifier(ctx);

            Nodes parameters;
            expect_parameters(ctx, &parameters, NULL, bb);
            Node* continuation = basic_block_helper(arena, parameters);
            shd_set_abstraction_body(continuation, expect_body(ctx, shd_get_abstraction_mem(continuation), NULL));
            ids = shd_nodes_append(arena, ids, string_lit_helper(arena, name));
            conts = shd_nodes_append(arena, conts, continuation);
        }
    }

    shd_bld_ext_instruction(cont_wrapper_bb, "shady.frontend", SlimFrontendOpsSlimBindContinuationsSHADY, unit_type(arena), shd_concat_nodes(arena, ids, conts));
    expect(accept_token(ctx, rbracket_tok), "']'");

    shd_set_abstraction_body(cont_wrapper_case, shd_bld_jump(cont_wrapper_bb, terminator_case, shd_empty(arena)));
    return shd_bld_jump(bb, cont_wrapper_case, shd_empty(arena));
}

static Nodes accept_annotations(ctxparams) {
    struct List* list = shd_new_list(const Node*);

    while (true) {
        if (accept_token(ctx, at_tok)) {
            const char* id = accept_identifier(ctx);
            const Node* annot = NULL;
            if (accept_token(ctx, lpar_tok)) {
                const Node* first_value = accept_value(ctx, NULL);
                if (!first_value) {
                    expect(accept_token(ctx, rpar_tok), "value");
                    goto no_params;
                }

                // TODO: AnnotationCompound ?
                if (shd_curr_token(tokenizer).tag == comma_tok) {
                    shd_next_token(tokenizer);
                    struct List* values = shd_new_list(const Node*);
                    shd_list_append(const Node*, values, first_value);
                    while (true) {
                        const Node* next_value = accept_value(ctx, NULL);
                        expect(next_value, "value");
                        shd_list_append(const Node*, values, next_value);
                        if (accept_token(ctx, comma_tok))
                            continue;
                        else break;
                    }
                    annot = annotation_values(arena, (AnnotationValues) {
                        .name = id,
                        .values = shd_nodes(arena, shd_list_count(values), shd_read_list(const Node*, values))
                    });
                    shd_destroy_list(values);
                } else {
                    annot = annotation_value(arena, (AnnotationValue) {
                        .name = id,
                        .value = first_value
                    });
                }

                expect(accept_token(ctx, rpar_tok), "')'");
            } else {
                no_params:
                annot = annotation(arena, (Annotation) {
                    .name = id,
                });
            }
            expect(annot, "annotation");
            shd_list_append(const Node*, list, annot);
            continue;
        }
        break;
    }

    Nodes annotations = shd_nodes(arena, shd_list_count(list), shd_read_list(const Node*, list));
    shd_destroy_list(list);
    return annotations;
}

static const Node* accept_const(ctxparams, String* bind_name, Nodes annotations) {
    if (!accept_token(ctx, const_tok))
        return NULL;

    const Type* type = accept_unqualified_type(ctx);
    const char* name = accept_identifier(ctx);
    expect(name, "constant name");
    *bind_name = name;
    expect(accept_token(ctx, equal_tok), "'='");
    BodyBuilder* bb = shd_bld_begin_pure(arena);
    const Node* definition = accept_expr(ctx, bb, max_precedence());
    expect(definition, "expression");

    expect(accept_token(ctx, semi_tok), "';'");

    Node* cnst = constant_helper(mod, type);
    cnst->payload.constant.value = shd_bld_to_instr_pure_with_values(bb, shd_singleton(definition));
    cnst->annotations = annotations;
    shd_set_debug_name(cnst, name);
    return cnst;
}

static const Node* make_return_void(const Node* mem) {
    IrArena* a = mem->arena;
    return fn_ret(a, (Return) { .args = shd_empty(a), .mem = mem });
}

static const Node* accept_fn_decl(ctxparams, String* bind_name, Nodes annotations) {
    if (!accept_token(ctx, fn_tok))
        return NULL;

    const char* name = accept_identifier(ctx);
    expect(name, "function name");
    *bind_name = name;
    Nodes types = accept_types(ctx, comma_tok, MaybeQualified);
    expect(shd_curr_token(tokenizer).tag == lpar_tok, "')'");
    Nodes parameters;
    expect_parameters(ctx, &parameters, NULL, NULL);

    Node* fn = function_helper(mod, parameters, types);
    fn->annotations = annotations;
    shd_set_debug_name(fn, name);
    if (!accept_token(ctx, semi_tok))
        shd_set_abstraction_body(fn, expect_body(ctx, shd_get_abstraction_mem(fn), types.count == 0 ? make_return_void : NULL));

    return fn;
}

static const Node* accept_global_var_decl(ctxparams, String* bind_name, Nodes annotations) {
    if (!accept_token(ctx, var_tok))
        return NULL;

    GlobalVariable payload = {
        .address_space = NumAddressSpaces,
    };
    bool uniform = false;
    while (true) {
        if (accept_token(ctx, logical_tok)) {
            payload.is_ref = true;
            continue;
        }
        if (accept_token(ctx, uniform_tok)) {
            uniform = true;
            continue;
        }
        AddressSpace nas = accept_address_space(ctx);
        if (nas != NumAddressSpaces) {
            if (payload.address_space != NumAddressSpaces && payload.address_space != nas) {
                syntax_error_fmt("Conflicting address spaces for definition: %s and %s", shd_get_address_space_name(payload.address_space), shd_get_address_space_name(nas));
            }
            payload.address_space = nas;
            continue;
        }
        break;
    }

    if (payload.address_space == NumAddressSpaces) {
        syntax_error("Address space required for global variable declaration.");
    }

    if (uniform) {
        if (payload.address_space == AsInput)
            payload.address_space = AsUInput;
        else {
            syntax_error("'uniform' can only be used with 'input'");
        }
    }

    payload.type = accept_unqualified_type(ctx);
    expect(payload.type, "global variable type");
    String name = accept_identifier(ctx);
    expect(name, "global variable name");
    *bind_name = name;

    const Node* initial_value = NULL;
    if (accept_token(ctx, equal_tok)) {
        initial_value = accept_value(ctx, NULL);
        expect_fmt(initial_value, "value for global variable '%s'", name);
    }

    expect(accept_token(ctx, semi_tok), "';'");

    Node* gv = shd_global_var(mod, payload);
    gv->payload.global_variable.init = initial_value;
    gv->annotations = annotations;
    shd_set_debug_name(gv, name);
    return gv;
}

static const Node* accept_struct_decl(ctxparams, String* bind_name, Nodes annotations) {
    if (!accept_token(ctx, struct_tok))
        return NULL;

    const char* name = accept_identifier(ctx);
    expect(name, "struct type name");
    *bind_name = name;

    expect(accept_token(ctx, lbracket_tok), "'{'");
    struct List* names = shd_new_list(String);
    struct List* types = shd_new_list(const Type*);
    while (true) {
        if (accept_token(ctx, rbracket_tok))
            break;
        const Type* elem = accept_unqualified_type(ctx);
        expect(elem, "struct member type");
        String id = accept_identifier(ctx);
        expect(id, "struct member name");
        shd_list_append(String, names, id);
        shd_list_append(const Type*, types, elem);
        expect(accept_token(ctx, semi_tok), "';'");
    }
    expect(accept_token(ctx, semi_tok), "';'");
    Nodes elem_types = shd_nodes(arena, shd_list_count(types), shd_read_list(const Type*, types));
    Strings names2 = shd_strings(arena, shd_list_count(names), shd_read_list(String, names));
    shd_destroy_list(names);
    shd_destroy_list(types);
    Type* st = shd_struct_type_with_members_named(arena, 0, elem_types, names2);
    st->annotations = annotations;
    shd_set_debug_name(st, name);
    return st;
}

static const Node* accept_alias(ctxparams, String* bind_name, Nodes annotations) {
    if (!accept_token(ctx, alias_tok))
        return NULL;

    const char* name = accept_identifier(ctx);
    expect(name, "alias type name");
    *bind_name = name;

    expect(annotations.count == 0, "annotations are not allowed on aliases");
    expect(accept_token(ctx, equal_tok), "'='");
    const Type* t = accept_unqualified_type(ctx);
    expect(accept_token(ctx, semi_tok), "';'");
    return t;
}

void slim_parse_string(const SlimParserConfig* config, const char* contents, Module* mod) {
    IrArena* arena = shd_module_get_arena(mod);
    Tokenizer* tokenizer = shd_new_tokenizer(contents);

    Node* file_top_level = shd_module_get_exported(mod, "_top_level_bindings");
    if (!file_top_level) {
        file_top_level = global_variable_helper(mod, shd_uint16_type(arena), AsGlobal);
        shd_module_add_export(mod, "_top_level_bindings", file_top_level);
    }

    while (true) {
        Token token = shd_curr_token(tokenizer);
        if (token.tag == EOF_tok)
            break;

        Nodes annotations = accept_annotations(ctx);
        // annotations = shd_nodes_append(arena, annotations, annotation_value_helper(arena, "SlimTopLevelBindings", file_top_level));

        String bind_name = NULL;
        const Node* decl = accept_const(ctx, &bind_name, annotations);
        if (!decl)  decl = accept_fn_decl(ctx, &bind_name, annotations);
        if (!decl)  decl = accept_global_var_decl(ctx, &bind_name, annotations);
        if (!decl)  decl = accept_struct_decl(ctx, &bind_name, annotations);
        if (!decl)  decl = accept_alias(ctx, &bind_name, annotations);

        if (!decl)
            syntax_error("expected a declaration");

        const Node* oea = shd_lookup_annotation(decl, "Exported");
        String exported_name = NULL;
        if (oea) {
            // if we specify a different export name with the annotation
            if (oea->tag == AnnotationValue_TAG) {
                exported_name = shd_get_exported_name(decl);
            } else {
                shd_remove_annotation_by_name(decl, "Exported");
                exported_name = bind_name;
                assert(exported_name);
            }
        }

        assert(bind_name && "top-level things should be named");
        shd_add_annotation(file_top_level, annotation_id_helper(arena, bind_name, decl));

        if (exported_name)
            shd_module_add_export(mod, exported_name, decl);

        shd_log_fmt(DEBUGVV, "decl parsed: %s = ", exported_name);
        shd_log_node(DEBUGVV, decl);
        shd_log_fmt(DEBUGVV, "\n");
    }

    shd_destroy_tokenizer(tokenizer);
}

