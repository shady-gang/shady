#include "token.h"
#include "parser.h"

#include "list.h"
#include "portability.h"
#include "log.h"
#include "util.h"

#include "../type.h"
#include "../ir_private.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

static int max_precedence() {
    return 10;
}

static int get_precedence(InfixOperators op) {
    switch (op) {
#define INFIX_OPERATOR(name, token, primop_op, precedence) case Infix##name: return precedence;
INFIX_OPERATORS()
#undef INFIX_OPERATOR
        default: error("unknown operator");
    }
}
static bool is_primop_op(InfixOperators op, Op* out) {
    switch (op) {
#define INFIX_OPERATOR(name, token, primop_op, precedence) case Infix##name: if (primop_op != -1) { *out = primop_op; return true; } else return false;
INFIX_OPERATORS()
#undef INFIX_OPERATOR
        default: error("unknown operator");
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
#define ctxparams SHADY_UNUSED ParserConfig config, SHADY_UNUSED const char* contents, SHADY_UNUSED Module* mod, SHADY_UNUSED IrArena* arena, SHADY_UNUSED Tokenizer* tokenizer
#define ctx config, contents, mod, arena, tokenizer

#define expect(condition) expect_impl(condition, #condition)
static void expect_impl(bool condition, const char* err) {
    if (!condition) {
        error_print("expected to parse: %s\n", err);
        exit(-4);
    }
}

static bool accept_token(ctxparams, TokenTag tag) {
    if (curr_token(tokenizer).tag == tag) {
        next_token(tokenizer);
        return true;
    }
    return false;
}

static const char* accept_identifier(ctxparams) {
    Token tok = curr_token(tokenizer);
    if (tok.tag == identifier_tok) {
        next_token(tokenizer);
        size_t size = tok.end - tok.start;
        return string_sized(arena, (int) size, &contents[tok.start]);
    }
    return NULL;
}

static const Node* expect_body(ctxparams, Node* fn, const Node* default_terminator);
static const Node* accept_value(ctxparams);
static const Type* accept_unqualified_type(ctxparams);
static const Node* accept_primop(ctxparams);
static const Node* accept_expr(ctxparams, int);
static Nodes expect_operands(ctxparams);

static const Type* accept_numerical_type(ctxparams) {
    if (accept_token(ctx, i8_tok)) {
        return int8_type(arena);
    } else if (accept_token(ctx, i16_tok)) {
        return int16_type(arena);
    } else if (accept_token(ctx, i32_tok)) {
        return int32_type(arena);
    } else if (accept_token(ctx, i64_tok)) {
        return int64_type(arena);
    } else if (accept_token(ctx, u8_tok)) {
        return uint8_type(arena);
    } else if (accept_token(ctx, u16_tok)) {
        return uint16_type(arena);
    } else if (accept_token(ctx, u32_tok)) {
        return uint32_type(arena);
    } else if (accept_token(ctx, u64_tok)) {
        return uint64_type(arena);
    } else if (accept_token(ctx, f16_tok)) {
        return fp16_type(arena);
    } else if (accept_token(ctx, f32_tok)) {
        return fp32_type(arena);
    } else if (accept_token(ctx, f64_tok)) {
        return fp64_type(arena);
    }
    return NULL;
}

static const Node* accept_numerical_literal(ctxparams) {
    const Type* num_type = accept_numerical_type(ctx);

    bool negate = accept_token(ctx, minus_tok);

    Token tok = curr_token(tokenizer);
    size_t size = tok.end - tok.start;
    String str = string_sized(arena, (int) size, &contents[tok.start]);

    switch (tok.tag) {
        case hex_lit_tok:
            if (negate)
                error("hexadecimal literals can't start with '-'");
        case dec_lit_tok: {
            next_token(tokenizer);
            break;
        }
        default: {
            if (negate || num_type)
                error("expected numerical literal");
            return NULL;
        }
    }

    if (negate) // add back the - in front
        str = format_string(arena, "-%s", str);

    const Node* n = untyped_number(arena, (UntypedNumber) {
            .plaintext = str
    });

    if (num_type)
        n = constrained(arena, (ConstrainedValue) {
            .type = num_type,
            .value = n
        });

    return n;
}

static const Node* accept_value(ctxparams) {
    Token tok = curr_token(tokenizer);
    size_t size = tok.end - tok.start;

    const Node* number = accept_numerical_literal(ctx);
    if (number)
        return number;

    switch (tok.tag) {
        case identifier_tok: {
            const char* id = string_sized(arena, (int) size, &contents[tok.start]);
            next_token(tokenizer);
            return unbound(arena, (Unbound) { .name = id });
        }
        case hex_lit_tok:
        case dec_lit_tok: {
            next_token(tokenizer);
            return untyped_number(arena, (UntypedNumber) {
                .plaintext = string_sized(arena, (int) size, &contents[tok.start])
            });
        }
        case string_lit_tok: {
            next_token(tokenizer);
            char* unescaped = calloc(size + 1, 1);
            size_t j = apply_escape_codes(&contents[tok.start], size, unescaped);
            const Node* lit = string_lit(arena, (StringLiteral) {.string = string_sized(arena, (int) j, unescaped) });
            free(unescaped);
            return lit;
        }
        case true_tok: next_token(tokenizer); return true_lit(arena);
        case false_tok: next_token(tokenizer); return false_lit(arena);
        case lpar_tok: {
            next_token(tokenizer);
            if (accept_token(ctx, rpar_tok)) {
                return quote_helper(arena, empty(arena));
            }
            const Node* atom = config.front_end ? accept_expr(ctx, max_precedence()) : accept_value(ctx);
            expect(atom);
            if (curr_token(tokenizer).tag == rpar_tok) {
                next_token(tokenizer);
            } else {
                struct List* elements = new_list(const Node*);
                append_list(const Node*, elements, atom);

                while (!accept_token(ctx, rpar_tok)) {
                    expect(accept_token(ctx, comma_tok));
                    const Node* element = config.front_end ? accept_expr(ctx, max_precedence()) : accept_value(ctx);
                    expect(elements);
                    append_list(const Node*, elements, element);
                }

                Nodes tcontents = nodes(arena, entries_count_list(elements), read_list(const Node*, elements));
                destroy_list(elements);
                atom = tuple_helper(arena, tcontents);
            }
            return atom;
        }
        case composite_tok: {
            next_token(tokenizer);
            const Type* elem_type = accept_unqualified_type(ctx);
            expect(elem_type);
            Nodes elems = expect_operands(ctx);
            return composite_helper(arena, elem_type, elems);
        }
        default: return NULL;
    }
}

static AddressSpace expect_ptr_address_space(ctxparams) {
    switch (curr_token(tokenizer).tag) {
        case global_tok:  next_token(tokenizer); return AsGlobalPhysical;
        case private_tok: next_token(tokenizer); return AsPrivatePhysical;
        case shared_tok:  next_token(tokenizer); return AsSharedPhysical;
        case subgroup_tok:  next_token(tokenizer); return AsSubgroupPhysical;
        case generic_tok:  next_token(tokenizer); return AsGeneric;
        default: error("expected address space qualifier");
    }
    SHADY_UNREACHABLE;
}

static const Type* accept_unqualified_type(ctxparams) {
    const Type* prim_type = accept_numerical_type(ctx);
    if (prim_type) return prim_type;
    else if (accept_token(ctx, bool_tok)) {
        return bool_type(arena);
    } else if (accept_token(ctx, mask_t_tok)) {
        return mask_type(arena);
    } else if (accept_token(ctx, ptr_tok)) {
        AddressSpace as = expect_ptr_address_space(ctx);
        const Type* elem_type = accept_unqualified_type(ctx);
        expect(elem_type);
        return ptr_type(arena, (PtrType) {
           .address_space = as,
           .pointed_type = elem_type,
        });
    } else if (config.front_end && accept_token(ctx, lsbracket_tok)) {
        const Type* elem_type = accept_unqualified_type(ctx);
        expect(elem_type);
        const Node* size = NULL;
        if(accept_token(ctx, semi_tok)) {
            size = accept_value(ctx);
            expect(size);
        }
        expect(accept_token(ctx, rsbracket_tok));
        return arr_type(arena, (ArrType) {
            .element_type = elem_type,
            .size = size
        });
    } else if (accept_token(ctx, pack_tok)) {
        expect(accept_token(ctx, lsbracket_tok));
        const Type* elem_type = accept_unqualified_type(ctx);
        expect(elem_type);
        const Node* size = NULL;
        expect(accept_token(ctx, semi_tok));
        size = accept_numerical_literal(ctx);
        expect(size && size->tag == UntypedNumber_TAG);
        expect(accept_token(ctx, rsbracket_tok));
        return pack_type(arena, (PackType) {
            .element_type = elem_type,
            .width = strtoll(size->payload.untyped_number.plaintext, NULL, 10)
        });
    } else if (accept_token(ctx, struct_tok)) {
        expect(accept_token(ctx, lbracket_tok));
        struct List* names = new_list(String);
        struct List* types = new_list(const Type*);
        while (true) {
            if (accept_token(ctx, rbracket_tok))
                break;
            const Type* elem = accept_unqualified_type(ctx);
            expect(elem);
            String id = accept_identifier(ctx);
            expect(id);
            append_list(String, names, id);
            append_list(const Type*, types, elem);
            expect(accept_token(ctx, semi_tok));
        }
        Nodes elem_types = nodes(arena, entries_count_list(types), read_list(const Type*, types));
        Strings names2 = strings(arena, entries_count_list(names), read_list(String, names));
        destroy_list(names);
        destroy_list(types);
        return record_type(arena, (RecordType) {
            .names = names2,
            .members = elem_types,
            .special = NotSpecial,
        });
    } else {
        String id = accept_identifier(ctx);
        if (id)
            return unbound(arena, (Unbound) { .name = id });

        return NULL;
    }
}

static DivergenceQualifier accept_uniformity_qualifier(ctxparams) {
    DivergenceQualifier divergence = Unknown;
    if (accept_token(ctx, uniform_tok))
        divergence = Uniform;
    else if (accept_token(ctx, varying_tok))
        divergence = Varying;
    return divergence;
}

static const Type* accept_maybe_qualified_type(ctxparams) {
    DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    const Type* unqualified = accept_unqualified_type(ctx);
    if (qualifier != Unknown)
        expect(unqualified && "we read a uniformity qualifier and expected a type to follow");
    if (qualifier == Unknown)
        return unqualified;
    else
        return qualified_type(arena, (QualifiedType) { .is_uniform = qualifier == Uniform, .type = unqualified });
}

static const Type* accept_qualified_type(ctxparams) {
    DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    if (qualifier == Unknown)
        return NULL;
    const Type* unqualified = accept_unqualified_type(ctx);
    expect(unqualified);
    return qualified_type(arena, (QualifiedType) { .is_uniform = qualifier == Uniform, .type = unqualified });
}

static const Node* accept_operand(ctxparams) {
    return config.front_end ? accept_expr(ctx, max_precedence()) : accept_value(ctx);
}

static void expect_parameters(ctxparams, Nodes* parameters, Nodes* default_values) {
    expect(accept_token(ctx, lpar_tok));
    struct List* params = new_list(Node*);
    struct List* default_vals = default_values ? new_list(Node*) : NULL;

    while (true) {
        if (accept_token(ctx, rpar_tok))
            break;

        next: {
            const Type* qtype = accept_qualified_type(ctx);
            expect(qtype);
            const char* id = accept_identifier(ctx);
            expect(id);

            const Node* node = var(arena, qtype, id);
            append_list(Node*, params, node);

            if (default_values) {
                expect(accept_token(ctx, equal_tok));
                const Node* default_val = accept_operand(ctx);
                append_list(const Node*, default_vals, default_val);
            }

            if (accept_token(ctx, comma_tok))
                goto next;
        }
    }

    size_t count = entries_count_list(params);
    *parameters = nodes(arena, count, read_list(const Node*, params));
    destroy_list(params);
    if (default_values) {
        *default_values = nodes(arena, count, read_list(const Node*, default_vals));
        destroy_list(default_vals);
    }
}

typedef enum { MustQualified, MaybeQualified, NeverQualified } Qualified;

static Nodes accept_types(ctxparams, TokenTag separator, Qualified qualified) {
    struct List* tmp = new_list(Type*);
    while (true) {
        const Type* type;
        switch (qualified) {
            case MustQualified:  type = accept_qualified_type(ctx);       break;
            case MaybeQualified: type = accept_maybe_qualified_type(ctx); break;
            case NeverQualified: type = accept_unqualified_type(ctx);     break;
        }
        if (!type)
            break;

        append_list(Type*, tmp, type);

        if (separator != 0)
            accept_token(ctx, separator);
    }

    Nodes types2 = nodes(arena, tmp->elements_count, (const Type**) tmp->alloc);
    destroy_list(tmp);
    return types2;
}

static const Node* accept_primary_expr(ctxparams) {
    if (accept_token(ctx, minus_tok)) {
        const Node* expr = accept_primary_expr(ctx);
        expect(expr);
        if (expr->tag == IntLiteral_TAG) {
            return int_literal(arena, (IntLiteral) {
                // We always treat that value like an signed integer, because it makes no sense to negate an unsigned number !
                .value = -get_int_literal_value(expr, true)
            });
        } else {
            return prim_op(arena, (PrimOp) {
                .op = neg_op,
                .operands = nodes(arena, 1, (const Node* []) {expr})
            });
        }
    } else if (accept_token(ctx, unary_excl_tok)) {
        const Node* expr = accept_primary_expr(ctx);
        expect(expr);
        return prim_op(arena, (PrimOp) {
            .op = not_op,
            .operands = singleton(expr),
        });
    } else if (accept_token(ctx, star_tok)) {
        const Node* expr = accept_primary_expr(ctx);
        expect(expr);
        return prim_op(arena, (PrimOp) {
            .op = deref_op,
            .operands = singleton(expr),
        });
    }

    const Node* expr = accept_value(ctx);
    if (!expr) expr = accept_primop(ctx);
    while (expr) {
        switch (curr_token(tokenizer).tag) {
            case lpar_tok: {
                Nodes args = expect_operands(ctx);
                expr = call(arena, (Call) {
                    .callee = expr,
                    .args = args
                });
                continue;
            }
            default: break;
        }
        break;
    }
    return expr;
}

static const Node* accept_expr(ctxparams, int outer_precedence) {
    const Node* expr = accept_primary_expr(ctx);
    while (expr) {
        InfixOperators infix;
        if (is_infix_operator(curr_token(tokenizer).tag, &infix)) {
            int precedence = get_precedence(infix);
            if (precedence > outer_precedence) break;
            next_token(tokenizer);

            const Node* rhs = accept_expr(ctx, precedence - 1);
            expect(rhs);
            Op primop_op;
            if (is_primop_op(infix, &primop_op)) {
                expr = prim_op(arena, (PrimOp) {
                    .op = primop_op,
                    .operands = nodes(arena, 2, (const Node* []) {expr, rhs})
                });
            } else switch (infix) {
                default: error("unknown infix operator")
            }
            continue;
        }
        break;
    }
    return expr;
}

static Nodes expect_operands(ctxparams) {
    if (!accept_token(ctx, lpar_tok))
        error("Expected left parenthesis")

    struct List* list = new_list(Node*);

    bool expect = false;
    while (true) {
        const Node* val = accept_operand(ctx);
        if (!val) {
            if (expect)
                error("expected value but got none")
            else if (accept_token(ctx, rpar_tok))
                break;
            else
                error("Expected value or closing parenthesis")
        }

        append_list(Node*, list, val);

        if (accept_token(ctx, comma_tok))
            expect = true;
        else if (accept_token(ctx, rpar_tok))
            break;
        else
            error("Expected comma or closing parenthesis")
    }

    Nodes final = nodes(arena, list->elements_count, (const Node**) list->alloc);
    destroy_list(list);
    return final;
}

static const Node* accept_control_flow_instruction(ctxparams, Node* fn) {
    Token current_token = curr_token(tokenizer);
    switch (current_token.tag) {
        case if_tok: {
            next_token(tokenizer);
            Nodes yield_types = accept_types(ctx, 0, NeverQualified);
            expect(accept_token(ctx, lpar_tok));
            const Node* condition = accept_operand(ctx);
            expect(condition);
            expect(accept_token(ctx, rpar_tok));
            const Node* merge = config.front_end ? yield(arena, (Yield) { .args = nodes(arena, 0, NULL) }) : NULL;

            const Node* if_true = lambda(arena, nodes(arena, 0, NULL), expect_body(ctx, fn, merge));

            // else defaults to an empty body
            bool has_else = accept_token(ctx, else_tok);
            const Node* if_false = NULL;
            if (has_else) {
                if_false = lambda(arena, nodes(arena, 0, NULL), expect_body(ctx, fn, merge));
            }
            return if_instr(arena, (If) {
                .yield_types = yield_types,
                .condition = condition,
                .if_true = if_true,
                .if_false = if_false
            });
        }
        case loop_tok: {
            next_token(tokenizer);
            Nodes yield_types = accept_types(ctx, 0, NeverQualified);
            Nodes parameters;
            Nodes default_values;
            expect_parameters(ctx, &parameters, &default_values);
            // by default loops continue forever
            const Node* default_loop_end_behaviour = config.front_end ? merge_continue(arena, (MergeContinue) { .args = nodes(arena, 0, NULL) }) : NULL;
            const Node* body = lambda(arena, parameters, expect_body(ctx, fn, default_loop_end_behaviour));

            return loop_instr(arena, (Loop) {
                .initial_args = default_values,
                .yield_types = yield_types,
                .body = body
            });
        }
        case control_tok: {
            next_token(tokenizer);
            Nodes yield_types = accept_types(ctx, 0, NeverQualified);
            expect(accept_token(ctx, lpar_tok));
            String str = accept_identifier(ctx);
            expect(str);
            const Node* param = var(arena, join_point_type(arena, (JoinPointType) {
                .yield_types = yield_types,
            }), str);
            expect(accept_token(ctx, rpar_tok));
            const Node* body = lambda(arena, singleton(param), expect_body(ctx, fn, NULL));
            return control(arena, (Control) {
                .inside = body,
                .yield_types = yield_types
            });
        }
        default: break;
    }
    return NULL;
}

static bool translate_token_to_primop(TokenTag token, Op* op) {
    switch (token) {
#define TRANSLATE_PRIMOP_CASE(has_side_effects, name) case name##_tok: { *op = name##_op; return true; }
PRIMOPS(TRANSLATE_PRIMOP_CASE)
#undef TRANSLATE_PRIMOP_CASE
        default: return false;
    }
}

static const Node* accept_primop(ctxparams) {
    Op op = not_op;
    switch (curr_token(tokenizer).tag) {
        /// Only used for IR parsing
        /// Otherwise accept_expression handles this
        case call_tok: {
            next_token(tokenizer);
            expect(accept_token(ctx, lpar_tok));
            const Node* callee = accept_operand(ctx);
            expect(callee);
            expect(accept_token(ctx, rpar_tok));
            Nodes args = expect_operands(ctx);
            return call(arena, (Call) {
                .callee = callee,
                .args = args,
            });
        }
        default: if (translate_token_to_primop(curr_token(tokenizer).tag, &op)) break; else return NULL;
    }
    next_token(tokenizer);

    Nodes ty_args = nodes(arena, 0, NULL);
    if (accept_token(ctx, lsbracket_tok)) {
        while (true) {
            const Type* t = accept_unqualified_type(ctx);
            expect(t);
            ty_args = append_nodes(arena, ty_args, t);
            if (accept_token(ctx, comma_tok))
                continue;
            if (accept_token(ctx, rsbracket_tok))
                break;
        }
    }

    return prim_op(arena, (PrimOp) {
        .op = op,
        .type_arguments = ty_args,
        .operands = expect_operands(ctx)
    });
}

static const Node* accept_instruction(ctxparams, Node* fn, bool in_list) {
    const Node* instr = config.front_end ? accept_expr(ctx, max_precedence()) : accept_primop(ctx);

    if (in_list && instr)
        expect(accept_token(ctx, semi_tok) && "Non-control flow instructions must be followed by a semicolon");

    if (!instr) instr = accept_control_flow_instruction(ctx, fn);
    return instr;
}

static void expect_identifiers(ctxparams, Strings* out_strings) {
    struct List* list = new_list(const char*);
    while (true) {
        const char* id = accept_identifier(ctx);
        expect(id);

        append_list(const char*, list, id);

        if (accept_token(ctx, comma_tok))
            continue;
        else
            break;
    }

    *out_strings = strings(arena, list->elements_count, (const char**) list->alloc);
    destroy_list(list);
}

static void expect_types_and_identifiers(ctxparams, Strings* out_strings, Nodes* out_types) {
    struct List* slist = new_list(const char*);
    struct List* tlist = new_list(const char*);

    while (true) {
        const Type* type = accept_unqualified_type(ctx);
        expect(type);
        const char* id = accept_identifier(ctx);
        expect(id);

        append_list(const char*, tlist, type);
        append_list(const char*, slist, id);

        if (accept_token(ctx, comma_tok))
            continue;
        else
            break;
    }

    *out_strings = strings(arena, slist->elements_count, (const char**) slist->alloc);
    *out_types = nodes(arena, tlist->elements_count, (const Node**) tlist->alloc);
    destroy_list(slist);
    destroy_list(tlist);
}

static bool accept_non_terminator_instr(ctxparams, BodyBuilder* bb, Node* fn) {
    Strings ids;
    if (accept_token(ctx, val_tok)) {
        expect_identifiers(ctx, &ids);
        expect(accept_token(ctx, equal_tok));
        const Node* instruction = accept_instruction(ctx, fn, true);
        bind_instruction_outputs_count(bb, instruction, ids.count, ids.strings, false);
    } else if (accept_token(ctx, var_tok)) {
        Nodes types;
        expect_types_and_identifiers(ctx, &ids, &types);
        expect(accept_token(ctx, equal_tok));
        const Node* instruction = accept_instruction(ctx, fn, true);
        bind_instruction_explicit_result_types(bb, instruction, types, ids.strings, true);
    } else {
        const Node* instr = accept_instruction(ctx, fn, true);
        if (!instr) return false;
        bind_instruction_outputs_count(bb, instr, 0, NULL, false);
    }
    return true;
}

static const Node* accept_anonymous_lambda(ctxparams, Node* fn) {
    if (!accept_token(ctx, lambda_tok))
        return NULL;

    Nodes params;
    expect_parameters(ctx, &params, NULL);
    const Node* body = expect_body(ctx, fn, NULL);
    return lambda(arena, params, body);
}

static const Node* expect_jump(ctxparams) {
    String target = accept_identifier(ctx);
    expect(target);
    Nodes args = curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx) : nodes(arena, 0, NULL);
    return jump(arena, (Jump) {
            .target = unbound(arena, (Unbound) { .name = target }),
            .args = args
    });
}

static const Node* accept_terminator(ctxparams, Node* fn) {
    TokenTag tag = curr_token(tokenizer).tag;
    switch (tag) {
        case let_tok: {
            next_token(tokenizer);
            const Node* instruction = accept_instruction(ctx, fn, false);
            expect(instruction);
            expect(accept_token(ctx, in_tok));
            switch (tag) {
                case let_tok: {
                    const Node* lam = accept_anonymous_lambda(ctx, fn);
                    expect(lam);
                    return let(arena, instruction, lam);
                }
                default: SHADY_UNREACHABLE;
            }
        }
        case jump_tok: {
            next_token(tokenizer);
            return expect_jump(ctx);
        }
        case branch_tok: {
            next_token(tokenizer);

            expect(accept_token(ctx, lpar_tok));
            const Node* condition = accept_value(ctx);
            expect(condition);
            expect(accept_token(ctx, comma_tok));
            const Node* true_target = expect_jump(ctx);
            expect(accept_token(ctx, comma_tok));
            const Node* false_target = expect_jump(ctx);
            expect(accept_token(ctx, rpar_tok));

            Nodes args = curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx) : nodes(arena, 0, NULL);
            return branch(arena, (Branch) {
                .branch_condition = condition,
                .true_jump = true_target,
                .false_jump = false_target,
            });
        }
        case switch_tok: {
            next_token(tokenizer);

            expect(accept_token(ctx, lpar_tok));
            const Node* inspectee = accept_value(ctx);
            expect(inspectee);
            expect(accept_token(ctx, comma_tok));
            Nodes values = empty(arena);
            Nodes cases = empty(arena);
            const Node* default_jump;
            while (true) {
                if (accept_token(ctx, default_tok)) {
                    default_jump = expect_jump(ctx);
                    break;
                }
                expect(accept_token(ctx, case_tok));
                const Node* value = accept_value(ctx);
                expect(value);
                expect(accept_token(ctx, comma_tok) && 1);
                const Node* j = expect_jump(ctx);
                expect(accept_token(ctx, comma_tok) && true);
                values = append_nodes(arena, values, value);
                cases = append_nodes(arena, cases, j);
            }
            expect(accept_token(ctx, rpar_tok));

            return br_switch(arena, (Switch) {
                .switch_value = first(values),
                .case_values = values,
                .case_jumps = cases,
                .default_jump = default_jump,
            });
        }
        case return_tok: {
            next_token(tokenizer);
            Nodes args = expect_operands(ctx);
            return fn_ret(arena, (Return) {
                .fn = NULL,
                .args = args
            });
        }
        case yield_tok: {
            next_token(tokenizer);
            Nodes args = curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx) : nodes(arena, 0, NULL);
            return yield(arena, (Yield) {
                .args = args
            });
        }
        case continue_tok: {
            next_token(tokenizer);
            Nodes args = curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx) : nodes(arena, 0, NULL);
            return merge_continue(arena, (MergeContinue) {
                .args = args
            });
        }
        case break_tok: {
            next_token(tokenizer);
            Nodes args = curr_token(tokenizer).tag == lpar_tok ? expect_operands(ctx) : nodes(arena, 0, NULL);
            return merge_break(arena, (MergeBreak) {
                .args = args
            });
        }
        case join_tok: {
            next_token(tokenizer);
            expect(accept_token(ctx, lpar_tok));
            const Node* jp = accept_operand(ctx);
            expect(accept_token(ctx, rpar_tok));
            Nodes args = expect_operands(ctx);
            return join(arena, (Join) {
                .join_point = jp,
                .args = args
            });
        }
        case unreachable_tok: {
            next_token(tokenizer);
            expect(accept_token(ctx, lpar_tok));
            expect(accept_token(ctx, rpar_tok));
            return unreachable(arena);
        }
        default: break;
    }
    return NULL;
}

static const Node* expect_body(ctxparams, Node* fn, const Node* default_terminator) {
    assert(fn->tag == Function_TAG);
    expect(accept_token(ctx, lbracket_tok));
    BodyBuilder* bb = begin_body(arena);

    while (true) {
        if (!accept_non_terminator_instr(ctx, bb, fn))
            break;
    }

    const Node* terminator = accept_terminator(ctx, fn);

    if (terminator)
        expect(accept_token(ctx, semi_tok));

    if (!terminator) {
        if (default_terminator)
            terminator = default_terminator;
        else
            error("expected terminator: return, jump, branch ...");
    }

    if (curr_token(tokenizer).tag == cont_tok) {
        struct List* conts = new_list(Node*);
        while (true) {
            if (!accept_token(ctx, cont_tok))
                break;
            const char* name = accept_identifier(ctx);

            Nodes parameters;
            expect_parameters(ctx, &parameters, NULL);
            Node* continuation = basic_block(arena, fn, parameters, name);
            continuation->payload.fun.body = expect_body(ctx, fn, NULL);
            append_list(Node*, conts, continuation);
        }

        terminator = unbound_bbs(arena, (UnboundBBs) { .body = terminator, .children_blocks = nodes(arena, entries_count_list(conts), read_list(const Node*, conts)) });
        destroy_list(conts);
    }

    expect(accept_token(ctx, rbracket_tok));

    return finish_body(bb, terminator);
}

static Nodes accept_annotations(ctxparams) {
    struct List* list = new_list(const Node*);

    while (true) {
        if (accept_token(ctx, at_tok)) {
            const char* id = accept_identifier(ctx);
            const Node* annot = NULL;
            if (accept_token(ctx, lpar_tok)) {
                const Node* first_value = accept_value(ctx);
                if (!first_value) {
                    expect(accept_token(ctx, rpar_tok));
                    goto no_params;
                }

                // this is a map
                if (first_value->tag == Unbound_TAG && accept_token(ctx, equal_tok)) {
                    error("TODO: parse map")
                } else if (curr_token(tokenizer).tag == comma_tok) {
                    next_token(tokenizer);
                    struct List* values = new_list(const Node*);
                    append_list(const Node*, values, first_value);
                    while (true) {
                        const Node* next_value = accept_value(ctx);
                        expect(next_value);
                        append_list(const Node*, values, next_value);
                        if (accept_token(ctx, comma_tok))
                            continue;
                        else break;
                    }
                    annot = annotation_values(arena, (AnnotationValues) {
                        .name = id,
                        .values = nodes(arena, entries_count_list(values), read_list(const Node*, values))
                    });
                    destroy_list(values);
                } else {
                    annot = annotation_value(arena, (AnnotationValue) {
                        .name = id,
                        .value = first_value
                    });
                }

                expect(accept_token(ctx, rpar_tok));
            } else {
                no_params:
                annot = annotation(arena, (Annotation) {
                    .name = id,
                });
            }
            expect(annot);
            append_list(const Node*, list, annot);
            continue;
        }
        break;
    }

    Nodes annotations = nodes(arena, entries_count_list(list), read_list(const Node*, list));
    destroy_list(list);
    return annotations;
}

static const Node* accept_const(ctxparams, Nodes annotations) {
    if (!accept_token(ctx, const_tok))
        return NULL;

    const Type* type = accept_unqualified_type(ctx);
    const char* id = accept_identifier(ctx);
    expect(id);
    expect(accept_token(ctx, equal_tok));
    const Node* definition = accept_value(ctx);
    expect(definition);

    expect(accept_token(ctx, semi_tok));

    Node* cnst = constant(mod, annotations, type, id);
    cnst->payload.constant.value = definition;
    return cnst;
}

static const Node* accept_fn_decl(ctxparams, Nodes annotations) {
    if (!accept_token(ctx, fn_tok))
        return NULL;

    const char* name = accept_identifier(ctx);
    expect(name);
    Nodes types = accept_types(ctx, comma_tok, MaybeQualified);
    expect(curr_token(tokenizer).tag == lpar_tok);
    Nodes parameters;
    expect_parameters(ctx, &parameters, NULL);

    Node* fn = function(mod, parameters, name, annotations, types);
    if (!accept_token(ctx, semi_tok))
        fn->payload.fun.body = expect_body(ctx, fn, types.count == 0 ? fn_ret(arena, (Return) { .args = types }) : NULL);

    const Node* declaration = fn;
    expect(declaration);

    return declaration;
}

static const Node* accept_global_var_decl(ctxparams, Nodes annotations) {
    AddressSpace as;
    if (accept_token(ctx, private_tok))
        as = AsPrivateLogical;
    else if (accept_token(ctx, shared_tok))
        as = AsSharedLogical;
    else if (accept_token(ctx, subgroup_tok))
        as = AsSubgroupLogical;
    else if (accept_token(ctx, global_tok))
        as = AsGlobalLogical;
    else if (accept_token(ctx, extern_tok))
        as = AsExternal;
    else if (accept_token(ctx, input_tok))
        as = AsInput;
    else if (accept_token(ctx, output_tok))
        as = AsOutput;
    else if (accept_token(ctx, uniform_tok)) {
        if (accept_token(ctx, input_tok)) {
            as = AsUInput;
        } else {
            expect(false && "expected 'input'");
            return NULL;
        }
    } else
        return NULL;

    const Type* type = accept_unqualified_type(ctx);
    expect(type);
    const char* id = accept_identifier(ctx);
    expect(id);

    const Node* initial_value = NULL;
    if (accept_token(ctx, equal_tok)) {
        initial_value = accept_value(ctx);
        expect(initial_value);
    }

    expect(accept_token(ctx, semi_tok));

    Node* gv = global_var(mod, annotations, type, id, as);
    gv->payload.global_variable.init = initial_value;
    return gv;
}

static const Node* accept_nominal_type_decl(ctxparams, Nodes annotations) {
    if (!accept_token(ctx, type_tok))
        return NULL;

    const char* id = accept_identifier(ctx);
    expect(id);

    expect(accept_token(ctx, equal_tok));

    Node* nom = nominal_type(mod, annotations, id);
    nom->payload.nom_type.body = accept_unqualified_type(ctx);
    expect(nom->payload.nom_type.body);

    expect(accept_token(ctx, semi_tok));
    return nom;
}

void parse_shady_ir(ParserConfig config, const char* contents, Module* mod) {
    IrArena* arena = get_module_arena(mod);
    Tokenizer* tokenizer = new_tokenizer(contents);

    while (true) {
        Token token = curr_token(tokenizer);
        if (token.tag == EOF_tok)
            break;

        Nodes annotations = accept_annotations(ctx);

        const Node* decl = accept_const(ctx, annotations);
        if (!decl)  decl = accept_fn_decl(ctx, annotations);
        if (!decl)  decl = accept_global_var_decl(ctx, annotations);
        if (!decl)  decl = accept_nominal_type_decl(ctx, annotations);

        if (decl) {
            debugv_print("decl parsed : ");
            log_node(DEBUGV, decl);
            debugv_print("\n");
            continue;
        }

        error_print("No idea what to parse here... (tok=(tag = %s, pos = %zu))\n", token_tags[token.tag], token.start);
        exit(-3);
    }

    destroy_tokenizer(tokenizer);
}
