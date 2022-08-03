#ifndef SHADY_PARSER_H

#define SHADY_PARSER_H

#include "shady/ir.h"

typedef struct {
    bool front_end;
} ParserConfig;

#define INFIX_OPERATORS() \
INFIX_OPERATOR(Mul, star_tok,                 mul_op,            1) \
INFIX_OPERATOR(Sub, minus_tok,                sub_op,            1) \
INFIX_OPERATOR(Add, plus_tok,                 add_op,            2) \
INFIX_OPERATOR(RSL, infix_rshift_logical_tok, rshift_logical_op, 3) \
INFIX_OPERATOR(RSA, infix_rshift_arithm_tok,  rshift_arithm_op,  3) \
INFIX_OPERATOR(LS,  infix_lshift_tok,         lshift_op,         3) \
INFIX_OPERATOR(And, infix_and_tok,            and_op,            4) \
INFIX_OPERATOR(Xor, infix_xor_tok,            xor_op,            5) \
INFIX_OPERATOR(Or,  infix_or_tok,             or_op,             6) \
INFIX_OPERATOR(Eq,  infix_eq_tok,             eq_op,             7) \
INFIX_OPERATOR(Neq, infix_neq_tok,            neq_op,            7) \
INFIX_OPERATOR(Gt,  infix_gt_tok,             gt_op,             7) \
INFIX_OPERATOR(Ge,  infix_geq_tok,            gte_op,            7) \
INFIX_OPERATOR(Lt,  infix_ls_tok,             lt_op,             7) \
INFIX_OPERATOR(Le,  infix_leq_tok,            lte_op,            7) \
INFIX_OPERATOR(Ass, equal_tok,                assign_op,        10) \

typedef enum {
#define INFIX_OPERATOR(name, token, primop, precedence) Infix##name,
INFIX_OPERATORS()
#undef INFIX_OPERATOR
    InfixOperatorsCount
} InfixOperators;

const Node* parse(ParserConfig config, char* contents, IrArena* arena);

#endif
