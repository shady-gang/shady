#ifndef SHADY_PARSER_H

#define SHADY_PARSER_H

#include "shady/ir.h"

typedef struct {
    bool front_end;
} ParserConfig;

#define INFIX_OPERATORS() \
INFIX_OPERATOR(Add, plus_tok, add_op, 2) \
INFIX_OPERATOR(Mul, star_tok, mul_op, 1) \
INFIX_OPERATOR(Eq, infix_eq_tok, eq_op, 7) \
INFIX_OPERATOR(Assign, equal_tok, -1, 10) \

typedef enum {
#define INFIX_OPERATOR(name, token, primop, precedence) Infix##name,
INFIX_OPERATORS()
#undef INFIX_OPERATOR
    InfixOperatorsCount
} InfixOperators;

const Node* parse(ParserConfig config, char* contents, IrArena* arena);

#endif