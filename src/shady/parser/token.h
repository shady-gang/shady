#ifndef SHADY_TOKEN_H

#include "shady/ir.h"
#include <stddef.h>

#define TEXT_TOKEN(t) TOKEN(t, #t)

#define REGISTER_PRIMOP_AS_TOKEN(has_side_effects, name) TEXT_TOKEN(name)

#define TOKENS() \
TOKEN(EOF, NULL) \
TOKEN(identifier, NULL) \
TOKEN(dec_lit, NULL) \
TOKEN(hex_lit, NULL) \
TOKEN(string_lit, NULL) \
TEXT_TOKEN(uniform) \
TEXT_TOKEN(varying) \
TOKEN(struct, "struct") \
TOKEN(union, "union") \
TOKEN(private, "private") \
TEXT_TOKEN(shared) \
TEXT_TOKEN(subgroup) \
TEXT_TOKEN(global) \
TEXT_TOKEN(input) \
TEXT_TOKEN(output) \
TOKEN(extern, "extern") \
TEXT_TOKEN(var) \
TEXT_TOKEN(let) \
TEXT_TOKEN(ptr) \
TEXT_TOKEN(fn) \
TEXT_TOKEN(type) \
TEXT_TOKEN(cont) \
TEXT_TOKEN(i8) \
TEXT_TOKEN(i16) \
TEXT_TOKEN(i32) \
TEXT_TOKEN(i64) \
TEXT_TOKEN(float) \
TEXT_TOKEN(mask) \
TOKEN(const, "const") \
TOKEN(bool, "bool") \
TOKEN(true, "true") \
TOKEN(false, "false") \
TOKEN(if, "if") \
TOKEN(else, "else") \
TEXT_TOKEN(merge) \
TEXT_TOKEN(loop) \
TOKEN(continue, "continue") \
TOKEN(break, "break") \
TEXT_TOKEN(jump) \
TEXT_TOKEN(branch) \
TEXT_TOKEN(join) \
TEXT_TOKEN(call) \
TEXT_TOKEN(callf) \
TEXT_TOKEN(callc) \
TOKEN(return, "return") \
TEXT_TOKEN(unreachable) \
PRIMOPS(REGISTER_PRIMOP_AS_TOKEN) \
TOKEN(infix_rshift_logical, ">>>") \
TOKEN(infix_rshift_arithm, ">>") \
TOKEN(infix_lshift, "<<") \
TOKEN(infix_eq, "==") \
TOKEN(infix_neq, "!=") \
TOKEN(infix_geq, ">=") \
TOKEN(infix_leq, "<=") \
TOKEN(infix_gt, ">") \
TOKEN(infix_ls, "<") \
TOKEN(infix_and, "&") \
TOKEN(infix_xor, "^") \
TOKEN(infix_or, "|") \
TOKEN(plus, "+") \
TOKEN(minus, "-") \
TOKEN(star, "*") \
TOKEN(fslash, "/") \
TOKEN(lpar, "(") \
TOKEN(rpar, ")") \
TOKEN(lbracket, "{") \
TOKEN(rbracket, "}") \
TOKEN(lsbracket, "[") \
TOKEN(rsbracket, "]") \
TOKEN(at, "@") \
TOKEN(dot, ".") \
TOKEN(semi, ";") \
TOKEN(colon, ":") \
TOKEN(comma, ",") \
TOKEN(equal, "=") \
TOKEN(LIST_END, NULL)

typedef struct Tokenizer_ Tokenizer;
Tokenizer* new_tokenizer(const char* source);
void destroy_tokenizer(Tokenizer*);

typedef enum {
#define TOKEN(name, str) name##_tok,
    TOKENS()
#undef TOKEN
} TokenTag;

extern const char* token_tags[];

typedef struct {
    TokenTag tag;
    size_t start;
    size_t end;
} Token;

Token curr_token(Tokenizer* tokenizer);
Token next_token(Tokenizer* tokenizer);

#define SHADY_TOKEN_H

#endif //SHADY_TOKEN_H
