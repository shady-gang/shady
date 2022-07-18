#ifndef SHADY_TOKEN_H

#include <stddef.h>

#define TEXT_TOKEN(t) TOKEN(t, #t)

#define TOKENS() \
TOKEN(EOF, NULL) \
TOKEN(identifier, NULL) \
TOKEN(dec_lit, NULL) \
TOKEN(hex_lit, NULL) \
TEXT_TOKEN(uniform) \
TEXT_TOKEN(varying) \
TEXT_TOKEN(struct) \
TEXT_TOKEN(union) \
TEXT_TOKEN(private) \
TEXT_TOKEN(shared) \
TEXT_TOKEN(subgroup) \
TEXT_TOKEN(global) \
TEXT_TOKEN(input) \
TEXT_TOKEN(output) \
TEXT_TOKEN(extern) \
TOKEN(compute, "@compute") \
TEXT_TOKEN(var) \
TEXT_TOKEN(let) \
TEXT_TOKEN(ptr) \
TEXT_TOKEN(fn) \
TEXT_TOKEN(cont) \
TEXT_TOKEN(int) \
TEXT_TOKEN(float) \
TEXT_TOKEN(mask) \
TEXT_TOKEN(return) \
TEXT_TOKEN(const) \
TEXT_TOKEN(add) \
TEXT_TOKEN(sub) \
TEXT_TOKEN(mul) \
TEXT_TOKEN(div) \
TEXT_TOKEN(mod) \
TEXT_TOKEN(lt) \
TEXT_TOKEN(lte) \
TEXT_TOKEN(eq) \
TEXT_TOKEN(neq) \
TEXT_TOKEN(gt) \
TEXT_TOKEN(gte) \
TEXT_TOKEN(and) \
TEXT_TOKEN(or) \
TEXT_TOKEN(xor) \
TEXT_TOKEN(not) \
TEXT_TOKEN(bool) \
TEXT_TOKEN(true) \
TEXT_TOKEN(false) \
TEXT_TOKEN(if) \
TEXT_TOKEN(else) \
TEXT_TOKEN(merge) \
TEXT_TOKEN(loop) \
TEXT_TOKEN(continue) \
TEXT_TOKEN(break) \
TEXT_TOKEN(jump) \
TEXT_TOKEN(branch) \
TEXT_TOKEN(join) \
TEXT_TOKEN(call) \
TEXT_TOKEN(callf) \
TEXT_TOKEN(callc) \
TEXT_TOKEN(load) \
TEXT_TOKEN(store) \
TEXT_TOKEN(alloca) \
TEXT_TOKEN(unreachable) \
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
TOKEN(semi, ";") \
TOKEN(colon, ":") \
TOKEN(comma, ",") \
TOKEN(equal, "=") \
TOKEN(LIST_END, NULL)

struct Tokenizer;
struct Tokenizer* new_tokenizer(char* str);
void destroy_tokenizer(struct Tokenizer*);

enum TokenTag {
#define TOKEN(name, str) name##_tok,
    TOKENS()
#undef TOKEN
};

extern const char* token_tags[];

struct Token {
    enum TokenTag tag;
    size_t start;
    size_t end;
};

struct Token curr_token(struct Tokenizer* tokenizer);
struct Token next_token(struct Tokenizer* tokenizer);

#define SHADY_TOKEN_H

#endif //SHADY_TOKEN_H
