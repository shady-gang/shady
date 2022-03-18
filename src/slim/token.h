#ifndef SHADY_TOKEN_H

#include <stddef.h>

#define TOKENS() \
TOKEN(EOF, NULL) \
TOKEN(identifier, NULL) \
TOKEN(dec_lit, NULL) \
TOKEN(hex_lit, NULL) \
TOKEN(uniform, "uniform") \
TOKEN(varying, "varying") \
TOKEN(struct, "struct") \
TOKEN(union, "union") \
TOKEN(private, "private") \
TOKEN(shared, "shared") \
TOKEN(global, "global") \
TOKEN(input, "input") \
TOKEN(output, "output") \
TOKEN(var, "var") \
TOKEN(let, "let") \
TOKEN(def, "def") \
TOKEN(ptr, "ptr") \
TOKEN(fn, "fn") \
TOKEN(cont, "cont") \
TOKEN(int, "int") \
TOKEN(float, "float") \
TOKEN(return, "return") \
TOKEN(const, "const") \
TOKEN(add, "add") \
TOKEN(sub, "sub") \
TOKEN(lpar, "(") \
TOKEN(rpar, ")") \
TOKEN(lbracket, "{") \
TOKEN(rbracket, "}") \
TOKEN(semi, ";") \
TOKEN(comma, ",") \
TOKEN(equal, "=") \
TOKEN(LIST_END, NULL)

void init_tokenizer_constants();

struct Tokenizer;
struct Tokenizer* new_tokenizer(char* str);
void destroy_tokenizer(struct Tokenizer*);

enum TokenTag {
#define TOKEN(name, str) name##_tok,
    TOKENS()
#undef TOKEN
};

struct Token {
    enum TokenTag tag;
    size_t start;
    size_t end;
};

struct Token curr_token(struct Tokenizer* tokenizer);
struct Token next_token(struct Tokenizer* tokenizer);

#define SHADY_TOKEN_H

#endif //SHADY_TOKEN_H
