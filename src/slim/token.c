#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "token.h"

static const char* token_strings[] = {
#define TOKEN(name, str) str,
        TOKENS()
#undef TOKEN
};

static size_t token_strings_size[LIST_END_tok];
void init_tokenizer_constants() {
    for (int i = 0; i < LIST_END_tok; i++) {
        token_strings_size[i] = token_strings[i] == NULL ? -1 : strlen(token_strings[i]);
    }
}

const char whitespace[] = " \t\b\n";

struct Tokenizer {
    char* str;
    char* rest;
    char* original;
    size_t remaining;
    struct Token current;
};

struct Tokenizer* new_tokenizer(char* str) {
    struct Tokenizer* tokenizer = (struct Tokenizer*) malloc(sizeof(struct Tokenizer));
    *tokenizer = (struct Tokenizer) {
        .str = NULL,
        .rest = str,
        .original = str,
        .remaining = 0,
    };
    next_token(tokenizer);
    return tokenizer;
}

bool is_alpha(char c) {
    return c >= 'A' && c <= 'z';
}

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

bool can_start_identifier(char c) {
    return is_alpha(c) || c == '_';
}
bool can_make_up_identifier(char c) {
    return can_start_identifier(c) || is_digit(c);
}

struct Token next_token(struct Tokenizer* tokenizer) {
    if (tokenizer->remaining == 0) {
        tokenizer->str = strtok(tokenizer->rest, whitespace);
        if (tokenizer->str == NULL) {
            return (struct Token) {
                .tag = EOF_tok
            };
        }
        tokenizer->remaining = strlen(tokenizer->str);
        tokenizer->rest = tokenizer->str + tokenizer->remaining + 1;
    }

    struct Token token = {
        .start = (size_t)tokenizer->str - (size_t)tokenizer->original,
    };

    size_t token_size = 0;
    // First, try to do alphanumeric tokenization
    bool can_be_identifier = false;
    if (can_start_identifier(tokenizer->str[0])) {
        can_be_identifier = true;
        while (token_size <= tokenizer->remaining) {
            if (can_make_up_identifier(tokenizer->str[token_size])) {
                token_size++;
            } else break;
        }
    }

    for (int i = 0; i < LIST_END_tok; i++) {
        size_t tok_size = token_strings_size[i];
        if (tokenizer->remaining >= tok_size && strncmp(token_strings[i], tokenizer->str, tok_size) == 0) {
            token.tag = i;
            token_size = tok_size;
            goto parsed_successfully;
        }
    }

    // If it was like an identifier, but wasn't a token, then we consider it an identifier !
    if (can_be_identifier) {
        token.tag = identifier_tok;
        goto parsed_successfully;
    }

    printf("We don't know how to tokenize %s...\n", tokenizer->str);
    exit(-2);

    parsed_successfully:
    token.end = token.start + token_size;
    tokenizer->remaining -= token_size;
    tokenizer->str+= token_size;
    tokenizer->current = token;
    return token;
}

struct Token curr_token(struct Tokenizer* tokenizer) {
    return tokenizer->current;
}
