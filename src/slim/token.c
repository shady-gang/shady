#include "token.h"

#include "../log.h"

#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

static const char* token_strings[] = {
#define TOKEN(name, str) str,
        TOKENS()
#undef TOKEN
};

const char* token_tags[] = {
#define TOKEN(name, str) #name,
        TOKENS()
#undef TOKEN
};

static size_t token_strings_size[LIST_END_tok];
static bool constants_initialized = false;

static void init_tokenizer_constants() {
    for (int i = 0; i < LIST_END_tok; i++) {
        token_strings_size[i] = token_strings[i] == NULL ? -1U : strlen(token_strings[i]);
    }
}

struct Tokenizer {
    char* str;
    size_t original_size;
    size_t pos;
    struct Token current;
};

struct Tokenizer* new_tokenizer(char* str) {
    if (!constants_initialized) {
        init_tokenizer_constants();
        constants_initialized = true;
    }

    struct Tokenizer* tokenizer = (struct Tokenizer*) malloc(sizeof(struct Tokenizer));
    *tokenizer = (struct Tokenizer) {
        .str = str,
        .original_size = strlen(str),
        .pos = 0
    };
    next_token(tokenizer);
    return tokenizer;
}

void destroy_tokenizer(struct Tokenizer* tokenizer) {
    free(tokenizer);
}

const char whitespace[] = { ' ', '\t', '\b', '\n' };

static inline bool is_alpha(char c) { return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z'; }
static inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
static inline bool is_whitespace(char c) { for (size_t i = 0; i < sizeof(whitespace); i++) { if(c == whitespace[i]) return true; } return false; }
static inline bool can_start_identifier(char c) { return is_alpha(c) || c == '_'; }
static inline bool can_make_up_identifier(char c) { return can_start_identifier(c) || is_digit(c); }

static void eat_whitespace_and_comments(struct Tokenizer* tokenizer) {
    while (tokenizer->pos < tokenizer->original_size) {
        if (is_whitespace(tokenizer->str[tokenizer->pos])) {
            tokenizer->pos++;
        } else if (tokenizer->pos + 2 <= tokenizer->original_size && tokenizer->str[tokenizer->pos] == '/' && tokenizer->str[tokenizer->pos + 1] == '/') {
            while (tokenizer->pos < tokenizer->original_size) {
                if (tokenizer->str[tokenizer->pos] == '\n')
                    break;
                tokenizer->pos++;
            }
        } else
            break;
    }
}

struct Token next_token(struct Tokenizer* tokenizer) {
    eat_whitespace_and_comments(tokenizer);
    if (tokenizer->pos == tokenizer->original_size) {
        debug_print("EOF\n");
        struct Token token = (struct Token) {
            .tag = EOF_tok
        };
        tokenizer->current = token;
        return token;
    }

    assert(tokenizer->pos <= tokenizer->original_size);

    struct Token token = {
        .start = tokenizer->pos,
    };

    const char* slice = &tokenizer->str[tokenizer->pos];

    size_t token_size = 0;
    // First, try to do alphanumeric tokenization
    bool can_be_identifier = false;
    if (can_start_identifier(slice[0])) {
        can_be_identifier = true;
        while (tokenizer->pos + token_size <= tokenizer->original_size) {
            if (can_make_up_identifier(slice[token_size])) {
                token_size++;
            } else break;
        }
    } else if (is_digit(slice[0])) {
        token.tag = dec_lit_tok;

        if (slice[0] == '0' && slice[1] == 'x') {
            token.tag = hex_lit_tok;
            token_size += 2;
            // slice = &slice[2];
        }

        while (is_digit(slice[token_size])) {
            token_size++;
        }
        goto parsed_successfully;
    }

    for (int i = 0; i < LIST_END_tok; i++) {
        size_t tok_size = token_strings_size[i];
        // if there is a match ...
        if (tokenizer->pos + tok_size <= tokenizer->original_size && strncmp(token_strings[i], slice, tok_size) == 0) {
            // if this is an identifier, we need the size to match exactly
            if (!can_be_identifier || tok_size == token_size) {
                token.tag = i;
                token_size = tok_size;
                goto parsed_successfully;
            }
        }
    }

    // If it was like an identifier, but wasn't a token, then we consider it an identifier !
    if (can_be_identifier) {
        token.tag = identifier_tok;
        goto parsed_successfully;
    }

    error_print("We don't know how to tokenize %.16s...\n", slice);
    exit(-2);

    parsed_successfully:
    token.end = token.start + token_size;

    tokenizer->pos += token_size;
    tokenizer->current = token;

    debug_print("Token parsed: (tag = %s, pos = %zu", token_tags[token.tag], token.start);
    if (token.tag == identifier_tok) {
        debug_print(", str=");
        for (size_t i = token.start; i < token.end; i++)
            debug_print("%c", tokenizer->str[i]);
    }
    debug_print(")\n");
    return token;
}

struct Token curr_token(struct Tokenizer* tokenizer) {
    return tokenizer->current;
}
