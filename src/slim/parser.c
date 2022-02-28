#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "ir.h"
#include "token.h"

#include "../implem.h"

// to avoid some repetition
#define ctxparams char* contents, struct IrArena* arena, struct Tokenizer* tokenizer
#define ctx contents, arena, tokenizer

bool accept_token(ctxparams, enum TokenTag tag) {
    if (curr_token(tokenizer).tag == tag) {
        next_token(tokenizer);
        return true;
    }
    return false;
};

const char* accept_identifier(ctxparams) {
    struct Token tok = curr_token(tokenizer);
    if (tok.tag == identifier_tok) {
        next_token(tokenizer);
        size_t size = tok.end - tok.start;
        return string(arena, (int) size, &contents[tok.start]);
    }
    return NULL;
}

enum ParsedDivergence {
    Unknown,
    Uniform,
    Varying
};

bool is_uniform_or_error(enum ParsedDivergence divergence) {
    switch (divergence) {
        case Unknown: error("We need an explicit divergence annotation");
        case Uniform: return true;
        case Varying: return false;
    }
}

const struct Type* accept_type(ctxparams, enum ParsedDivergence divergence_hint) {
    enum ParsedDivergence divergence = divergence_hint;
    // set when we have started parsing something
    bool expect_type = false;

    if (accept_token(ctx, uniform_tok)) {
        if (divergence == Varying)
            error("conflicting divergence annotations")
        divergence = Uniform;
        expect_type = true;
    } else if (accept_token(ctx, varying_tok)) {
        if (divergence == Uniform)
           error("conflicting divergence annotations")
        divergence = Varying;
        expect_type = true;
    }

    struct Type* parsed_type = NULL;

    if (accept_token(ctx, int_tok)) {
        parsed_type = int_type(arena, is_uniform_or_error(divergence));
    } else if (accept_token(ctx, float_tok)) {
        parsed_type = float_type(arena, is_uniform_or_error(divergence));
    }

    if (expect_type && parsed_type == NULL)
        assert(false);
    else
        return parsed_type;
}

const struct Node* accept_function(ctxparams) {
    if (accept_token(ctx, fn_tok)) {
        struct Type* return_type = accept_type(ctx);
        assert(return_type);
    }
    return NULL;
}

void parse(char* contents, struct IrArena* arena) {
    struct Tokenizer* tokenizer = new_tokenizer(contents);

    while (true) {
        struct Token token = curr_token(tokenizer);
        if (token.tag == EOF_tok)
            break;

        char* id = accept_identifier(ctx);
        if (id) {
            printf("parsed identifier: %s\n", id);
            continue;
        }

        printf("No idea what to parse here... (tok=(tag = %d, pos = %zu))", token.tag, token.start);
        exit(-3);
    }
}
