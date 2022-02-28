#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "ir.h"
#include "token.h"

#include "../containers/list.h"

#include "../implem.h"

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

// to avoid some repetition
#define ctxparams char* contents, struct IrArena* arena, struct Tokenizer* tokenizer
#define ctx contents, arena, tokenizer

bool accept_token(ctxparams, enum TokenTag tag) {
    if (curr_token(tokenizer).tag == tag) {
        next_token(tokenizer);
        return true;
    }
    return false;
}

const char* accept_identifier(ctxparams) {
    struct Token tok = curr_token(tokenizer);
    if (tok.tag == identifier_tok) {
        next_token(tokenizer);
        size_t size = tok.end - tok.start;
        return string(arena, (int) size, &contents[tok.start]);
    }
    return NULL;
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
    } else if (accept_token(ctx, void_tok)) {
        parsed_type = void_type(arena);
    }

    if (expect_type && parsed_type == NULL)
        assert(false);
    else
        return parsed_type;
}

struct Variables eat_parameters(ctxparams) {
    assert(accept_token(ctx, lpar_tok));
    struct List* params = new_list(struct Variable*);
    while (true) {
        if (accept_token(ctx, rpar_tok))
            break;
        const struct Type* type = accept_type(ctx, Unknown);
        assert(type);
        const char* id = accept_identifier(ctx);
        assert(id);

        const struct Node* node = var(arena, (struct Variable) {
            .name = id,
            .type = type
        });

        append_list(struct Variable*, params, node->payload.var);

        if (accept_token(ctx, comma_tok))
            continue;
        assert(accept_token(ctx, rpar_tok));
    }

    struct Variables variables2 = variables(arena, params->elements, (struct Variable**) params->alloc);
    destroy_list(params);
    return variables2;
}

const struct Node* accept_decl(ctxparams) {
    if (accept_token(ctx, fn_tok)) {
        const struct Type* return_type = accept_type(ctx, Uniform);
        assert(return_type);
        const char* id = accept_identifier(ctx);
        assert(id);
        struct Variables parameters = eat_parameters(ctx);

        return fn(arena, (struct Function){
            .params = parameters,
            .return_type = return_type,
            .instructions = nodes(arena, 0, NULL)
        });
    }
    return NULL;
}

void parse(char* contents, struct IrArena* arena) {
    struct Tokenizer* tokenizer = new_tokenizer(contents);

    while (true) {
        struct Token token = curr_token(tokenizer);
        if (token.tag == EOF_tok)
            break;

        /*char* id = accept_identifier(ctx);
        if (id) {
            printf("parsed identifier: %s\n", id);
            continue;
        }*/
        const struct Node* decl = accept_decl(ctx);
        if (decl) {
            printf("decl parsed\n");
        }

        printf("No idea what to parse here... (tok=(tag = %d, pos = %zu))", token.tag, token.start);
        exit(-3);
    }
}
