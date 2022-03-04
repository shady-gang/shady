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
        default: SHADY_UNREACHABLE;
    }
}

// to avoid some repetition
#define ctxparams char* contents, struct IrArena* arena, struct Tokenizer* tokenizer
#define ctx contents, arena, tokenizer

#define expect(condition) expect_impl(condition, #condition)
void expect_impl(bool condition, const char* err) {
    if (!condition) {
        fprintf(stderr, "expected to parse: %s\n", err);
        exit(-4);
    }
}

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

    const struct Type* parsed_type = NULL;

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
    expect(accept_token(ctx, lpar_tok));
    struct List* params = new_list(struct Variable*);
    while (true) {
        if (accept_token(ctx, rpar_tok))
            break;
        const struct Type* type = accept_type(ctx, Unknown);
        expect(type);
        const char* id = accept_identifier(ctx);
        expect(id);

        const struct Node* node = var(arena, (struct Variable) {
            .name = id,
            .type = type
        });

        append_list(struct Variable*, params, node->payload.var);

        if (accept_token(ctx, comma_tok))
            continue;
        expect(accept_token(ctx, rpar_tok));
    }

    struct Variables variables2 = variables(arena, params->elements, (const struct Variable**) params->alloc);
    destroy_list(params);
    return variables2;
}

struct Nodes eat_block(ctxparams) {
    expect(accept_token(ctx, lbracket_tok));
    expect(accept_token(ctx, rbracket_tok));
    return nodes(arena, 0, NULL);
}

const struct Node* eat_decl(ctxparams) {
    const struct Type* type = accept_type(ctx, Uniform);
    expect(type);
    const char* id = accept_identifier(ctx);
    expect(id);

    if (curr_token(tokenizer).tag == lpar_tok) {
        struct Variables parameters = eat_parameters(ctx);

        if (accept_token(ctx, semi_tok)) {
            const struct Type* param_types[parameters.count];
            for (size_t i = 0; i < parameters.count; i++)
                param_types[i] = parameters.variables[i]->type;

            const struct Type* ret_type = type;
            type = fn_type(arena, true, types(arena, parameters.count, param_types), ret_type);
        } else {
            struct Nodes instructions = eat_block(ctx);

            return fn(arena, (struct Function) {
                .name = id,
                .params = parameters,
                .return_type = type,
                .instructions = instructions
            });
        }
    }

    expect(accept_token(ctx, semi_tok));

    return var_decl(arena, (struct VariableDecl) {
        .address_space = Private,
        .variable = var(arena, (struct Variable) {
            .type = type,
            .name = id
        }),
       .init = NULL
    });
}

struct Program parse(char* contents, struct IrArena* arena) {
    struct Tokenizer* tokenizer = new_tokenizer(contents);

    struct List* top_level = new_list(struct Node*);

    while (true) {
        struct Token token = curr_token(tokenizer);
        if (token.tag == EOF_tok)
            break;

        const struct Node* decl = eat_decl(ctx);
        if (decl) {
            printf("decl parsed\n");
            append_list(struct Node*, top_level, decl);
            continue;
        }

        printf("No idea what to parse here... (tok=(tag = %d, pos = %zu))\n", token.tag, token.start);
        exit(-3);
    }

    struct Nodes top_level_nodes = nodes(arena, top_level->elements, read_list(const struct Node*, top_level));
    destroy_list(top_level);
    destroy_tokenizer(tokenizer);

    return (struct Program) {
        .declarations_and_definitions = top_level_nodes,
    };
}
