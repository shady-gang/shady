#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "ir.h"
#include "token.h"

#include "../containers/list.h"

#include "../implem.h"

extern const char* token_tags[];

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
        return string_sized(arena, (int) size, &contents[tok.start]);
    }
    return NULL;
}

const struct Type* expect_unqualified_type(ctxparams) {
    const struct Type* parsed_type = NULL;

    if (accept_token(ctx, int_tok)) {
        parsed_type = int_type(arena);
    } else if (accept_token(ctx, float_tok)) {
        parsed_type = float_type(arena);
    } else if (accept_token(ctx, void_tok)) {
        parsed_type = void_type(arena);
    } else if (accept_token(ctx, ptr_tok)) {
        SHADY_NOT_IMPLEM
    } else if (accept_token(ctx, fn_tok)) {
        SHADY_NOT_IMPLEM
    } else {
        error("expected any of: int float void fn ptr")
    }

    return parsed_type;
}

enum DivergenceQualifier accept_uniformity_qualifier(ctxparams) {
    enum DivergenceQualifier divergence = Unknown;
    if (accept_token(ctx, uniform_tok))
        divergence = Uniform;
    else if (accept_token(ctx, varying_tok))
        divergence = Varying;
    return divergence;
}

const struct Type* expect_maybe_qualified_type(ctxparams) {
    enum DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    const struct Type* unqualified = expect_unqualified_type(ctx);
    if (qualifier == Unknown)
        return unqualified;
    else
        return qualified_type(arena, qualifier == Uniform, unqualified);
}

const struct Type* expect_qualified_type(ctxparams) {
    enum DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    expect(qualifier != Unknown);
    const struct Type* unqualified = expect_unqualified_type(ctx);
    return qualified_type(arena, qualifier == Uniform, unqualified);
}

struct Nodes eat_parameters(ctxparams) {
    expect(accept_token(ctx, lpar_tok));
    struct List* params = new_list(struct Node*);
    while (true) {
        if (accept_token(ctx, rpar_tok))
            break;

        next: {
            const struct Type* type = expect_qualified_type(ctx);
            expect(type);
            const char* id = accept_identifier(ctx);
            expect(id);

            const struct Node* node = var(arena, (struct Variable) {
                .name = id,
                .type = type
            });

            append_list(struct Node*, params, node);

            if (accept_token(ctx, comma_tok))
                goto next;
        }
    }

    struct Nodes variables2 = nodes(arena, params->elements_count, (const struct Node**) params->alloc);
    destroy_list(params);
    return variables2;
}

struct Types eat_parameter_types(ctxparams) {
    expect(accept_token(ctx, lpar_tok));
    struct List* params = new_list(struct Type*);
    while (true) {
        if (accept_token(ctx, rpar_tok))
            break;

        next: {
            const struct Type* type = expect_qualified_type(ctx);
            expect(type);

            append_list(struct Type*, params, type);

            if (accept_token(ctx, comma_tok))
                goto next;
        }
    }

    struct Types types2 = types(arena, params->elements_count, (const struct Type**) params->alloc);
    destroy_list(params);
    return types2;
}

const struct Node* accept_literal(ctxparams) {
    struct Token tok = curr_token(tokenizer);
    switch (tok.tag) {
        case dec_lit_tok: {
            next_token(tokenizer);
            size_t size = tok.end - tok.start;
            return untyped_number(arena, (struct UntypedNumber) {
                .plaintext = string_sized(arena, (int) size, &contents[tok.start])
            });
            //int64_t value = strtol(&contents[tok.start], NULL, 10)
            //return untyped_number(value);
        }

        default: return NULL;
    }
}

const struct Node* accept_value(ctxparams) {
    const char* id = accept_identifier(ctx);
    if (id) {
        return var(arena, (struct Variable) {
            .type = NULL,
            .name = id
        });
    }

    return accept_literal(ctx);
}

struct Strings eat_identifiers(ctxparams) {
    struct List* list = new_list(const char*);
    while (true) {
        const char* id = accept_identifier(ctx);
        expect(id);

        append_list(const char*, list, id);

        if (accept_token(ctx, comma_tok))
            continue;
        else
            break;
    }

    struct Strings final = strings(arena, list->elements_count, (const char**) list->alloc);
    destroy_list(list);
    return final;
}

struct Nodes eat_values(ctxparams, enum TokenTag separator) {
    struct List* list = new_list( struct Node*);

    bool expect = false;
    while (true) {
        const struct Node* val = accept_value(ctx);
        if (!val) {
            if (expect)
                error("expected value but got none")
            else
                break;
        }

        append_list(struct Node*, list, val);

        if (separator != 0) {
            if (accept_token(ctx, separator))
                expect = true;
            else
                break;
        }
    }

    struct Nodes final = nodes(arena, list->elements_count, (const struct Node**) list->alloc);
    destroy_list(list);
    return final;
}

const struct Node* eat_computation(ctxparams) {
    struct Token tok = curr_token(tokenizer);
    switch (tok.tag) {
        case add_tok: {
            next_token(tokenizer);
            struct Nodes args = eat_values(ctx, 0);
            expect(accept_token(ctx, semi_tok));
            return primop(arena, (struct PrimOp) {
                .op = add_op,
                .args = args
            });
        }
        default: error("cannot parse a computation");
    }
}

const struct Node* accept_instruction(ctxparams) {
    struct Token current_token = curr_token(tokenizer);
    switch (current_token.tag) {
        case return_tok: {
            next_token(tokenizer);
            struct Nodes values = eat_values(ctx, 0);
            expect(accept_token(ctx, semi_tok));
            return fn_ret(arena, (struct Return) {
                .values = values
            });
        }
        case let_tok: {
            next_token(tokenizer);
            struct Strings ids = eat_identifiers(ctx);
            expect(accept_token(ctx, equal_tok));
            const struct Node* comp = eat_computation(ctx);
            return let(arena, (struct Let) {
                .names = ids,
                .target = comp
            });
        }
        default: break;
    }
    return NULL;
}

struct Nodes eat_block(ctxparams) {
    expect(accept_token(ctx, lbracket_tok));
    struct List* instructions = new_list(struct Node*);
    while (true) {
        if (accept_token(ctx, rbracket_tok))
            break;

        const struct Node* instruction = accept_instruction(ctx);
        if (instruction)
            append_list(struct Node*, instructions, instruction);
        else {
            expect(accept_token(ctx, rbracket_tok));
            break;
        }
    }
    struct Nodes block = nodes(arena, entries_count_list(instructions), read_list(const struct Node*, instructions));
    destroy_list(instructions);
    return block;
}

const struct Node* accept_fn_decl(ctxparams) {
    if (!accept_token(ctx, fn_tok))
        return NULL;

    const struct Type* type = expect_maybe_qualified_type(ctx);
    expect(type);
    const char* id = accept_identifier(ctx);
    expect(id);
    expect(curr_token(tokenizer).tag == lpar_tok);
    struct Nodes parameters = eat_parameters(ctx);
    struct Nodes instructions = eat_block(ctx);
    return fn(arena, (struct Function) {
        .name = id,
        .params = parameters,
        .return_type = type,
        .instructions = instructions
    });
}

const struct Node* accept_var_decl(ctxparams) {
    if (!accept_token(ctx, var_tok))
        return NULL;

    const struct Type* type = expect_maybe_qualified_type(ctx);
    expect(type);
    const char* id = accept_identifier(ctx);
    expect(id);

    // TODO accept init
    expect(accept_token(ctx, semi_tok));

    return var_decl(arena, (struct VariableDecl) {
        .address_space = AsPrivate,
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

        const struct Node* decl = accept_fn_decl(ctx);
        if (!decl)
            decl = accept_var_decl(ctx);
        
        if (decl) {
            printf("decl parsed :");
            print_node(decl, true);
            printf("\n");
            append_list(struct Node*, top_level, decl);
            continue;
        }

        printf("No idea what to parse here... (tok=(tag = %s, pos = %zu))\n", token_tags[token.tag], token.start);
        exit(-3);
    }

    struct Nodes top_level_nodes = nodes(arena, top_level->elements_count, read_list(const struct Node*, top_level));
    destroy_list(top_level);
    destroy_tokenizer(tokenizer);

    return (struct Program) {
        .declarations_and_definitions = top_level_nodes,
    };
}
