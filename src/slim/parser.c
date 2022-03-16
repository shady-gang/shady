#include "token.h"

#include "../containers/list.h"

#include "ir.h"
#include "../implem.h"
#include "../type.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

extern const char* token_tags[];

// to avoid some repetition
#define ctxparams char* contents, IrArena* arena, struct Tokenizer* tokenizer
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

const Type* accept_unqualified_type(ctxparams) {
    if (accept_token(ctx, int_tok)) {
        return int_type(arena);
    } else if (accept_token(ctx, float_tok)) {
        return float_type(arena);
    } else if (accept_token(ctx, ptr_tok)) {
        SHADY_NOT_IMPLEM
    } else if (accept_token(ctx, fn_tok)) {
        SHADY_NOT_IMPLEM
    } else {
        return NULL;
    }
}

DivergenceQualifier accept_uniformity_qualifier(ctxparams) {
    DivergenceQualifier divergence = Unknown;
    if (accept_token(ctx, uniform_tok))
        divergence = Uniform;
    else if (accept_token(ctx, varying_tok))
        divergence = Varying;
    return divergence;
}

const Type* accept_maybe_qualified_type(ctxparams) {
    DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    const Type* unqualified = accept_unqualified_type(ctx);
    if (qualifier != Unknown)
        expect(unqualified && "we read a uniformity qualifier and expected a type to follow");
    if (qualifier == Unknown)
        return unqualified;
    else
        return qualified_type(arena, (QualifiedType) { .is_uniform = qualifier == Uniform, .type = unqualified });
}

const Type* accept_qualified_type(ctxparams) {
    DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    if (qualifier == Unknown)
        return NULL;
    const Type* unqualified = accept_unqualified_type(ctx);
    expect(unqualified);
    return qualified_type(arena, (QualifiedType) { .is_uniform = qualifier == Uniform, .type = unqualified });
}

Nodes expect_parameters(ctxparams) {
    expect(accept_token(ctx, lpar_tok));
    struct List* params = new_list(Node*);
    while (true) {
        if (accept_token(ctx, rpar_tok))
            break;

        next: {
            const Type* qtype = accept_qualified_type(ctx);
            expect(qtype);
            const char* id = accept_identifier(ctx);
            expect(id);

            const Node* node = var(arena, qtype, id);

            append_list(Node*, params, node);

            if (accept_token(ctx, comma_tok))
                goto next;
        }
    }

    Nodes variables2 = nodes(arena, params->elements_count, (const Node**) params->alloc);
    destroy_list(params);
    return variables2;
}

Nodes expect_types(ctxparams, enum TokenTag separator, bool expect_qualified) {
    struct List* types = new_list(Type*);
    while (true) {
        const Type* type = expect_qualified ? accept_qualified_type(ctx) : accept_maybe_qualified_type(ctx);
        if (!type)
            break;

        append_list(Type*, types, type);

        if (separator != 0)
            accept_token(ctx, separator);
    }

    Nodes types2 = nodes(arena, types->elements_count, (const Type**) types->alloc);
    destroy_list(types);
    return types2;
}

const Node* accept_literal(ctxparams) {
    struct Token tok = curr_token(tokenizer);
    switch (tok.tag) {
        case dec_lit_tok: {
            next_token(tokenizer);
            size_t size = tok.end - tok.start;
            return untyped_number(arena, (UntypedNumber) {
                .plaintext = string_sized(arena, (int) size, &contents[tok.start])
            });
            //int64_t value = strtol(&contents[tok.start], NULL, 10)
            //return untyped_number(value);
        }

        default: return NULL;
    }
}

const Node* accept_value(ctxparams) {
    const char* id = accept_identifier(ctx);
    if (id) {
        return unbound(arena, (Unbound) {
            .name = id
        });
    }

    return accept_literal(ctx);
}

Strings expect_identifiers(ctxparams) {
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

    Strings final = strings(arena, list->elements_count, (const char**) list->alloc);
    destroy_list(list);
    return final;
}

Nodes expect_values(ctxparams, enum TokenTag separator) {
    struct List* list = new_list( Node*);

    bool expect = false;
    while (true) {
        const Node* val = accept_value(ctx);
        if (!val) {
            if (expect)
                error("expected value but got none")
            else
                break;
        }

        append_list(Node*, list, val);

        if (separator != 0) {
            if (accept_token(ctx, separator))
                expect = true;
            else
                break;
        }
    }

    Nodes final = nodes(arena, list->elements_count, (const Node**) list->alloc);
    destroy_list(list);
    return final;
}

const Node* expect_computation(ctxparams) {
    struct Token tok = curr_token(tokenizer);
    switch (tok.tag) {
        case add_tok: {
            next_token(tokenizer);
            Nodes args = expect_values(ctx, 0);
            expect(accept_token(ctx, semi_tok));
            return primop(arena, (PrimOp) {
                .op = add_op,
                .args = args
            });
        }
        default: error("cannot parse a computation");
    }
}

const Node* accept_instruction(ctxparams) {
    struct Token current_token = curr_token(tokenizer);
    switch (current_token.tag) {
        case return_tok: {
            next_token(tokenizer);
            Nodes values = expect_values(ctx, 0);
            expect(accept_token(ctx, semi_tok));
            return fn_ret(arena, (Return) {
                .values = values
            });
        }
        case let_tok: {
            next_token(tokenizer);
            Strings ids = expect_identifiers(ctx);
            size_t bindings_count = ids.count;
            const Node* bindings[bindings_count];
            for (size_t i = 0; i < bindings_count; i++)
                bindings[i] = var(arena, NULL, ids.strings[i]);

            expect(accept_token(ctx, equal_tok));
            const Node* comp = expect_computation(ctx);
            return let(arena, (Let) {
                .variables = nodes(arena, bindings_count, bindings),
                .target = comp
            });
        }
        default: break;
    }
    return NULL;
}

Nodes expect_block(ctxparams) {
    expect(accept_token(ctx, lbracket_tok));
    struct List* instructions = new_list(Node*);
    while (true) {
        if (accept_token(ctx, rbracket_tok))
            break;

        const Node* instruction = accept_instruction(ctx);
        if (instruction)
            append_list(Node*, instructions, instruction);
        else {
            expect(accept_token(ctx, rbracket_tok));
            break;
        }
    }
    Nodes block = nodes(arena, entries_count_list(instructions), read_list(const Node*, instructions));
    destroy_list(instructions);
    return block;
}

struct TopLevelDecl {
    bool empty;
    const Node* variable;
    const Node* definition;
};

struct TopLevelDecl accept_fn_decl(ctxparams) {
    if (!accept_token(ctx, fn_tok))
        return (struct TopLevelDecl) { .empty = true };

    Nodes types = expect_types(ctx, comma_tok, false);
    const char* id = accept_identifier(ctx);
    expect(id);
    expect(curr_token(tokenizer).tag == lpar_tok);
    Nodes parameters = expect_parameters(ctx);
    Nodes instructions = expect_block(ctx);

    const Node* function = fn(arena, (Function) {
        .params = parameters,
        .return_types = types,
        .instructions = instructions
    });

    const Node* variable = var(arena, derive_fn_type(arena, &function->payload.fn), id);

    return (struct TopLevelDecl) {
        .empty = false,
        .variable = variable,
        .definition = function
    };
}

struct TopLevelDecl accept_var_decl(ctxparams) {
    if (!accept_token(ctx, var_tok))
        return (struct TopLevelDecl) { .empty = true };

    const Type* mqtype = accept_maybe_qualified_type(ctx);
    expect(mqtype);
    const char* id = accept_identifier(ctx);
    expect(id);

    // unspecified global variables default to varying
    if (get_qualifier(mqtype) == Unknown)
        mqtype = qualified_type(arena, (QualifiedType) {
            .is_uniform = false,
            .type = mqtype
        });

    expect(accept_token(ctx, semi_tok));

    const Node* variable = var(arena, mqtype, id);

    return (struct TopLevelDecl) {
        .empty = false,
        .variable = variable,
        .definition = NULL
    };
}

const Node* parse(char* contents, IrArena* arena) {
    struct Tokenizer* tokenizer = new_tokenizer(contents);

    struct List* top_level = new_list(struct TopLevelDecl);

    while (true) {
        struct Token token = curr_token(tokenizer);
        if (token.tag == EOF_tok)
            break;

        struct TopLevelDecl decl = accept_fn_decl(ctx);
        if (decl.empty)
            decl = accept_var_decl(ctx);
        
        if (!decl.empty) {
            expect(decl.variable->payload.var.type != NULL && "top-level declarations require types");

            printf("decl %s parsed :", decl.variable->payload.var.name);
            if (decl.definition)
                print_node(decl.definition);
            printf("\n");

            append_list(struct TopLevelDecl, top_level, decl);
            continue;
        }

        printf("No idea what to parse here... (tok=(tag = %s, pos = %zu))\n", token_tags[token.tag], token.start);
        exit(-3);
    }

    size_t count = top_level->elements_count;

    const Node* variables[count];
    const Node* definitions[count];

    for (size_t i = 0; i < count; i++) {
        variables[i] = read_list(struct TopLevelDecl, top_level)[i].variable;
        definitions[i] = read_list(struct TopLevelDecl, top_level)[i].definition;
    }

    destroy_list(top_level);
    destroy_tokenizer(tokenizer);

    return root(arena, (Root) {
        .variables = nodes(arena, count, variables),
        .definitions = nodes(arena, count, definitions)
    });
}
