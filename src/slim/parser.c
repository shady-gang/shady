#include "token.h"

#include "../containers/list.h"

#include "../log.h"
#include "../type.h"
#include "../local_array.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

extern const char* token_tags[];

// to avoid some repetition
#define ctxparams char* contents, IrArena* arena, struct Tokenizer* tokenizer
#define ctx contents, arena, tokenizer

#define expect(condition) expect_impl(condition, #condition)
static void expect_impl(bool condition, const char* err) {
    if (!condition) {
        error_print("expected to parse: %s\n", err);
        exit(-4);
    }
}

static bool accept_token(ctxparams, enum TokenTag tag) {
    if (curr_token(tokenizer).tag == tag) {
        next_token(tokenizer);
        return true;
    }
    return false;
}

static const char* accept_identifier(ctxparams) {
    struct Token tok = curr_token(tokenizer);
    if (tok.tag == identifier_tok) {
        next_token(tokenizer);
        size_t size = tok.end - tok.start;
        return string_sized(arena, (int) size, &contents[tok.start]);
    }
    return NULL;
}

static const Node* accept_function(ctxparams, String);
static const Node* expect_block(ctxparams, bool);

static const Type* accept_unqualified_type(ctxparams) {
    if (accept_token(ctx, int_tok)) {
        return int_type(arena);
    } else if (accept_token(ctx, float_tok)) {
        return float_type(arena);
    } else if (accept_token(ctx, bool__tok)) {
        return bool_type(arena);
    } else if (accept_token(ctx, ptr_tok)) {
        SHADY_NOT_IMPLEM
    } else if (accept_token(ctx, fn_tok)) {
        SHADY_NOT_IMPLEM
    } else {
        return NULL;
    }
}

static DivergenceQualifier accept_uniformity_qualifier(ctxparams) {
    DivergenceQualifier divergence = Unknown;
    if (accept_token(ctx, uniform_tok))
        divergence = Uniform;
    else if (accept_token(ctx, varying_tok))
        divergence = Varying;
    return divergence;
}

static const Type* accept_maybe_qualified_type(ctxparams) {
    DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    const Type* unqualified = accept_unqualified_type(ctx);
    if (qualifier != Unknown)
        expect(unqualified && "we read a uniformity qualifier and expected a type to follow");
    if (qualifier == Unknown)
        return unqualified;
    else
        return qualified_type(arena, (QualifiedType) { .is_uniform = qualifier == Uniform, .type = unqualified });
}

static const Type* accept_qualified_type(ctxparams) {
    DivergenceQualifier qualifier = accept_uniformity_qualifier(ctx);
    if (qualifier == Unknown)
        return NULL;
    const Type* unqualified = accept_unqualified_type(ctx);
    expect(unqualified);
    return qualified_type(arena, (QualifiedType) { .is_uniform = qualifier == Uniform, .type = unqualified });
}

static Nodes expect_parameters(ctxparams) {
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

static Nodes accept_types(ctxparams, enum TokenTag separator, bool expect_qualified) {
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

static const Node* accept_literal(ctxparams) {
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
        case true__tok: next_token(tokenizer); return true_lit(arena);
        case false__tok: next_token(tokenizer); return false_lit(arena);
        default: return NULL;
    }
}

static const Node* accept_value(ctxparams) {
    const char* id = accept_identifier(ctx);
    if (id) {
        return unbound(arena, (Unbound) {
            .name = id
        });
    }

    const Node* lit = accept_literal(ctx);
    if (lit) return lit;

    return NULL;
}

static Strings expect_identifiers(ctxparams) {
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

static Nodes expect_values(ctxparams, enum TokenTag separator) {
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

static Nodes expect_computation(ctxparams, Op* op) {
    struct Token tok = curr_token(tokenizer);
    switch (tok.tag) {
        case add_tok:    *op = add_op; break;
        case sub_tok:    *op = sub_op; break;
        case call_tok:   *op = call_op; break;
        default: error("cannot parse a computation");
    }

    next_token(tokenizer);
    Nodes args = expect_values(ctx, 0);
    return args;
}

static const Node* accept_instruction(ctxparams) {
    struct Token current_token = curr_token(tokenizer);
    switch (current_token.tag) {
        case let_tok: {
            next_token(tokenizer);
            Strings ids = expect_identifiers(ctx);
            size_t bindings_count = ids.count;
            LARRAY(const Node*, bindings, bindings_count);
            for (size_t i = 0; i < bindings_count; i++)
                bindings[i] = var(arena, NULL, ids.strings[i]);

            expect(accept_token(ctx, equal_tok));
            Op op;
            Nodes args = expect_computation(ctx, &op);
            expect(accept_token(ctx, semi_tok));
            return let(arena, (Let) {
                .variables = nodes(arena, bindings_count, bindings),
                .op = op,
                .args = args
            });
        }
        case if_tok: {
            next_token(tokenizer);
            const Node* condition = accept_value(ctx);
            expect(condition);
            const Node* if_true = expect_block(ctx, true);
            bool has_else = accept_token(ctx, else_tok);
            // default to an empty block
            const Node* if_false = NULL;
            if (has_else) {
                if_false = expect_block(ctx, true);
            }
            return if_instr(arena, (IfInstr) {
                .condition = condition,
                .if_true = if_true,
                .if_false = if_false
            });
        }
        default: break;
    }
    return NULL;
}

static const Node* accept_terminator(ctxparams) {
    struct Token current_token = curr_token(tokenizer);
    switch (current_token.tag) {
        case jump_tok: {
            next_token(tokenizer);
            const Node* target = accept_value(ctx);
            expect(target);
            Nodes args = expect_values(ctx, 0);
            return jump(arena, (Jump) {
                .target = target,
                .args = args
            });
        }
        case return_tok: {
            next_token(tokenizer);
            Nodes values = expect_values(ctx, 0);
            return fn_ret(arena, (Return) {
                .fn = NULL,
                .values = values
            });
        }
        case unreachable_tok: {
            next_token(tokenizer);
            return unreachable(arena);
        }
        default: break;
    }
    return NULL;
}

static const Node* expect_block(ctxparams, bool implicit_join) {
    expect(accept_token(ctx, lbracket_tok));
    struct List* instructions = new_list(Node*);

    Nodes continuations = nodes(arena, 0, NULL);
    Nodes continuations_names = nodes(arena, 0, NULL);

    while (true) {
        const Node* instruction = accept_instruction(ctx);
        if (!instruction) break;
        append_list(Node*, instructions, instruction);
    }

    Nodes instrs = nodes(arena, entries_count_list(instructions), read_list(const Node*, instructions));
    destroy_list(instructions);

    const Node* terminator = accept_terminator(ctx);
    expect(accept_token(ctx, semi_tok));
    if (!terminator) {
        if (implicit_join)
            terminator = join(arena);
        else
            error("expected terminator: return, jump, branch ...");
    }

    if (curr_token(tokenizer).tag == identifier_tok) {
        struct List* conts = new_list(Node*);
        struct List* names = new_list(Node*);
        while (true) {
            const char* identifier = accept_identifier(ctx);
            if (!identifier)
                break;
            expect(accept_token(ctx, colon_tok));

            Nodes parameters = expect_parameters(ctx);
            const Node* block = expect_block(ctx, false);

            Node* continuation = fn(arena, true, identifier, parameters, nodes(arena, 0, NULL));
            continuation->payload.fn.block= block;
            const Node* contvar = var(arena, qualified_type(arena, (QualifiedType) {
                .type = derive_fn_type(arena, &continuation->payload.fn),
                .is_uniform = true
            }), identifier);
            append_list(Node*, conts, continuation);
            append_list(Node*, names, contvar);
        }

        continuations = nodes(arena, entries_count_list(conts), read_list(const Node*, conts));
        continuations_names = nodes(arena, entries_count_list(names), read_list(const Node*, names));
        destroy_list(conts);
        destroy_list(names);
    }

    expect(accept_token(ctx, rbracket_tok));

    return parsed_block(arena, (ParsedBlock) {
        .instructions = instrs,
        .continuations = continuations,
        .continuations_vars = continuations_names,
        .terminator = terminator,
    });
}

static const Node* accept_function(ctxparams, String id) {
    if (!accept_token(ctx, fn_tok))
        return NULL;

    Nodes types = accept_types(ctx, comma_tok, false);
    expect(curr_token(tokenizer).tag == lpar_tok);
    Nodes parameters = expect_parameters(ctx);
    const Node* block = expect_block(ctx, false);

    Node* function = fn(arena, false, id, parameters, types);
    function->payload.fn.block = block;

    return function;
}

struct TopLevelDecl {
    bool empty;
    const Node* variable;
    const Node* definition;
};

static struct TopLevelDecl accept_def(ctxparams) {
    if (!accept_token(ctx, def_tok))
        return (struct TopLevelDecl) { .empty = true };

    const Type* type = accept_unqualified_type(ctx);
    const char* id = accept_identifier(ctx);
    expect(id);
    expect(accept_token(ctx, equal_tok));
    const Node* definition = accept_value(ctx);
    if (!definition)
        definition = accept_function(ctx, id);
    assert(definition);

    expect(accept_token(ctx, semi_tok));

    const Node* variable = var(arena, type, id);

    return (struct TopLevelDecl) {
        .empty = false,
        .variable = variable,
        .definition = definition
    };
}

static struct TopLevelDecl accept_var_decl(ctxparams) {
    AddressSpace as;
    if (accept_token(ctx, private_tok))
        as = AsPrivate;
    else if (accept_token(ctx, shared_tok))
        as = AsShared;
    // the global address space cannot be used here
    //else if (accept_token(ctx, global_tok))
    //    as = AsGlobal;
    else if (accept_token(ctx, extern_tok))
        as = AsExternal;
    else if (accept_token(ctx, input_tok))
        as = AsInput;
    else if (accept_token(ctx, output_tok))
        as = AsOutput;
    else
        return (struct TopLevelDecl) { .empty = true };

    const Type* type = accept_unqualified_type(ctx);
    expect(type);

    type = ptr_type(arena, (PtrType) {
        .pointed_type = type,
        .address_space = as
    });

    // global variables are uniform (remember our global variables are _pointers_ to the data manipulated)
    type = qualified_type(arena, (QualifiedType) {
        .type = type,
        .is_uniform = true
    });

    const char* id = accept_identifier(ctx);
    expect(id);

    expect(accept_token(ctx, semi_tok));

    const Node* variable = var(arena, type, id);

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

        struct TopLevelDecl decl = accept_def(ctx);
        if (decl.empty)
            decl = accept_var_decl(ctx);
        
        if (!decl.empty) {
            // expect(decl.variable->payload.var.type != NULL && "top-level declarations require types");

            debug_print("decl %s parsed :", decl.variable->payload.var.name);
            if (decl.definition)
                debug_node(decl.definition);
            debug_print("\n");

            append_list(struct TopLevelDecl, top_level, decl);
            continue;
        }

        error_print("No idea what to parse here... (tok=(tag = %s, pos = %zu))\n", token_tags[token.tag], token.start);
        exit(-3);
    }

    size_t count = top_level->elements_count;

    LARRAY(const Node*, variables, count);
    LARRAY(const Node*, definitions, count);

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
