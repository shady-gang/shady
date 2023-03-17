#include "clang_ast.h"

#include <assert.h>
#include <string.h>

#include "log.h"
#include "portability.h"

static struct { const char* str; Op op; } binary_opcodes[] = {
    {"+", add_op},
};

static struct { const char* str; Op op; } unary_opcodes[] = {
    {"*", deref_op},
};

static Op binary_opcode_to_shady(const char* str) {
    for (int i = 0; i < sizeof(binary_opcodes) / sizeof(*binary_opcodes); i++) {
        if (strcmp(binary_opcodes[i].str, str) == 0)
            return binary_opcodes[i].op;
    }
    assert(false);
}

static Op unary_opcode_to_shady(const char* str) {
    for (int i = 0; i < sizeof(unary_opcodes) / sizeof(*unary_opcodes); i++) {
        if (strcmp(unary_opcodes[i].str, str) == 0)
            return unary_opcodes[i].op;
    }
    assert(false);
}

static bool has_inner(json_object* obj) {
    json_object* inner = json_object_object_get(obj, "inner");
    if (!inner)
        return false;
    assert(inner && json_object_get_type(inner) == json_type_array);
    int len = json_object_array_length(inner);
    return len > 0;
}

static json_object* get_first_inner(json_object* obj) {
    json_object* inner = json_object_object_get(obj, "inner");
    assert(inner && json_object_get_type(inner) == json_type_array);
    int len = json_object_array_length(inner);
    assert(len >= 1);
    return json_object_array_get_idx(inner, 0);
}

const Type* get_node_type(ClangAst* ast, json_object* obj, bool value_type) {
    json_object* jtype = json_object_object_get(obj, "type");
    const char* qualified_type = json_object_get_string(json_object_object_get(jtype, "qualType"));
    return convert_qualtype(ast, value_type, qualified_type);
}

static const Node* expr_to_shady(ClangAst* ast, BodyBuilder* bb, json_object* expr) {
    const char* kind = json_object_get_string(json_object_object_get(expr, "kind"));

    if (strcmp(kind, "UnaryOperator") == 0) {
        const char* opcode = json_object_get_string(json_object_object_get(expr, "opcode"));
        return prim_op(ast->arena, (PrimOp) {
                .op = unary_opcode_to_shady(opcode),
                .type_arguments = empty(ast->arena),
                .operands = singleton(expr_to_shady(ast, bb, get_first_inner(expr)))
        });
    }
    else if (strcmp(kind, "BinaryOperator") == 0) {
        const char* opcode = json_object_get_string(json_object_object_get(expr, "opcode"));

        json_object* inner = json_object_object_get(expr, "inner");
        assert(inner && json_object_get_type(inner) == json_type_array);
        int len = json_object_array_length(inner);

        LARRAY(const Node*, operands, len);
        for (size_t i = 0; i < len; i++) {
            operands[i] = expr_to_shady(ast, bb, json_object_array_get_idx(inner, i));
        }

        return prim_op(ast->arena, (PrimOp) {
            .op = binary_opcode_to_shady(opcode),
            .type_arguments = empty(ast->arena),
            .operands = nodes(ast->arena, len, operands)
        });
    } else if (strcmp(kind, "IntegerLiteral") == 0) {
        return untyped_number(ast->arena, (UntypedNumber) {
            .plaintext = json_object_get_string(json_object_object_get(expr, "value"))
        });
    } else if (strcmp(kind, "VarDecl") == 0 || strcmp(kind, "ParmVarDecl") == 0) {
        const char* name = json_object_get_string(json_object_object_get(expr, "name"));
        return unbound(ast->arena, (Unbound) { .name = name });
    } else if (strcmp(kind, "DeclRefExpr") == 0) {
        return expr_to_shady(ast, bb, json_object_object_get(expr, "referencedDecl"));
    } else if (strcmp(kind, "ImplicitCastExpr") == 0) {
        const char* cast_kind = json_object_get_string(json_object_object_get(expr, "castKind"));

        if (strcmp(cast_kind, "LValueToRValue") == 0) {
            return expr_to_shady(ast, bb, get_first_inner(expr));
        } else if (strcmp(cast_kind, "IntegralToFloating") == 0) {
            const Type* dst_type = get_node_type(ast, expr, false);
            return prim_op(ast->arena, (PrimOp) {
                    .op = convert_op,
                    .type_arguments = singleton(dst_type),
                    .operands = singleton(expr_to_shady(ast, bb, get_first_inner(expr)))
            });
        } else {
            assert(false);
        }
    } else {
        assert(false);
    }
}

static void var_decl_to_shady(ClangAst* ast, BodyBuilder* bb, json_object* decl) {
    const char* kind = json_object_get_string(json_object_object_get(decl, "kind"));
    assert(strcmp(kind, "VarDecl") == 0);

    const char* name = json_object_get_string(json_object_object_get(decl, "name"));

    const Type* type = get_node_type(ast, decl, false);

    json_object* inner = json_object_object_get(decl, "inner");
    assert(inner && json_object_get_type(inner) == json_type_array);
    int len = json_object_array_length(inner);
    assert(len >= 0 && len <= 1);

    const Node* init = NULL;
    if (len == 1) {
        init = expr_to_shady(ast, bb, json_object_array_get_idx(inner, 0));
    }

    Nodes types = singleton(type);
    bind_instruction_extra_mutable(bb, init, 1, &types, &name);
}

static const Node* stmt_to_shady(ClangAst* ast, BodyBuilder* bb, json_object* stmt) {
    const char* stmt_kind = json_object_get_string(json_object_object_get(stmt, "kind"));

    if (strcmp(stmt_kind, "NullStmt") == 0) {}
    else if (strcmp(stmt_kind, "CompoundStmt") == 0) {
        json_object* inner = json_object_object_get(stmt, "inner");
        int num_stmts = json_object_array_length(inner);
        for (size_t j = 0; j < num_stmts; j++) {
            json_object* sub_stmt = json_object_array_get_idx(inner, j);
            const Node* rslt = stmt_to_shady(ast, bb, sub_stmt);
            if (j == num_stmts - 1 && rslt)
                return rslt;
        }
    } else if (strcmp(stmt_kind, "ReturnStmt") == 0) {
        Nodes ret_vals;
        if (has_inner(stmt)) {
            ret_vals = singleton(expr_to_shady(ast, bb, get_first_inner(stmt)));
        } else {
            ret_vals = empty(ast->arena);
        }
        return finish_body(bb, fn_ret(ast->arena, (Return) {
                .args = ret_vals
        }));
    } else if (strcmp(stmt_kind, "DeclStmt") == 0) {
        json_object* inner = json_object_object_get(stmt, "inner");
        int num_decls = json_object_array_length(inner);
        for (size_t j = 0; j < num_decls; j++) {
            json_object* variable = json_object_array_get_idx(inner, j);
            var_decl_to_shady(ast, bb, variable);
        }
    } else {
        const Node* expr = expr_to_shady(ast, bb, stmt);
        bind_instruction_extra(bb, expr, 0, NULL, NULL);
    }

    return NULL;
}

static void toplevel_to_shady(ClangAst* ast, json_object* decl) {
    assert(decl);
    const char* name = json_object_get_string(json_object_object_get(decl, "name"));
    const char* kind = json_object_get_string(json_object_object_get(decl, "kind"));
    assert(name && kind);
    printf("decl: %s\n", name);

    if (strcmp(kind, "TypedefDecl") == 0) {}
    else if (strcmp(kind, "FunctionDecl") == 0) {
        const char* mangled_name = json_object_get_string(json_object_object_get(decl, "mangledName"));
        printf("fn: %s\n", mangled_name);
        const char* qualified_type = json_object_get_string(json_object_object_get(json_object_object_get(decl, "type"), "qualType"));
        const Type* fn_type = convert_qualtype(ast, false, qualified_type);
        dump_node(fn_type);

        int params_count = fn_type->payload.fn_type.param_types.count;
        LARRAY(const Node*, params, params_count);
        int param_i = 0;

        const Node* body = NULL;

        json_object* inner = json_object_object_get(decl, "inner");
        int len = json_object_array_length(inner);
        for (size_t i = 0; i < len; i++) {
            json_object* sub_decl = json_object_array_get_idx(inner, i);
            const char* body_kind = json_object_get_string(json_object_object_get(sub_decl, "kind"));
            if (strcmp(body_kind, "ParmVarDecl") == 0) {
                const char* param_qualified_type = json_object_get_string(json_object_object_get(json_object_object_get(sub_decl, "type"), "qualType"));
                const char* param_name = json_object_get_string(json_object_object_get(sub_decl, "name"));
                const Node* type = convert_qualtype(ast, true, param_qualified_type);
                params[param_i++] = var(ast->arena, type, param_name);
            } else if (strcmp(body_kind, "CompoundStmt") == 0) {
                assert(param_i == params_count);
                BodyBuilder* bb = begin_body(ast->mod);
                body = stmt_to_shady(ast, bb, sub_decl);
                if (!body)
                    body = finish_body(bb, unreachable(ast->arena)); // TODO
            }
            else {
                assert(false);
            }
        }

        Node* fn = function(ast->mod, nodes(ast->arena, params_count, params), mangled_name, empty(ast->arena), fn_type->payload.fn_type.return_types);
        fn->payload.fun.body = body;
        dump_node(fn);
    }
    else {
        assert(false);
    }
}

void ast_to_shady(json_object* object, Module* mod) {
    ClangAst converter = {
        .arena = get_module_arena(mod),
        .mod = mod,
    };
    assert(object && json_object_is_type(object, json_type_object));
    json_object* kind = json_object_object_get(object, "kind");
    assert(kind && json_object_get_string(kind) && strcmp(json_object_get_string(kind), "TranslationUnitDecl") == 0);
    debug_print("Parsed root json object successfully\n");

    json_object* inner = json_object_object_get(object, "inner");
    assert(inner && json_object_get_type(inner) == json_type_array);
    int len = json_object_array_length(inner);
    for (size_t i = 0; i < len; i++) {
        toplevel_to_shady(&converter, json_object_array_get_idx(inner, i));
    }
}
