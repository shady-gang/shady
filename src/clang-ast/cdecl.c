#include "clang_ast.h"

#include <assert.h>
#include <string.h>
#include <stddef.h>

#include "list.h"

static void skip_whitespace(size_t* const plen, const char** const ptype) {
    while (*plen > 0 && **ptype == ' ') {
        (*plen)--;
        *ptype = &(*ptype)[1];
    }
}

static bool accept_token(size_t* const plen, const char** const ptype, const char* token) {
    skip_whitespace(plen, ptype);
    size_t tokenlen = strlen(token);
    if (tokenlen > *plen)
        return false;
    if (memcmp(*ptype, token, tokenlen) == 0) {
        *ptype = &(*ptype)[tokenlen];
        *plen -= tokenlen;
        return true;
    }
    return false;
}

static const Type* accept_primtype(ClangAst* ast, size_t* plen, const char** ptype) {
    accept_token(plen, ptype, "const");
    if (accept_token(plen, ptype, "int"))
        return int32_type(ast->arena);
    return NULL;
}

static const Type* eat_qualtype(ClangAst* ast, size_t* plen, const char** ptype) {
    const Type* acc = accept_primtype(ast, plen, ptype);
    while (acc) {
        if (accept_token(plen, ptype, "(")) {
            bool first = true;
            struct List* params = new_list(const Type*);
            while (true) {
                if (accept_token(plen, ptype, ")"))
                    break;
                if (!first) {
                    bool b = accept_token(plen, ptype, ",");
                    assert(b);
                }
                first = false;
                const Type* param = eat_qualtype(ast, plen, ptype);
                assert(param);
                append_list(const Type*, params, param);
            }
            acc = fn_type(ast->arena, (FnType) {
                .return_types = singleton(acc),
                .param_types = nodes(ast->arena, entries_count_list(params), read_list(const Type*, params))
            });
            destroy_list(params);
            continue;
        } else if (accept_token(plen, ptype, "*")) {
            acc = ptr_type(ast->arena, (PtrType) {
                .address_space = AsGeneric,
                .pointed_type = acc,
            });
            continue;
        }
        break;
    }
    return acc;
}

const Type* convert_qualtype(ClangAst* ast, const char* type) {
    assert(type);
    size_t len = strlen(type);
    const Type* eaten = eat_qualtype(ast, &len, &type);
    skip_whitespace(&len, &type);
    assert(len == 0 && *type == '\0' && eaten);
    return eaten;
}
