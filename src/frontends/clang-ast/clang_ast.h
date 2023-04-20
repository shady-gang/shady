#ifndef CLANG_AST_PARSER
#define CLANG_AST_PARSER

#include "shady/ir.h"

typedef struct ClangAst_ ClangAst;

struct ClangAst_ {
    IrArena* arena;
    Module* mod;
};

const Type* convert_qualtype(ClangAst*, bool, const char*);

void parse_c_file(const char* filename, Module* mod);

#endif
