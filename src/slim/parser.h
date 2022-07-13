#ifndef SHADY_PARSER_H

#define SHADY_PARSER_H

#include "shady/ir.h"

typedef struct {
    bool front_end;
} ParserConfig;

const Node* parse(ParserConfig config, char* contents, IrArena* arena);

#endif