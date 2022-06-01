#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "shady/ir.h"
#include "list.h"

typedef struct {
    Node* function;
    size_t next_instruction;
} ProgramPosition;

typedef struct {
    Node* program;
    ProgramPosition position;
    struct List* structured_constructs;
} Simulator;

#endif SIMULATOR_H
