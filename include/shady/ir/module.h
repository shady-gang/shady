#ifndef SHADY_IR_MODULE_H
#define SHADY_IR_MODULE_H

#include "shady/ir/base.h"

typedef struct Module_ Module;

Module* new_module(IrArena*, String name);

IrArena* get_module_arena(const Module*);
String get_module_name(const Module*);
Nodes get_module_declarations(const Module*);
Node* get_declaration(const Module*, String);

void link_module(Module* dst, Module* src);

#endif
