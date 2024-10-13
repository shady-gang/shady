#ifndef SHADY_IR_DECL_H
#define SHADY_IR_DECL_H

#include "shady/ir/base.h"
#include "shady/ir/grammar.h"

Node* constant(Module*, Nodes annotations, const Type*, const char* name);
Node* global_var(Module*, Nodes annotations, const Type*, String, AddressSpace);

#endif
