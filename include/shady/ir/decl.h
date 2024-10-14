#ifndef SHADY_IR_DECL_H
#define SHADY_IR_DECL_H

#include "shady/ir/base.h"
#include "shady/ir/grammar.h"

Node* _shd_constant(Module*, Nodes annotations, const Type*, const char* name);
Node* _shd_global_var(Module*, Nodes annotations, const Type*, String, AddressSpace);

static inline Node* constant(Module* m, Nodes annotations, const Type* t, const char* name) { return _shd_constant(m, annotations, t, name); }
static inline Node* global_var(Module* m, Nodes annotations, const Type* t, String name, AddressSpace as) { return _shd_global_var(m, annotations, t, name, as); }

typedef struct Rewriter_ Rewriter;
const Node* shd_find_or_process_decl(Rewriter* rewriter, const char* name);

#endif
