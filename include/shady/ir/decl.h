#ifndef SHADY_IR_DECL_H
#define SHADY_IR_DECL_H

#include "shady/ir/base.h"
#include "shady/ir/grammar.h"
#include "shady/ir/builtin.h"

Node* _shd_constant(Module*, Nodes annotations, const Type*, const char* name);
Node* _shd_global_var(Module*, Nodes annotations, const Type*, String, AddressSpace, bool);

static inline Node* constant_helper(Module* m, Nodes annotations, const Type* t, const char* name) { return _shd_constant(m, annotations, t, name); }
static inline Node* global_variable_helper(Module* m, Nodes annotations, const Type* t, String name, AddressSpace as, bool is_ref) { return _shd_global_var(m, annotations, t, name, as, is_ref); }

typedef struct Rewriter_ Rewriter;
const Node* shd_find_or_process_decl(Rewriter* rewriter, const char* name);

#endif
