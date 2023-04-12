#ifndef SHADY_USAGES
#define SHADY_USAGES

#include "shady/ir.h"
#include "scope.h"
#include "list.h"
#include "dict.h"
#include "arena.h"

typedef struct {
    CFNode* defined;
    size_t uses_count;
    bool escapes_defining_block;
    // Join tokens care about this
    bool in_non_callee_position;
    bool sealed;
} Uses;

typedef struct {
    Scope* scope;
    Arena* arena;
    struct Dict* map;
} ScopeUses;

ScopeUses* analyse_uses_scope(Scope*);
void destroy_uses_scope(ScopeUses*);

#endif
