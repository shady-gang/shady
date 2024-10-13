#ifndef SHADY_USAGES
#define SHADY_USAGES

#include "shady/ir.h"
#include "cfg.h"
#include "list.h"
#include "dict.h"
#include "arena.h"

typedef struct UsesMap_ UsesMap;

const UsesMap* create_fn_uses_map(const Node* root, NodeClass exclude);
const UsesMap* create_module_uses_map(const Module* m, NodeClass exclude);
void destroy_uses_map(const UsesMap*);

typedef struct Use_ Use;
struct Use_ {
    const Node* user;
    NodeClass operand_class;
    String operand_name;
    size_t operand_index;
    const Use* next_use;
};

const Use* get_first_use(const UsesMap*, const Node*);

#endif
