#ifndef SHADY_USAGES
#define SHADY_USAGES

#include "shady/ir.h"
#include "cfg.h"
#include "list.h"
#include "dict.h"
#include "arena.h"

typedef struct UsesMap_ UsesMap;

const UsesMap* shd_new_uses_map_fn(const Node* root, NodeClass exclude);
const UsesMap* shd_new_uses_map_module(const Module* m, NodeClass exclude);
void shd_destroy_uses_map(const UsesMap* map);

typedef struct Use_ Use;
struct Use_ {
    const Node* user;
    NodeClass operand_class;
    String operand_name;
    size_t operand_index;
    const Use* next_use;
};

const Use* shd_get_first_use(const UsesMap* map, const Node* n);

#endif
