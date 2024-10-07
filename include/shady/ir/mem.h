#ifndef SHADY_IR_MEM_H
#define SHADY_IR_MEM_H

#include "shady/ir/base.h"

const Node* shd_get_parent_mem(const Node* mem);
const Node* shd_get_original_mem(const Node* mem);

#endif
