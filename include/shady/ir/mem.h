#ifndef SHADY_IR_MEM_H
#define SHADY_IR_MEM_H

#include "shady/ir/base.h"

const Node* get_parent_mem(const Node* mem);
const Node* get_original_mem(const Node* mem);

#endif
