#ifndef SHADY_IR_MEM_H
#define SHADY_IR_MEM_H

#include "shady/ir/base.h"

const Node* shd_get_parent_mem(const Node* mem);
const Node* shd_get_original_mem(const Node* mem);

const Node* shd_bld_stack_alloc(BodyBuilder* bb, const Type* type);
const Node* shd_bld_local_alloc(BodyBuilder* bb, const Type* type);

const Node* shd_bld_load(BodyBuilder* bb, const Node* ptr);
void shd_bld_store(BodyBuilder* bb, const Node* ptr, const Node* value);

#endif
