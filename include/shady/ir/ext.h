#ifndef SHADY_IR_EXT_H
#define SHADY_IR_EXT_H

#include "shady/ir/base.h"
#include "shady/ir/builder.h"

const Node* shd_bld_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands);

#endif
