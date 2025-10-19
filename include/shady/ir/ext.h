#ifndef SHADY_IR_EXT_H
#define SHADY_IR_EXT_H

#include "shady/ir/base.h"
#include "shady/ir/builder.h"

const Node* shd_make_ext_spv_op(IrArena* a, String set, int opcode, bool has_result, const Type* result_t, size_t argc);

const Node* shd_bld_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands);

bool shd_is_ext_instruction(const Node* instr, String set, int opcode);

static inline bool shd_is_ext_core_instruction(const Node* instr, int opcode) { return shd_is_ext_instruction(instr, "spirv.core", opcode); }

/// External operations have 3 kinds of operands, that map to SPIR-V id references, u32 literals (including all enums) and string literals
typedef enum {
    ShdExtOperandNode,
    ShdExtOperandLiteral32,
    ShdExtOperandLiteralString,
} ShdExtOperandKind;

/// the header word, result ids and result types are not operands, operand zero is the one after them
/// ( opcode, len ), [[result type, ] result id, ] op0, op1, op2, ...
size_t shd_ext_instruction_num_operands(const Node* instr);
ShdExtOperandKind shd_ext_instruction_get_operand_kind(const Node* instr, size_t i);
const Node* shd_ext_instruction_get_node_operand(const Node* instr, size_t i);
uint32_t shd_ext_instruction_get_u32_operand(const Node* instr, size_t i);
String shd_ext_instruction_get_string_operand(const Node* instr, size_t i);

#endif
