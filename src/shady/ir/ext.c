#include "shady/ir/ext.h"

#include "log.h"
#include "shady/analysis/literal.h"
#include "shady/ir/grammar.h"
#include "shady/ir/int.h"

const Node* shd_make_ext_spv_op(IrArena* a, String set, int opcode, bool has_result, const Type* result_t, size_t argc) {
    Nodes pattern = shd_empty(a);
    for (size_t i = 0; i < argc; i++) {
        pattern = shd_nodes_append(a, pattern, NULL);
    }
    return ext_op_def_helper(a, set, opcode, has_result, result_t, pattern);
}

const Node* shd_bld_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands) {
    IrArena* a = shd_get_bb_arena(bb);
    const Node* ext_op = shd_make_ext_spv_op(a, set, opcode, true, return_t, operands.count);

    return shd_bld_add_instruction(bb, ext_instr(a, (ExtInstr) {
        .mem = shd_bld_mem(bb),
        .def = ext_op,
        .arguments = operands,
    }));
}

bool shd_is_ext_instruction(const Node* instr, String set, int opcode) {
    if (!is_ext_op(instr))
        return false;
    ExtOpDef def = get_ext_op_def(instr)->payload.ext_op_def;
    return strcmp(def.set, set) == 0 && def.opcode == opcode;
}

size_t shd_ext_instruction_num_operands(const Node* instr) {
    ExtOpDef def = get_ext_op_def(instr)->payload.ext_op_def;
    Nodes arguments = get_ext_op_arguments(instr);
    size_t used_args = 0;
    for (size_t i = 0; i < def.ops_pattern.count; i++) {
        const Node* pattern_item = def.ops_pattern.nodes[i];
        if (!pattern_item)
            used_args++;
    }
    assert(used_args <= arguments.count);
    return def.ops_pattern.count + arguments.count - used_args;
}

ShdExtOperandKind shd_ext_instruction_get_operand_kind(const Node* instr, size_t i) {
    ExtOpDef def = get_ext_op_def(instr)->payload.ext_op_def;
    if (i < def.ops_pattern.count) {
        const Node* pattern_item = def.ops_pattern.nodes[i];
        if (!pattern_item)
            return ShdExtOperandNode;
        if (pattern_item->tag == IntLiteral_TAG)
            return ShdExtOperandLiteral32;
        if (pattern_item->tag == StringLiteral_TAG)
            return ShdExtOperandLiteralString;
        shd_error("Invalid pattern");
    }
    return ShdExtOperandNode;
}

const Node* shd_ext_instruction_get_node_operand(const Node* instr, size_t j) {
    ExtOpDef def = get_ext_op_def(instr)->payload.ext_op_def;
    Nodes arguments = get_ext_op_arguments(instr);
    size_t used_args = 0;
    for (size_t i = 0; i < def.ops_pattern.count; i++) {
        const Node* pattern_item = def.ops_pattern.nodes[i];
        if (!pattern_item) {
            if (i == j)
                return arguments.nodes[used_args];
            used_args++;
        }
    }
    return arguments.nodes[used_args + j - def.ops_pattern.count];
}

uint32_t shd_ext_instruction_get_u32_operand(const Node* instr, size_t i) {
    ExtOpDef def = get_ext_op_def(instr)->payload.ext_op_def;
    if (i < def.ops_pattern.count) {
        const Node* pattern_item = def.ops_pattern.nodes[i];
        if (pattern_item->tag == IntLiteral_TAG)
            return shd_get_int_value(pattern_item, false);
    }
    shd_error("Invalid operand");
}

String shd_ext_instruction_get_string_operand(const Node* instr, size_t i) {
    ExtOpDef def = get_ext_op_def(instr)->payload.ext_op_def;
    if (i < def.ops_pattern.count) {
        const Node* pattern_item = def.ops_pattern.nodes[i];
        if (pattern_item->tag == StringLiteral_TAG)
            return shd_get_string_literal(instr->arena, pattern_item);
    }
    shd_error("Invalid operand");
}
