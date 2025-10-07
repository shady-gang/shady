#include "shady/ir/ext.h"
#include "shady/ir/grammar.h"

const Node* shd_make_ext_spv_op(IrArena* a, String set, int opcode, bool has_result, const Type* result_t, size_t argc) {
    Nodes pattern = shd_empty(a);
    for (size_t i = 0; i < argc; i++) {
        pattern = shd_nodes_append(a, pattern, NULL);
    }
    return ext_spv_op_helper(a, set, opcode, has_result, result_t, pattern);
}

const Node* shd_bld_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands) {
    IrArena* a = shd_get_bb_arena(bb);
    const Node* ext_op = shd_make_ext_spv_op(a, set, opcode, true, return_t, operands.count);

    return shd_bld_add_instruction(bb, ext_instr(a, (ExtInstr) {
        .mem = shd_bld_mem(bb),
        .op = ext_op,
        .arguments = operands,
    }));
}
