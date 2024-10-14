#include "shady/ir/ext.h"
#include "shady/ir/grammar.h"

const Node* gen_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands) {
    return shd_bld_add_instruction(bb, ext_instr(shd_get_bb_arena(bb), (ExtInstr) {
        .mem = shd_bb_mem(bb),
        .set = set,
        .opcode = opcode,
        .result_t = return_t,
        .operands = operands,
    }));
}
