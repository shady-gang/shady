#include "ir_gen_helpers.h"

#include "shady/rewrite.h"
#include "shady/ir/memory_layout.h"

#include "../ir_private.h"

#include "list.h"
#include "portability.h"
#include "log.h"
#include "util.h"

#include <string.h>
#include <assert.h>

Nodes gen_call(BodyBuilder* bb, const Node* callee, Nodes args) {
    assert(shd_get_arena_config(shd_get_bb_arena(bb))->check_types);
    const Node* instruction = call(shd_get_bb_arena(bb), (Call) { .callee = callee, .args = args, .mem = shd_bb_mem(bb) });
    return shd_bld_add_instruction_extract(bb, instruction);
}

const Node* gen_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands) {
    return shd_bld_add_instruction(bb, ext_instr(shd_get_bb_arena(bb), (ExtInstr) {
        .mem = shd_bb_mem(bb),
        .set = set,
        .opcode = opcode,
        .result_t = return_t,
        .operands = operands,
    }));
}
