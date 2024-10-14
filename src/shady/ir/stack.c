#include "shady/ir/stack.h"

#include "shady/ir/grammar.h"

void shd_bld_stack_push_value(BodyBuilder* bb, const Node* value) {
    shd_bld_add_instruction_extract(bb, push_stack(shd_get_bb_arena(bb), (PushStack) { .value = value, .mem = shd_bb_mem(bb) }));
}

void shd_bld_stack_push_values(BodyBuilder* bb, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        shd_bld_stack_push_value(bb, value);
    }
}

const Node* shd_bld_stack_pop_value(BodyBuilder* bb, const Type* type) {
    const Node* instruction = pop_stack(shd_get_bb_arena(bb), (PopStack) { .type = type, .mem = shd_bb_mem(bb) });
    return shd_first(shd_bld_add_instruction_extract(bb, instruction));
}

const Node* shd_bld_get_stack_base_addr(BodyBuilder* bb) {
    return get_stack_base_addr(shd_get_bb_arena(bb), (GetStackBaseAddr) { .mem = shd_bb_mem(bb) });
}

const Node* shd_bld_get_stack_size(BodyBuilder* bb) {
    return shd_first(shd_bld_add_instruction_extract(bb, get_stack_size(shd_get_bb_arena(bb), (GetStackSize) { .mem = shd_bb_mem(bb) })));
}

void shd_bld_set_stack_size(BodyBuilder* bb, const Node* new_size) {
    shd_bld_add_instruction_extract(bb, set_stack_size(shd_get_bb_arena(bb), (SetStackSize) { .value = new_size, .mem = shd_bb_mem(bb) }));
}
