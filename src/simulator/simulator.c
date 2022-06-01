#include "simulator.h"
#include "../log.h"
#include "../portability.h"

#include <assert.h>

void sim_instruction(Simulator* sim, const Node* instruction);
void sim_terminator(Simulator* sim, const Node* terminator);
void sim_block(Simulator* sim, const Node* block);

void sim_primop(Simulator* sim, PrimOp prim_op) {
    switch (prim_op.op) {
        case add_op:break;
        case sub_op:break;
        case mul_op:break;
        case div_op:break;
        case mod_op:break;
        case gt_op:break;
        case gte_op:break;
        case lt_op:break;
        case lte_op:break;
        case eq_op:break;
        case neq_op:break;
        case and_op:break;
        case or_op:break;
        case xor_op:break;
        case not_op:break;
        case alloca_op:break;
        case load_op:break;
        case store_op:break;
        case lea_op:break;
        case select_op:break;
        case cast_ptr_op:break;
        case cast_prim_op:break;
        case push_stack_op:break;
        case pop_stack_op:break;
        case push_stack_uniform_op:break;
        case pop_stack_uniform_op:break;
    }
}

void sim_terminator(Simulator* sim, const Node* terminator) {
    switch (terminator->tag) {
        case Merge_TAG: {
            assert(entries_count_list(sim->structured_constructs) > 0);
        }
        default: error("Unhandled terminator");
    }
}

void sim_instruction(Simulator* sim, const Node* instruction) {
    const Node* real_instruction = instruction;
    if (real_instruction->tag == Let_TAG)
        real_instruction = real_instruction->payload.let.instruction;

    switch (real_instruction->tag) {
        case PrimOp_TAG: sim_primop(sim, real_instruction->payload.prim_op); break;
        case If_TAG:
        default: error("Not known instruction");
    }
}

int main(int argc, char** argv) {

}