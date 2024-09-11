#include "emit_spv.h"

#include "../shady/type.h"
#include "../shady/analysis/cfg.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"

#include <assert.h>

BBBuilder spv_find_basic_block_builder(Emitter* emitter, const Node* bb) {
    BBBuilder* found = find_value_dict(const Node*, BBBuilder, emitter->bb_builders, bb);
    assert(found);
    return *found;
}

static void add_phis(Emitter* emitter, FnBuilder* fn_builder, SpvId src, BBBuilder dst_builder, Nodes args) {
    struct List* phis = spbv_get_phis(dst_builder);
    assert(entries_count_list(phis) == args.count);
    for (size_t i = 0; i < args.count; i++) {
        SpvbPhi* phi = read_list(SpvbPhi*, phis)[i];
        spvb_add_phi_source(phi, src, spv_emit_value(emitter, fn_builder, args.nodes[i]));
    }
}

static void add_branch_phis(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, const Node* dst, Nodes args) {
    // because it's forbidden to jump back into the entry block of a function
    // (which is actually a Function in this IR, not a BasicBlock)
    // we assert that the destination must be an actual BasicBlock
    assert(is_basic_block(dst));
    add_phis(emitter, fn_builder, get_block_builder_id(bb_builder), spv_find_basic_block_builder(emitter, dst), args);
}

static void add_branch_phis_from_jump(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, Jump jump) {
    return add_branch_phis(emitter, fn_builder, bb_builder, jump.target, jump.args);
}

static void emit_if(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, If if_instr) {
    spv_emit_mem(emitter, fn_builder, if_instr.mem);
    SpvId join_bb_id = spv_find_emitted(emitter, fn_builder, if_instr.tail);

    SpvId true_id = spv_find_emitted(emitter, fn_builder, if_instr.if_true);
    SpvId false_id = if_instr.if_false ? spv_find_emitted(emitter, fn_builder, if_instr.if_false) : join_bb_id;

    spvb_selection_merge(bb_builder, join_bb_id, 0);
    SpvId condition = spv_emit_value(emitter, fn_builder, if_instr.condition);
    spvb_branch_conditional(bb_builder, condition, true_id, false_id);
}

static void emit_match(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, Match match) {
    spv_emit_mem(emitter, fn_builder, match.mem);
    SpvId join_bb_id = spv_find_emitted(emitter, fn_builder, match.tail);

    assert(get_unqualified_type(match.inspect->type)->tag == Int_TAG);
    SpvId inspectee = spv_emit_value(emitter, fn_builder, match.inspect);

    SpvId default_id = spv_find_emitted(emitter, fn_builder, match.default_case);

    const Type* inspectee_t = match.inspect->type;
    deconstruct_qualified_type(&inspectee_t);
    assert(inspectee_t->tag == Int_TAG);
    size_t literal_width = inspectee_t->payload.int_type.width == IntTy64 ? 2 : 1;
    size_t literal_case_entry_size = literal_width + 1;
    LARRAY(uint32_t, literals_and_cases, match.cases.count * literal_case_entry_size);
    for (size_t i = 0; i < match.cases.count; i++) {
        uint64_t value = (uint64_t) get_int_literal_value(*resolve_to_int_literal(match.literals.nodes[i]), false);
        if (inspectee_t->payload.int_type.width == IntTy64) {
            literals_and_cases[i * literal_case_entry_size + 0] = (SpvId) (uint32_t) (value & 0xFFFFFFFF);
            literals_and_cases[i * literal_case_entry_size + 1] = (SpvId) (uint32_t) (value >> 32);
        } else {
            literals_and_cases[i * literal_case_entry_size + 0] = (SpvId) (uint32_t) value;
        }
        literals_and_cases[i * literal_case_entry_size + literal_width] = spv_find_emitted(emitter, fn_builder, match.cases.nodes[i]);
    }

    spvb_selection_merge(bb_builder, join_bb_id, 0);
    spvb_switch(bb_builder, inspectee, default_id, match.cases.count * literal_case_entry_size, literals_and_cases);
}

static void emit_loop(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, const Node* abs, Loop loop_instr) {
    spv_emit_mem(emitter, fn_builder, loop_instr.mem);
    SpvId body_id = spv_find_emitted(emitter, fn_builder, loop_instr.body);

    SpvId continue_id = spvb_fresh_id(emitter->file_builder);
    BBBuilder continue_builder = spvb_begin_bb(fn_builder->base, continue_id);
    spvb_name(emitter->file_builder, continue_id, "loop_continue");

    SpvId header_id = spvb_fresh_id(emitter->file_builder);
    BBBuilder header_builder = spvb_begin_bb(fn_builder->base, header_id);
    spvb_name(emitter->file_builder, header_id, "loop_header");

    Nodes body_params = get_abstraction_params(loop_instr.body);
    LARRAY(SpvbPhi*, loop_continue_phis, body_params.count);
    for (size_t i = 0; i < body_params.count; i++) {
        SpvId loop_param_type = spv_emit_type(emitter, get_unqualified_type(body_params.nodes[i]->type));

        SpvId continue_phi_id = spvb_fresh_id(emitter->file_builder);
        SpvbPhi* continue_phi = spvb_add_phi(continue_builder, loop_param_type, continue_phi_id);
        loop_continue_phis[i] = continue_phi;

        // To get the actual loop parameter, we make a second phi for the nodes that go into the header
        // We already know the two edges into the header so we immediately add the Phi sources for it.
        SpvId header_phi_id = spvb_fresh_id(emitter->file_builder);
        SpvbPhi* header_phi = spvb_add_phi(header_builder, loop_param_type, header_phi_id);
        SpvId param_initial_value = spv_emit_value(emitter, fn_builder, loop_instr.initial_args.nodes[i]);
        spvb_add_phi_source(header_phi, get_block_builder_id(bb_builder), param_initial_value);
        spvb_add_phi_source(header_phi, get_block_builder_id(continue_builder), continue_phi_id);

        BBBuilder body_builder = spv_find_basic_block_builder(emitter, loop_instr.body);
        spvb_add_phi_source(read_list(SpvbPhi*, spbv_get_phis(body_builder))[i], get_block_builder_id(header_builder), header_phi_id);
    }

    fn_builder->per_bb[cfg_lookup(fn_builder->cfg, loop_instr.body)->rpo_index].continue_id = continue_id;
    fn_builder->per_bb[cfg_lookup(fn_builder->cfg, loop_instr.body)->rpo_index].continue_builder = continue_builder;

    SpvId tail_id = spv_find_emitted(emitter, fn_builder, loop_instr.tail);

    // the header block receives the loop merge annotation
     spvb_loop_merge(header_builder, tail_id, continue_id, 0, 0, NULL);
    spvb_branch(header_builder, body_id);

    spvb_add_bb(fn_builder->base, header_builder);

    // the continue block just jumps back into the header
    spvb_branch(continue_builder, header_id);

    spvb_branch(bb_builder, header_id);
}

typedef enum {
    SelectionConstruct,
    LoopConstruct,
} Construct;

static CFNode* find_surrounding_structured_construct_node(Emitter* emitter, FnBuilder* fn_builder, const Node* abs, Construct construct) {
    const Node* oabs = abs;
    for (CFNode* n = cfg_lookup(fn_builder->cfg, abs); n; oabs = n->node, n = n->idom) {
        const Node* terminator = get_abstraction_body(n->node);
        assert(terminator);
        if (is_structured_construct(terminator) && get_structured_construct_tail(terminator) == oabs) {
             continue;
        }
        if (construct == LoopConstruct && terminator->tag == Loop_TAG)
            return n;
        if (construct == SelectionConstruct && terminator->tag == If_TAG)
            return n;
        if (construct == SelectionConstruct && terminator->tag == Match_TAG)
            return n;

    }
    return NULL;
}

static const Node* find_construct(Emitter* emitter, FnBuilder* fn_builder, const Node* abs, Construct construct) {
    CFNode* found = find_surrounding_structured_construct_node(emitter, fn_builder, abs, construct);
    return found ? get_abstraction_body(found->node) : NULL;
}

void spv_emit_terminator(Emitter* emitter, FnBuilder* fn_builder, BBBuilder basic_block_builder, const Node* abs, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case Return_TAG: {
            Return payload = terminator->payload.fn_ret;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            const Nodes* ret_values = &terminator->payload.fn_ret.args;
            switch (ret_values->count) {
                case 0: spvb_return_void(basic_block_builder); return;
                case 1: spvb_return_value(basic_block_builder, spv_emit_value(emitter, fn_builder, ret_values->nodes[0])); return;
                default: {
                    LARRAY(SpvId, arr, ret_values->count);
                    for (size_t i = 0; i < ret_values->count; i++)
                        arr[i] = spv_emit_value(emitter, fn_builder, ret_values->nodes[i]);
                    SpvId return_that = spvb_composite(basic_block_builder, fn_ret_type_id(fn_builder->base), ret_values->count, arr);
                    spvb_return_value(basic_block_builder, return_that);
                    return;
                }
            }
        }
        case Unreachable_TAG: {
            Unreachable payload = terminator->payload.unreachable;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            spvb_unreachable(basic_block_builder);
            return;
        }
        case Jump_TAG: {
            Jump payload = terminator->payload.jump;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            add_branch_phis_from_jump(emitter, fn_builder, basic_block_builder, terminator->payload.jump);
            spvb_branch(basic_block_builder, spv_find_emitted(emitter, fn_builder, terminator->payload.jump.target));
            return;
        }
        case Branch_TAG: {
            Branch payload = terminator->payload.branch;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            SpvId condition = spv_emit_value(emitter, fn_builder, terminator->payload.branch.condition);
            add_branch_phis_from_jump(emitter, fn_builder, basic_block_builder, terminator->payload.branch.true_jump->payload.jump);
            add_branch_phis_from_jump(emitter, fn_builder, basic_block_builder, terminator->payload.branch.false_jump->payload.jump);
            spvb_branch_conditional(basic_block_builder, condition, spv_find_emitted(emitter, fn_builder, terminator->payload.branch.true_jump->payload.jump.target), spv_find_emitted(emitter, fn_builder, terminator->payload.branch.false_jump->payload.jump.target));
            return;
        }
        case Switch_TAG: {
            Switch payload = terminator->payload.br_switch;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            SpvId inspectee = spv_emit_value(emitter, fn_builder, terminator->payload.br_switch.switch_value);
            LARRAY(SpvId, targets, terminator->payload.br_switch.case_jumps.count * 2);
            for (size_t i = 0; i < terminator->payload.br_switch.case_jumps.count; i++) {
                add_branch_phis_from_jump(emitter, fn_builder, basic_block_builder, terminator->payload.br_switch.case_jumps.nodes[i]->payload.jump);
            }
            add_branch_phis_from_jump(emitter, fn_builder, basic_block_builder, terminator->payload.br_switch.default_jump->payload.jump);
            SpvId default_tgt = spv_find_emitted(emitter, fn_builder, terminator->payload.br_switch.default_jump->payload.jump.target);

            spvb_switch(basic_block_builder, inspectee, default_tgt, terminator->payload.br_switch.case_jumps.count, targets);
            return;
        }
        case If_TAG: return emit_if(emitter, fn_builder, basic_block_builder, terminator->payload.if_instr);
        case Match_TAG: return emit_match(emitter, fn_builder, basic_block_builder, terminator->payload.match_instr);
        case Loop_TAG: return emit_loop(emitter, fn_builder, basic_block_builder, abs, terminator->payload.loop_instr);
        case MergeSelection_TAG: {
            MergeSelection payload = terminator->payload.merge_selection;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            const Node* construct = find_construct(emitter, fn_builder, abs, SelectionConstruct);
            assert(construct);
            const Node* tail = get_structured_construct_tail(construct);
            Nodes args = terminator->payload.merge_selection.args;
            add_branch_phis(emitter, fn_builder, basic_block_builder, tail, args);
            assert(tail != abs);
            spvb_branch(basic_block_builder, spv_find_emitted(emitter, fn_builder, tail));
            return;
        }
        case MergeContinue_TAG: {
            MergeContinue payload = terminator->payload.merge_continue;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            const Node* construct = find_construct(emitter, fn_builder, abs, LoopConstruct);
            assert(construct);
            Loop loop_payload = construct->payload.loop_instr;
            CFNode* loop_body = cfg_lookup(fn_builder->cfg, loop_payload.body);
            assert(loop_body);
            Nodes args = terminator->payload.merge_continue.args;
            add_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), fn_builder->per_bb[loop_body->rpo_index].continue_builder, args);
            spvb_branch(basic_block_builder, fn_builder->per_bb[loop_body->rpo_index].continue_id);
            return;
        }
        case MergeBreak_TAG: {
            MergeBreak payload = terminator->payload.merge_break;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            const Node* construct = find_construct(emitter, fn_builder, abs, LoopConstruct);
            assert(construct);
            Loop loop_payload = construct->payload.loop_instr;
            Nodes args = terminator->payload.merge_break.args;
            add_branch_phis(emitter, fn_builder, basic_block_builder, loop_payload.tail, args);
            spvb_branch(basic_block_builder, spv_find_emitted(emitter, fn_builder, loop_payload.tail));
            return;
        }
        case Terminator_Control_TAG:
        case TailCall_TAG:
        case Join_TAG: error("Lower me");
        case NotATerminator: error("TODO: emit terminator %s", node_tags[terminator->tag]);
    }
    SHADY_UNREACHABLE;
}
