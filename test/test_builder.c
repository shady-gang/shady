#include "shady/ir.h"
#include "shady/driver.h"
#include "shady/be/dump.h"

#include "../shady/transform/ir_gen_helpers.h"
#include "../shady/analysis/cfg.h"

#include "log.h"
#include "type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x, failure_handler) { if (!(x)) { error_print(#x " failed\n"); failure_handler; } }

static void test_body_builder_constants(IrArena* a) {
    BodyBuilder* bb = begin_block_pure(a);
    const Node* sum = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, int32_literal(a, 4), int32_literal(a, 38)));
    const Node* result = yield_value_and_wrap_in_block(bb, sum);
    CHECK(sum == result, exit(-1));
    CHECK(result->tag == IntLiteral_TAG, exit(-1));
    CHECK(get_int_literal_value(result->payload.int_literal, false) == 42, exit(-1));
}

static void test_body_builder_fun_body(IrArena* a) {
    Module* m = new_module(a, "test_module");
    const Node* p1 = param(a, qualified_type_helper(ptr_type(a, (PtrType) {
        .address_space = AsGeneric,
        .pointed_type = uint32_type(a),
    }), false), NULL);
    const Node* p2 = param(a, qualified_type_helper(ptr_type(a, (PtrType) {
        .address_space = AsGeneric,
        .pointed_type = uint32_type(a),
    }), false), NULL);
    // const Node* p3 = param(a, qualified_type_helper(bool_type(a), false), NULL);
    // const Node* p4 = param(a, qualified_type_helper(uint32_type(a), false), NULL);
    Node* fun = function(m, mk_nodes(a, p1, p2), "fun", empty(a), empty(a));
    BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(fun));

    const Node* p1_value = gen_load(bb, p1);
    CHECK(p1_value->tag == Load_TAG, exit(-1));
    Node* true_case = case_(a, empty(a));
    BodyBuilder* tc_builder = begin_body_with_mem(a, get_abstraction_mem(true_case));
    gen_store(tc_builder, p1, uint32_literal(a, 0));
    set_abstraction_body(true_case, finish_body_with_selection_merge(tc_builder, empty(a)));
    gen_if(bb, empty(a), gen_primop_e(bb, gt_op, empty(a), mk_nodes(a, p1_value, uint32_literal(a, 0))), true_case, NULL);

    const Node* p2_value = gen_load(bb, p2);

    const Node* sum = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, p1_value, p2_value));
    const Node* return_terminator = fn_ret(a, (Return) {
        .mem = bb_mem(bb),
        .args = singleton(sum)
    });
    set_abstraction_body(fun, finish_body(bb, return_terminator));
    // set_abstraction_body(fun, finish_body_with_return(bb, singleton(sum)));

    dump_module(m);

    // Follow the CFG and the mems to make sure we arrive back at the initial start !
    CFG* cfg = build_fn_cfg(fun);
    const Node* mem = get_terminator_mem(return_terminator);
    do {
        mem = get_original_mem(mem);
        CHECK(mem->tag == AbsMem_TAG, exit(-1));
        CFNode* n = cfg_lookup(cfg, mem->payload.abs_mem.abs);
        if (n->idom) {
            mem = get_terminator_mem(get_abstraction_body(n->idom->node));
            continue;
        }
    } while (false);
    mem = get_original_mem(mem);
    CHECK(mem == get_abstraction_mem(fun), exit(-1));
    destroy_cfg(cfg);
}

/// There is some "magic" code in body_builder and set_abstraction_body to enable inserting control-flow
/// where there is only a mem dependency. This is useful when writing some complex polyfills.
static void test_body_builder_impure_block(IrArena* a) {
    Module* m = new_module(a, "test_module");
    const Node* p1 = param(a, qualified_type_helper(ptr_type(a, (PtrType) {
        .address_space = AsGeneric,
        .pointed_type = uint32_type(a),
    }), false), NULL);
    Node* fun = function(m, mk_nodes(a, p1), "fun", empty(a), empty(a));
    BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(fun));

    const Node* first_load = gen_load(bb, p1);

    BodyBuilder* block_builder = begin_block_with_side_effects(a, bb_mem(bb));
    gen_store(block_builder, p1, uint32_literal(a, 0));
    bind_instruction(bb, yield_values_and_wrap_in_block(block_builder, empty(a)));

    const Node* second_load = gen_load(bb, p1);

    const Node* sum = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, first_load, second_load));
    const Node* return_terminator = fn_ret(a, (Return) {
        .mem = bb_mem(bb),
        .args = singleton(sum)
    });
    set_abstraction_body(fun, finish_body(bb, return_terminator));

    dump_module(m);

    bool found_store = false;
    const Node* mem = get_terminator_mem(return_terminator);
    while (mem) {
        if (mem->tag == Store_TAG)
            found_store = true;
        mem = get_parent_mem(mem);
    }

    CHECK(found_store, exit(-1));
}

/// There is some "magic" code in body_builder and set_abstraction_body to enable inserting control-flow
/// where there is only a mem dependency. This is useful when writing some complex polyfills.
static void test_body_builder_impure_block_with_control_flow(IrArena* a) {
    Module* m = new_module(a, "test_module");
    const Node* p1 = param(a, qualified_type_helper(ptr_type(a, (PtrType) {
        .address_space = AsGeneric,
        .pointed_type = uint32_type(a),
    }), false), NULL);
    Node* fun = function(m, mk_nodes(a, p1), "fun", empty(a), empty(a));
    BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(fun));

    const Node* first_load = gen_load(bb, p1);

    BodyBuilder* block_builder = begin_block_with_side_effects(a, bb_mem(bb));
    Node* if_true_case = case_(a, empty(a));
    BodyBuilder* if_true_builder = begin_body_with_mem(a, get_abstraction_mem(if_true_case));
    gen_store(if_true_builder, p1, uint32_literal(a, 0));
    set_abstraction_body(if_true_case, finish_body_with_selection_merge(if_true_builder, empty(a)));
    gen_if(block_builder, empty(a), gen_primop_e(block_builder, neq_op, empty(a), mk_nodes(a, first_load, uint32_literal(a, 0))), if_true_case, NULL);
    bind_instruction(bb, yield_values_and_wrap_in_block(block_builder, empty(a)));

    const Node* second_load = gen_load(bb, p1);

    const Node* sum = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, first_load, second_load));
    const Node* return_terminator = fn_ret(a, (Return) {
        .mem = bb_mem(bb),
        .args = singleton(sum)
    });
    set_abstraction_body(fun, finish_body(bb, return_terminator));

    dump_module(m);
}

int main(int argc, char** argv) {
    cli_parse_common_args(&argc, argv);

    TargetConfig target_config = default_target_config();
    ArenaConfig aconfig = default_arena_config(&target_config);
    IrArena* a = new_ir_arena(&aconfig);
    test_body_builder_constants(a);
    test_body_builder_fun_body(a);
    test_body_builder_impure_block(a);
    test_body_builder_impure_block_with_control_flow(a);
    destroy_ir_arena(a);
}
