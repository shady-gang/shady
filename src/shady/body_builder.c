#include "shady/ir/builder.h"
#include "shady/ir/grammar.h"
#include "shady/ir/function.h"
#include "shady/ir/composite.h"
#include "shady/ir/arena.h"
#include "shady/ir/mem.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"

#include <stdlib.h>
#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

struct BodyBuilder_ {
    IrArena* arena;
    struct List* stack;
    const Node* block_entry_block;
    const Node* block_entry_mem;
    const Node* mem;
    Node* tail_block;
};

typedef struct {
    Structured_constructTag tag;
    union NodesUnion payload;
} BlockEntry;

typedef struct {
    BlockEntry structured;
    Nodes vars;
} StackEntry;

BodyBuilder* shd_bld_begin(IrArena* a, const Node* mem) {
    BodyBuilder* bb = malloc(sizeof(BodyBuilder));
    *bb = (BodyBuilder) {
        .arena = a,
        .stack = shd_new_list(StackEntry),
        .mem = mem,
    };
    return bb;
}

BodyBuilder* shd_bld_begin_pseudo_instr(IrArena* a, const Node* mem) {
    Node* block = basic_block_helper(a, shd_empty(a));
    BodyBuilder* builder = shd_bld_begin(a, shd_get_abstraction_mem(block));
    builder->tail_block = block;
    builder->block_entry_block = block;
    builder->block_entry_mem = mem;
    return builder;
}

BodyBuilder* shd_bld_begin_pure(IrArena* a) {
    BodyBuilder* builder = shd_bld_begin(a, NULL);
    return builder;
}

IrArena* shd_get_bb_arena(BodyBuilder* bb) {
    return bb->arena;
}

const Node* _shd_bb_insert_mem(BodyBuilder* bb) {
    return bb->block_entry_mem;
}

const Node* _shd_bb_insert_block(BodyBuilder* bb) {
    return bb->block_entry_block;
}

const Node* shd_bld_mem(BodyBuilder* bb) {
    return bb->mem;
}

static void bind_internal(BodyBuilder* bb, const Node* instruction) {
    if (shd_get_arena_config(bb->arena)->check_types) {
        assert(is_mem(instruction));
    }
    if (is_mem(instruction) && /* avoid things like ExtInstr with null mem input! */ shd_get_parent_mem(instruction))
        bb->mem = instruction;
}

const Node* shd_bld_add_instruction(BodyBuilder* bb, const Node* instr) {
    bind_internal(bb, instr);
    return instr;
    // return shd_first(shd_bld_add_instruction_extract_count(bb, instr, 1));
}

Nodes shd_bld_add_instruction_extract(BodyBuilder* bb, const Node* instruction) {
    assert(shd_get_arena_config(bb->arena)->check_types);
    assert(is_value(instruction));
    Nodes types = shd_unwrap_multiple_yield_types(bb->arena, instruction->type);
    bind_internal(bb, instruction);
    return shd_deconstruct_composite(bb->arena, instruction, types.count);
}

Nodes shd_bld_add_instruction_extract_count(BodyBuilder* bb, const Node* instruction, size_t outputs_count) {
    assert(is_value(instruction));
    bind_internal(bb, instruction);
    return shd_deconstruct_composite(bb->arena, instruction, outputs_count);
}

static const Node* build_body(BodyBuilder* bb, const Node* terminator) {
    IrArena* a = bb->arena;
    size_t stack_size = shd_list_count(bb->stack);
    for (size_t i = stack_size - 1; i < stack_size; i--) {
        StackEntry entry = shd_read_list(StackEntry, bb->stack)[i];
        const Node* t2 = terminator;
        switch (entry.structured.tag) {
            case NotAStructured_construct: shd_error("")
            case Structured_construct_If_TAG: {
                terminator = if_instr(a, entry.structured.payload.if_instr);
                break;
            }
            case Structured_construct_Match_TAG: {
                terminator = match_instr(a, entry.structured.payload.match_instr);
                break;
            }
            case Structured_construct_Loop_TAG: {
                terminator = loop_instr(a, entry.structured.payload.loop_instr);
                break;
            }
            case Structured_construct_Control_TAG: {
                terminator = control(a, entry.structured.payload.control);
                break;
            }
        }
        shd_set_abstraction_body((Node*) get_structured_construct_tail(terminator), t2);
    }
    return terminator;
}

const Node* shd_bld_finish(BodyBuilder* bb, const Node* terminator) {
    assert(bb->mem && !bb->block_entry_mem);
    terminator = build_body(bb, terminator);
    shd_destroy_list(bb->stack);
    free(bb);
    return terminator;
}

const Node* shd_bld_return(BodyBuilder* bb, Nodes args) {
    return shd_bld_finish(bb, fn_ret(bb->arena, (Return) {
        .args = args,
        .mem = shd_bld_mem(bb)
    }));
}

const Node* shd_bld_unreachable(BodyBuilder* bb) {
    return shd_bld_finish(bb, unreachable(bb->arena, (Unreachable) {
        .mem = shd_bld_mem(bb)
    }));
}

const Node* shd_bld_selection_merge(BodyBuilder* bb, Nodes args) {
    return shd_bld_finish(bb, merge_selection(bb->arena, (MergeSelection) {
        .args = args,
        .mem = shd_bld_mem(bb),
    }));
}

const Node* shd_bld_loop_continue(BodyBuilder* bb, Nodes args)  {
    return shd_bld_finish(bb, merge_continue(bb->arena, (MergeContinue) {
        .args = args,
        .mem = shd_bld_mem(bb),
    }));
}

const Node* shd_bld_loop_break(BodyBuilder* bb, Nodes args) {
    return shd_bld_finish(bb, merge_break(bb->arena, (MergeBreak) {
        .args = args,
        .mem = shd_bld_mem(bb),
    }));
}

const Node* shd_bld_join(BodyBuilder* bb, const Node* jp, Nodes args) {
    return shd_bld_finish(bb, join(bb->arena, (Join) {
        .join_point = jp,
        .args = args,
        .mem = shd_bld_mem(bb),
    }));
}

const Node* shd_bld_jump(BodyBuilder* bb, const Node* target, Nodes args) {
    return shd_bld_finish(bb, jump(bb->arena, (Jump) {
        .target = target,
        .args = args,
        .mem = shd_bld_mem(bb),
    }));
}

const Node* shd_bld_indirect_tail_call(BodyBuilder* bb, const Node* target, Nodes args) {
    return shd_bld_finish(bb, indirect_tail_call(bb->arena, (IndirectTailCall) {
        .callee = target,
        .args = args,
        .mem = shd_bld_mem(bb),
    }));
}

const Node* shd_bld_to_instr_yield_value(BodyBuilder* bb, const Node* value) {
    IrArena* a = bb->arena;
    if (!bb->tail_block && shd_list_count(bb->stack) == 0) {
        const Node* last_mem = shd_bld_mem(bb);
        shd_bld_cancel(bb);
        if (last_mem)
            return mem_and_value(a, (MemAndValue) {
                .mem = last_mem,
                .value = value
            });
        return value;
    }
    assert(bb->block_entry_mem && "This builder wasn't started with 'shd_bld_begin_pure' or 'shd_bld_begin_pseudo_instr'");
    bb->tail_block->payload.basic_block.insert = bb;
    const Node* r = mem_and_value(bb->arena, (MemAndValue) {
        .mem = shd_bld_mem(bb),
        .value = value
    });
    return r;
}

const Node* shd_bld_to_instr_yield_values(BodyBuilder* bb, Nodes values) {
    return shd_bld_to_instr_yield_value(bb, shd_maybe_tuple_helper(bb->arena, values));
}

const Node* _shd_bld_finish_pseudo_instr(BodyBuilder* bb, const Node* terminator) {
    assert(bb->block_entry_mem);
    terminator = build_body(bb, terminator);
    shd_destroy_list(bb->stack);
    free(bb);
    return terminator;
}

const Node* shd_bld_to_instr_with_last_instr(BodyBuilder* bb, const Node* instruction) {
    size_t stack_size = shd_list_count(bb->stack);
    if (stack_size == 0) {
        shd_bld_cancel(bb);
        return instruction;
    }
    bind_internal(bb, instruction);
    return shd_bld_to_instr_yield_value(bb, instruction);
}

const Node* shd_bld_to_instr_pure_with_values(BodyBuilder* bb, Nodes values) {
    IrArena* arena = bb->arena;
    assert(!bb->mem && !bb->block_entry_mem && shd_list_count(bb->stack) == 0);
    shd_bld_cancel(bb);
    return shd_maybe_tuple_helper(arena, values);
}

static Nodes gen_variables(BodyBuilder* bb, Nodes yield_types) {
    IrArena* a = bb->arena;

    Nodes qyield_types = shd_add_qualifiers(a, yield_types, false);
    LARRAY(const Node*, tail_params, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++)
        tail_params[i] = param_helper(a, qyield_types.nodes[i]);
    return shd_nodes(a, yield_types.count, tail_params);
}

static Nodes add_structured_construct(BodyBuilder* bb, Nodes params, Structured_constructTag tag, union NodesUnion payload) {
    Node* tail = basic_block_helper(bb->arena, params);
    StackEntry entry = {
        .structured = {
            .tag = tag,
            .payload = payload,
        },
        .vars = params,
    };
    switch (entry.structured.tag) {
        case NotAStructured_construct: shd_error("")
        case Structured_construct_If_TAG: {
            entry.structured.payload.if_instr.tail = tail;
            entry.structured.payload.if_instr.mem = shd_bld_mem(bb);
            break;
        }
        case Structured_construct_Match_TAG: {
            entry.structured.payload.match_instr.tail = tail;
            entry.structured.payload.match_instr.mem = shd_bld_mem(bb);
            break;
        }
        case Structured_construct_Loop_TAG: {
            entry.structured.payload.loop_instr.tail = tail;
            entry.structured.payload.loop_instr.mem = shd_bld_mem(bb);
            break;
        }
        case Structured_construct_Control_TAG: {
            entry.structured.payload.control.tail = tail;
            entry.structured.payload.control.mem = shd_bld_mem(bb);
            break;
        }
    }
    bb->mem = shd_get_abstraction_mem(tail);
    shd_list_append(StackEntry , bb->stack, entry);
    bb->tail_block = tail;
    return entry.vars;
}

static Nodes gen_structured_construct(BodyBuilder* bb, Nodes yield_types, Structured_constructTag tag, union NodesUnion payload) {
    return add_structured_construct(bb, gen_variables(bb, yield_types), tag, payload);
}

Nodes shd_bld_if(BodyBuilder* bb, Nodes yield_types, const Node* condition, const Node* true_case, Node* false_case) {
    return gen_structured_construct(bb, yield_types, Structured_construct_If_TAG, (union NodesUnion) {
        .if_instr = {
            .condition = condition,
            .if_true = true_case,
            .if_false = false_case,
            .yield_types = yield_types,
        }
    });
}

Nodes shd_bld_match(BodyBuilder* bb, Nodes yield_types, const Node* inspectee, Nodes literals, Nodes cases, Node* default_case) {
    return gen_structured_construct(bb, yield_types, Structured_construct_Match_TAG, (union NodesUnion) {
        .match_instr = {
            .yield_types = yield_types,
            .inspect = inspectee,
            .literals = literals,
            .cases = cases,
            .default_case = default_case
        }
    });
}

Nodes shd_bld_loop(BodyBuilder* bb, Nodes yield_types, Nodes initial_args, Node* body) {
    return gen_structured_construct(bb, yield_types, Structured_construct_Loop_TAG, (union NodesUnion) {
        .loop_instr = {
            .yield_types = yield_types,
            .initial_args = initial_args,
            .body = body
        },
    });
}

Nodes shd_bld_control(BodyBuilder* bb, Nodes yield_types, Node* body) {
    return gen_structured_construct(bb, yield_types, Structured_construct_Control_TAG, (union NodesUnion) {
        .control = {
            .yield_types = yield_types,
            .inside = body
        },
    });
}

begin_control_t shd_bld_begin_control(BodyBuilder* bb, Nodes yield_types) {
    IrArena* a = bb->arena;
    const Type* jp_type = qualified_type(a, (QualifiedType) {
            .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
            .is_uniform = true
    });
    const Node* jp = param_helper(a, jp_type);
    Node* c = basic_block_helper(a, shd_singleton(jp));
    return (begin_control_t) {
        .results = shd_bld_control(bb, yield_types, c),
        .case_ = c,
        .jp = jp
    };
}

begin_loop_helper_t shd_bld_begin_loop_helper(BodyBuilder* bb, Nodes yield_types, Nodes arg_types, Nodes initial_values) {
    assert(arg_types.count == initial_values.count);
    IrArena* a = bb->arena;
    begin_control_t outer_control = shd_bld_begin_control(bb, yield_types);
    BodyBuilder* outer_control_case_builder = shd_bld_begin(a, shd_get_abstraction_mem(outer_control.case_));
    LARRAY(const Node*, params, arg_types.count);
    for (size_t i = 0; i < arg_types.count; i++) {
        params[i] = param_helper(a, shd_as_qualified_type(arg_types.nodes[i], false));
    }
    Node* loop_header = basic_block_helper(a, shd_nodes(a, arg_types.count, params));
    shd_set_abstraction_body(outer_control.case_, shd_bld_jump(outer_control_case_builder, loop_header, initial_values));
    BodyBuilder* loop_header_builder = shd_bld_begin(a, shd_get_abstraction_mem(loop_header));
    begin_control_t inner_control = shd_bld_begin_control(loop_header_builder, arg_types);
    shd_set_abstraction_body(loop_header, shd_bld_jump(loop_header_builder, loop_header, inner_control.results));

    return (begin_loop_helper_t) {
        .results = outer_control.results,
        .params = shd_nodes(a, arg_types.count, params),
        .loop_body = inner_control.case_,
        .break_jp = outer_control.jp,
        .continue_jp = inner_control.jp,
    };
}

void shd_bld_cancel(BodyBuilder* bb) {
    for (size_t i = 0; i < shd_list_count(bb->stack); i++) {
        StackEntry entry = shd_read_list(StackEntry, bb->stack)[i];
        // if (entry.structured.tag != NotAStructured_construct)
        //     destroy_list(entry.structured.stack);
    }
    shd_destroy_list(bb->stack);
    //destroy_list(bb->stack_stack);
    free(bb);
}

#include "shady/rewrite.h"

BodyBuilder* shd_bld_begin_fn_rewrite(Rewriter* r, const Node* old, Node** new) {
    assert(old);
    *new = shd_recreate_node_head(r, old);
    shd_register_processed(r, old, *new);
    BodyBuilder* bld = shd_bld_begin(r->dst_arena, shd_get_abstraction_mem(*new));
    return bld;
}

void shd_bld_finish_fn_rewrite(Rewriter* r, const Node* old, Node* new, BodyBuilder* bld) {
    shd_register_processed(r, shd_get_abstraction_mem(old), shd_bld_mem(bld));
    shd_set_abstraction_body(new, shd_bld_finish(bld, shd_rewrite_node(r, get_abstraction_body(old))));
}
