#include "passes.h"

#include "log.h"
#include "portability.h"
#include "list.h"

#include "../transform/memory_layout.h"
#include "../transform/ir_gen_helpers.h"

#include "../rewrite.h"
#include "../type.h"

#include <assert.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;

    const Node* stack;
    const Node* stack_pointer;
    const Node* uniform_stack;
    const Node* uniform_stack_pointer;
} Context;

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* old_instruction = node->payload.let.instruction;
    const Node* tail = rewrite_node(&ctx->rewriter, node->payload.let.tail);

    if (old_instruction->tag == PrimOp_TAG) {
        const PrimOp* oprim_op = &old_instruction->payload.prim_op;
        switch (oprim_op->op) {
            case get_stack_pointer_op:
            case get_stack_pointer_uniform_op: {
                BodyBuilder* bb = begin_body(arena);
                bool uniform = oprim_op->op == get_stack_pointer_uniform_op;
                const Node* sp = gen_load(bb, uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer);
                return finish_body(bb, let(arena, quote(arena, sp), tail));
            }
            case set_stack_pointer_op:
            case set_stack_pointer_uniform_op: {
                BodyBuilder* bb = begin_body(arena);
                bool uniform = oprim_op->op == set_stack_pointer_uniform_op;
                const Node* val = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[0]);
                gen_store(bb, uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer, val);
                return finish_body(bb, tail);
            }

            case push_stack_op:
            case push_stack_uniform_op:
            case pop_stack_op:
            case pop_stack_uniform_op: {
                BodyBuilder* bb = begin_body(arena);
                const Type* element_type = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[0]);
                TypeMemLayout layout = get_mem_layout(ctx->config, arena, element_type);
                const Node* element_size = int32_literal(arena, bytes_to_i32_cells(layout.size_in_bytes));

                bool push = oprim_op->op == push_stack_op || oprim_op->op == push_stack_uniform_op;
                bool uniform = oprim_op->op == push_stack_uniform_op || oprim_op->op == pop_stack_uniform_op;

                // TODO somehow annotate the uniform guys as uniform
                const Node* stack_pointer = uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer;
                const Node* stack = uniform ? ctx->uniform_stack : ctx->stack;

                const Node* stack_size = gen_load(bb, stack_pointer);

                if (!push) // for pop, we decrease the stack size first
                    stack_size = gen_primop_ce(bb, sub_op, 2, (const Node* []) { stack_size, element_size});

                const Node* addr = gen_lea(bb, stack, stack_size, nodes(arena, 1, (const Node* []) { int32_literal(arena, 0) }));
                assert(extract_operand_type(addr->type)->tag == PtrType_TAG);
                AddressSpace addr_space = extract_operand_type(addr->type)->payload.ptr_type.address_space;

                addr = gen_primop_ce(bb, reinterpret_op, 2, (const Node* []) { ptr_type(arena, (PtrType) {.address_space = addr_space, .pointed_type = element_type}), addr });

                if (uniform) {
                    assert(is_operand_uniform(stack_pointer->type));
                    assert(is_operand_uniform(stack_size->type));
                    assert(is_operand_uniform(stack->type));
                    assert(is_operand_uniform(addr->type));
                }

                const Node* popped_value = NULL;
                if (push)
                    gen_store(bb, addr, rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[1]));
                else
                    popped_value = gen_primop_ce(bb, load_op, 1, (const Node* []) { addr });

                if (push) // for push, we increase the stack size after the store
                    stack_size = gen_primop_ce(bb, add_op, 2, (const Node* []) { stack_size, element_size});

                // store updated stack size
                gen_store(bb, stack_pointer, stack_size);

                if (push)
                    return finish_body(bb, tail);

                assert(popped_value);
                return finish_body(bb, let(arena, quote(arena, popped_value), tail));
            }
            default: break;
        }
    }

    return let(arena, rewrite_node(&ctx->rewriter, old_instruction), tail);
}

static const Node* process_node(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    switch (old->tag) {
        case Let_TAG: return process_let(ctx, old);
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

void lower_stack(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    IrArena* dst_arena = get_module_arena(dst);

    const Type* stack_base_element = int32_type(dst_arena);
    const Type* stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_thread_stack_size),
    });
    const Type* uniform_stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_subgroup_stack_size),
    });
    const Type* stack_counter_t = int32_type(dst_arena);

    // TODO add a @Synthetic annotation to tag those
    Nodes annotations = nodes(dst_arena, 0, NULL);

    // Arrays for the stacks
    Node* stack_decl = global_var(dst, annotations, stack_arr_type, "stack", AsPrivatePhysical);
    Node* uniform_stack_decl = global_var(dst, annotations, uniform_stack_arr_type, "uniform_stack", AsSubgroupPhysical);

    // Pointers into those arrays
    Node* stack_ptr_decl = global_var(dst, annotations, stack_counter_t, "stack_ptr", AsPrivateLogical);
    stack_ptr_decl->payload.global_variable.init = int32_literal(dst_arena, 0);
    Node* uniform_stack_ptr_decl = global_var(dst, annotations, stack_counter_t, "uniform_stack_ptr", AsPrivateLogical);
    uniform_stack_ptr_decl->payload.global_variable.init = int32_literal(dst_arena, 0);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),

        .config = config,

        .stack = ref_decl(dst_arena, (RefDecl) { .decl = stack_decl }),
        .stack_pointer = ref_decl(dst_arena, (RefDecl) { .decl = stack_ptr_decl }),
        .uniform_stack = ref_decl(dst_arena, (RefDecl) { .decl = uniform_stack_decl }),
        .uniform_stack_pointer = ref_decl(dst_arena, (RefDecl) { .decl = uniform_stack_ptr_decl }),
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
