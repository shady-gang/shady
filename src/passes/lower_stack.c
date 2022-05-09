#include "shady/ir.h"

#include "../transform/memory_layout.h"
#include "../transform/ir_gen_helpers.h"

#include "../rewrite.h"
#include "../type.h"
#include "../log.h"
#include "../portability.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;

    const Node* stack;
    const Node* stack_pointer;
    const Node* uniform_stack;
    const Node* uniform_stack_pointer;

    struct List* new_decls;
} Context;

static const Node* handle_block(Context* ctx, const Node* node) {
    assert(node->tag == Block_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    Instructions instructions = begin_instructions(dst_arena);
    Nodes oinstructions = node->payload.block.instructions;

    for (size_t i = 0; i < oinstructions.count; i++) {
        const Node* olet = oinstructions.nodes[i];
        const Node* oinstruction = olet->payload.let.instruction;
        if (oinstruction->tag == PrimOp_TAG) {
            const PrimOp* oprim_op = &oinstruction->payload.prim_op;
            switch (oprim_op->op) {
                case push_stack_op:
                case push_stack_uniform_op:
                case pop_stack_op:
                case pop_stack_uniform_op: {
                    const Type* element_type = oprim_op->operands.nodes[0];
                    TypeMemLayout layout = get_mem_layout(ctx->config, element_type);
                    const Node* element_size = int_literal(dst_arena, (IntLiteral) {.value = layout.size_in_cells });

                    bool push = oprim_op->op == push_stack_op || oprim_op->op == push_stack_uniform_op;
                    bool uniform = oprim_op->op == push_stack_uniform_op || oprim_op->op == pop_stack_uniform_op;

                    // TODO somehow annotate the uniform guys as uniform
                    const Node* stack_pointer = uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer;
                    const Node* stack = uniform ? ctx->uniform_stack : ctx->stack;

                    const Node* stack_size = gen_load(instructions, stack_pointer);

                    if (!push) // for pop, we decrease the stack size first
                        stack_size = gen_primop(instructions, (PrimOp) {
                            .op = sub_op,
                            .operands = nodes(dst_arena, 2, (const Node* []) { stack_size, element_size})
                        }).nodes[0];

                    const Node* addr = gen_lea(instructions, stack, stack_size, nodes(dst_arena, 1, (const Node* []) { int_literal(dst_arena, (IntLiteral) {.value = 0})}));
                    assert(without_qualifier(addr->type)->tag == PtrType_TAG);
                    AddressSpace addr_space = without_qualifier(addr->type)->payload.ptr_type.address_space;

                    addr = gen_primop(instructions, (PrimOp) {
                        .op = cast_ptr_op,
                        .operands = nodes(dst_arena, 2, (const Node* []) { ptr_type(dst_arena, (PtrType) {.address_space = addr_space, .pointed_type = element_type}), addr })
                    }).nodes[0];

                    if (uniform) {
                        assert(get_qualifier(stack_pointer->type) == Uniform);
                        assert(get_qualifier(stack_size->type) == Uniform);
                        assert(get_qualifier(stack->type) == Uniform);
                        assert(get_qualifier(addr->type) == Uniform);
                    }

                    if (push)
                        gen_store(instructions, addr, oprim_op->operands.nodes[1]);
                    else {
                        const Node* popped = gen_primop(instructions, (PrimOp) {
                            .op = load_op,
                            .operands = nodes(dst_arena, 1, (const Node* []) {addr})
                        }).nodes[0];
                        register_processed(&ctx->rewriter, olet->payload.let.variables.nodes[0], popped);
                    }

                    if (push)
                        stack_size = gen_primop(instructions, (PrimOp) {
                            .op = add_op,
                            .operands = nodes(dst_arena, 2, (const Node* []) { stack_size, element_size})
                        }).nodes[0];

                    // store updated stack size
                    gen_store(instructions, stack_pointer, stack_size);

                    continue;
                }
                default: goto unchanged;
            }
        }

        unchanged:
        append_instr(instructions, recreate_node_identity(&ctx->rewriter, olet));
    }

    return block(ctx->rewriter.dst_arena, (Block) {
        .instructions = finish_instructions(instructions),
        .terminator = recreate_node_identity(&ctx->rewriter, node->payload.block.terminator),
    });
}

static const Node* process_node(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    switch (old->tag) {
        case Constant_TAG:
        case Function_TAG:
        case GlobalVariable_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            recreate_decl_body_identity(&ctx->rewriter, old, new);
            return new;
        }
        case Block_TAG: return handle_block(ctx, old);
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* lower_stack(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* new_decls_list = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    const Type* stack_base_element = int_type(dst_arena);
    const Type* stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = NULL,
    });
    const Type* stack_counter_t = int_type(dst_arena);

    Node* stack_decl = global_var(dst_arena, stack_arr_type, "stack", AsPrivatePhysical);
    Node* uniform_stack_decl = global_var(dst_arena, stack_arr_type, "uniform_stack", AsSubgroupPhysical);

    Node* stack_ptr_decl = global_var(dst_arena, stack_counter_t, "stack_ptr", AsPrivateLogical);
    stack_ptr_decl->payload.global_variable.init = int_literal(dst_arena, (IntLiteral) {.value = 0});
    Node* uniform_stack_ptr_decl = global_var(dst_arena, stack_counter_t, "uniform_stack_ptr", AsPrivateLogical);
    uniform_stack_ptr_decl->payload.global_variable.init = int_literal(dst_arena, (IntLiteral) {.value = 0});

    append_list(const Node*, new_decls_list, stack_decl);
    append_list(const Node*, new_decls_list, uniform_stack_decl);
    append_list(const Node*, new_decls_list, stack_ptr_decl);
    append_list(const Node*, new_decls_list, uniform_stack_ptr_decl);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) process_node,
            .rewrite_decl_body = NULL,
            .processed = done,
        },

        .config = config,

        .stack = stack_decl,
        .stack_pointer = stack_ptr_decl,
        .uniform_stack = uniform_stack_decl,
        .uniform_stack_pointer = uniform_stack_ptr_decl,

        .new_decls = new_decls_list,
    };

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);

    Nodes new_decls = rewritten->payload.root.declarations;
    for (size_t i = 0; i < entries_count_list(new_decls_list); i++) {
        new_decls = append_nodes(dst_arena, new_decls, read_list(const Node*, new_decls_list)[i]);
    }
    rewritten = root(dst_arena, (Root) {
        .declarations = new_decls
    });

    destroy_list(new_decls_list);

    destroy_dict(done);
    return rewritten;
}
