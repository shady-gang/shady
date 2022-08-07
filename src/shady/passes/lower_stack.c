#include "shady/ir.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"

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

    struct List* new_decls;
} Context;

static const Node* handle_block(Context* ctx, const Node* node) {
    assert(node->tag == Block_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    BlockBuilder* instructions = begin_block(dst_arena);
    Nodes oinstructions = node->payload.block.instructions;

    for (size_t i = 0; i < oinstructions.count; i++) {
        const Node* oinstruction = oinstructions.nodes[i];
        const Node* olet = NULL;
        if (oinstruction->tag == Let_TAG) {
            olet = oinstruction;
            oinstruction = olet->payload.let.instruction;
        }

        if (oinstruction->tag == PrimOp_TAG) {
            const PrimOp* oprim_op = &oinstruction->payload.prim_op;
            switch (oprim_op->op) {
                case push_stack_op:
                case push_stack_uniform_op:
                case pop_stack_op:
                case pop_stack_uniform_op: {
                    const Type* element_type = oprim_op->operands.nodes[0];
                    TypeMemLayout layout = get_mem_layout(ctx->config, dst_arena, element_type);
                    const Node* element_size = int_literal(dst_arena, (IntLiteral) { .value_i32 = bytes_to_i32_cells(layout.size_in_bytes), .width = IntTy32 });

                    bool push = oprim_op->op == push_stack_op || oprim_op->op == push_stack_uniform_op;
                    bool uniform = oprim_op->op == push_stack_uniform_op || oprim_op->op == pop_stack_uniform_op;

                    // TODO somehow annotate the uniform guys as uniform
                    const Node* stack_pointer = uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer;
                    const Node* stack = uniform ? ctx->uniform_stack : ctx->stack;

                    const Node* stack_size = gen_load(instructions, stack_pointer);

                    if (!push) // for pop, we decrease the stack size first
                        stack_size = gen_primop_ce(instructions, sub_op, 2, (const Node* []) { stack_size, element_size});

                    const Node* addr = gen_lea(instructions, stack, stack_size, nodes(dst_arena, 1, (const Node* []) { int_literal(dst_arena, (IntLiteral) { .value_i32 = 0, .width = IntTy32 }) }));
                    assert(without_qualifier(addr->type)->tag == PtrType_TAG);
                    AddressSpace addr_space = without_qualifier(addr->type)->payload.ptr_type.address_space;

                    addr = gen_primop_ce(instructions, reinterpret_op, 2, (const Node* []) { ptr_type(dst_arena, (PtrType) {.address_space = addr_space, .pointed_type = element_type}), addr });

                    if (uniform) {
                        assert(get_qualifier(stack_pointer->type) == Uniform);
                        assert(get_qualifier(stack_size->type) == Uniform);
                        assert(get_qualifier(stack->type) == Uniform);
                        assert(get_qualifier(addr->type) == Uniform);
                    }

                    if (push) {
                        const Node* new_value = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[1]);
                        gen_store(instructions, addr, new_value);
                    } else {
                        const Node* popped = gen_primop_ce(instructions, load_op, 1, (const Node* []) {addr});
                        register_processed(&ctx->rewriter, olet->payload.let.variables.nodes[0], popped);
                    }

                    if (push)
                        stack_size = gen_primop_ce(instructions, add_op, 2, (const Node* []) { stack_size, element_size});

                    // store updated stack size
                    gen_store(instructions, stack_pointer, stack_size);

                    continue;
                }
                default: goto unchanged;
            }
        }

        unchanged:
        append_block(instructions, recreate_node_identity(&ctx->rewriter, oinstructions.nodes[i]));
    }

    return finish_block(instructions, recreate_node_identity(&ctx->rewriter, node->payload.block.terminator));
}

static const Node* process_node(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    switch (old->tag) {
        case Constant_TAG:
        case Function_TAG:
        case GlobalVariable_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            debug_print("processing declaration %s \n", get_decl_name(old));
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

    const Type* stack_base_element = int32_type(dst_arena);
    const Type* stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = NULL,
    });
    const Type* stack_counter_t = int32_type(dst_arena);

    // TODO add a @Synthetic annotation to tag those
    Nodes annotations = nodes(dst_arena, 0, NULL);

    // Arrays for the stacks
    Node* stack_decl = global_var(dst_arena, annotations, stack_arr_type, "stack", AsPrivatePhysical);
    Node* uniform_stack_decl = global_var(dst_arena, annotations, stack_arr_type, "uniform_stack", AsSubgroupPhysical);

    // Pointers into those arrays
    Node* stack_ptr_decl = global_var(dst_arena, annotations, stack_counter_t, "stack_ptr", AsPrivateLogical);
    stack_ptr_decl->payload.global_variable.init = int_literal(dst_arena, (IntLiteral) { .value_i32 = 0, .width = IntTy32 });
    Node* uniform_stack_ptr_decl = global_var(dst_arena, annotations, stack_counter_t, "uniform_stack_ptr", AsPrivateLogical);
    uniform_stack_ptr_decl->payload.global_variable.init = int_literal(dst_arena, (IntLiteral) { .value_i32 = 0, .width = IntTy32});

    append_list(const Node*, new_decls_list, stack_decl);
    append_list(const Node*, new_decls_list, uniform_stack_decl);
    append_list(const Node*, new_decls_list, stack_ptr_decl);
    append_list(const Node*, new_decls_list, uniform_stack_ptr_decl);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) process_node,
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
