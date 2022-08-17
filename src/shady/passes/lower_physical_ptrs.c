#include "shady/ir.h"

#include "../transform/memory_layout.h"
#include "../transform/ir_gen_helpers.h"

#include "../rewrite.h"
#include "../type.h"
#include "log.h"
#include "portability.h"

#include "list.h"
#include "dict.h"

#include <string.h>
#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;
    const Node* src_program;

    uint32_t preallocated_private_memory;
    uint32_t preallocated_subgroup_memory;

    const Node* physical_private_buffer;
    const Node* physical_subgroup_buffer;

    struct List* new_decls;
} Context;

// TODO: make this configuration-dependant
static bool is_as_emulated(SHADY_UNUSED Context* ctx, AddressSpace as) {
    switch (as) {
        case AsSubgroupPhysical: return true;
        case AsPrivatePhysical:  return true;
        case AsSharedPhysical: error("TODO");
        case AsGlobalPhysical: return false; // TODO config
        default: return false;
    }
}

// TODO consume layouts from memory_layout.h
static const Node* lower_lea(Context* ctx, BlockBuilder* instructions, const PrimOp* lea) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    const Node* old_pointer = lea->operands.nodes[0];
    const Node* faked_pointer = rewrite_node(&ctx->rewriter, old_pointer);
    const Type* pointer_type = extract_operand_type(old_pointer->type);
    assert(pointer_type->tag == PtrType_TAG);

    const Node* old_offset = lea->operands.nodes[1];
    if (old_offset) {
        const Type* arr_type = pointer_type->payload.ptr_type.pointed_type;
        assert(arr_type->tag == ArrType_TAG);
        const Type* element_type = arr_type->payload.arr_type.element_type;
        TypeMemLayout element_t_layout = get_mem_layout(ctx->config, ctx->rewriter.dst_arena, element_type);

        const Node* elem_size_val = int_literal(dst_arena, (IntLiteral) { .value_i32 = bytes_to_i32_cells(element_t_layout.size_in_bytes), .width = IntTy32 });
        const Node* computed_offset = gen_primop_ce(instructions, mul_op, 2, (const Node* []) { rewrite_node(&ctx->rewriter, old_offset), elem_size_val});

        faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) { faked_pointer, computed_offset});
    }

    for (size_t i = 2; i < lea->operands.count; i++) {
        assert(pointer_type->tag == PtrType_TAG);
        const Type* pointed_type = pointer_type->payload.ptr_type.pointed_type;
        switch (pointed_type->tag) {
            case ArrType_TAG: {
                const Type* element_type = pointed_type->payload.arr_type.element_type;

                TypeMemLayout element_t_layout = get_mem_layout(ctx->config, ctx->rewriter.dst_arena, element_type);

                const Node* elem_size_val = int_literal(dst_arena, (IntLiteral) { .value_i32 = bytes_to_i32_cells(element_t_layout.size_in_bytes), .width = IntTy32 });
                const Node* computed_offset = gen_primop_ce(instructions, mul_op, 2, (const Node* []) { rewrite_node(&ctx->rewriter, lea->operands.nodes[i]), elem_size_val});

                faked_pointer = gen_primop_ce(instructions, add_op, 2, (const Node* []) { faked_pointer, computed_offset});

                pointer_type = ptr_type(dst_arena, (PtrType) {
                    .pointed_type = element_type,
                    .address_space = pointer_type->payload.ptr_type.address_space
                });
                break;
            }
            case RecordType_TAG: {
                error("TODO");
            }
            default: error("cannot index into this")
        }
    }

    return faked_pointer;
}

static const Node* find_decl(Context* ctx, const char* name) {
    for (size_t i = 0; i < ctx->src_program->payload.root.declarations.count; i++) {
        const Node* decl = ctx->src_program->payload.root.declarations.nodes[i];
        if (strcmp(get_decl_name(decl), name) == 0)
            return rewrite_node(&ctx->rewriter, decl);
    }
    assert(false);
}

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
                case alloca_op: {
                    if (!olet) continue;
                    const Type* element_type = oprim_op->operands.nodes[0];
                    TypeMemLayout layout = get_mem_layout(ctx->config, dst_arena, element_type);

                    const Node* stack_pointer = find_decl(ctx, "stack_ptr");
                    const Node* stack_size = gen_load(instructions, stack_pointer);
                    register_processed(&ctx->rewriter, olet->payload.let.variables.nodes[0], stack_size);
                    stack_size = gen_primop_ce(instructions, add_op, 2, (const Node* []) { stack_size, int_literal(dst_arena, (IntLiteral) { .value_i32 = bytes_to_i32_cells(layout.size_in_bytes), .width = IntTy32 }) });
                    gen_store(instructions, stack_pointer, stack_size);
                    continue;
                }
                case lea_op: {
                    const Type* ptr_type = oprim_op->operands.nodes[0]->type;
                    ptr_type = extract_operand_type(ptr_type);
                    assert(ptr_type->tag == PtrType_TAG);
                    if (!is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                        goto unchanged;
                    const Node* new = lower_lea(ctx, instructions, oprim_op);
                    if (olet)
                        register_processed(&ctx->rewriter, olet->payload.let.variables.nodes[0], new);
                    continue;
                }
                case reinterpret_op: {
                    const Type* source_type = oprim_op->operands.nodes[1]->type;
                    source_type = extract_operand_type(source_type);
                    if (source_type->tag != PtrType_TAG || !is_as_emulated(ctx, source_type->payload.ptr_type.address_space))
                        goto unchanged;
                    // TODO ensure source is an integer and the bit width is appropriate
                    if (olet)
                        register_processed(&ctx->rewriter, olet->payload.let.variables.nodes[0], rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[1]));
                    continue;
                }
                case load_op:
                case store_op: {
                    const Node* old_ptr = oprim_op->operands.nodes[0];
                    const Type* ptr_type = old_ptr->type;
                    ptr_type = extract_operand_type(ptr_type);
                    assert(ptr_type->tag == PtrType_TAG);
                    if (!is_as_emulated(ctx, ptr_type->payload.ptr_type.address_space))
                        goto unchanged;

                    const Type* element_type = ptr_type->payload.ptr_type.pointed_type;

                    const Node* base = NULL;
                    switch (ptr_type->payload.ptr_type.address_space) {
                        case AsSubgroupPhysical: base = ctx->physical_subgroup_buffer; break;
                        case AsPrivatePhysical: base = ctx->physical_private_buffer; break;
                        default: error("Emulation of this AS is not supported");
                    }

                    base = gen_lea(instructions, base, NULL, nodes(dst_arena, 1, (const Node* []) { int_literal(dst_arena, (IntLiteral) { .width = IntTy32, .value_i32 = 0 }) }));

                    const Node* fake_ptr = rewrite_node(&ctx->rewriter, old_ptr);

                    if (oprim_op->op == load_op) {
                        const Node* result = gen_deserialisation(instructions, element_type, base, fake_ptr);
                        if (olet)
                            register_processed(&ctx->rewriter, olet->payload.let.variables.nodes[0], result);
                    } else {
                        const Node* value = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[1]);
                        gen_serialisation(instructions, element_type, base, fake_ptr, value);
                    }
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
        case GlobalVariable_TAG: {
            const GlobalVariable* old_gvar = &old->payload.global_variable;
            // Global variables into emulated address spaces become integer constants (to index into arrays used for emulation of said address space)
            if (old_gvar->address_space == AsSubgroupPhysical || old_gvar->address_space == AsPrivatePhysical) {
                Nodes annotations = rewrite_nodes(&ctx->rewriter, old_gvar->annotations); // We keep the old annotations

                const char* emulated_heap_name = old_gvar->address_space == AsSubgroupPhysical ? "private" : "subgroup";

                Node* cnst = constant(ctx->rewriter.dst_arena, annotations, format_string(ctx->rewriter.dst_arena, "%s_offset_%s_arr", old_gvar->name, emulated_heap_name));

                uint32_t* preallocated = old_gvar->address_space == AsSubgroupPhysical ? &ctx->preallocated_subgroup_memory : &ctx->preallocated_private_memory;
                const Type* contents_type = rewrite_node(&ctx->rewriter, old_gvar->type);
                assert(!contains_qualified_type(contents_type));
                uint32_t required_space = bytes_to_i32_cells(get_mem_layout(ctx->config, ctx->rewriter.dst_arena, contents_type).size_in_bytes);

                cnst->payload.constant.value = int_literal(ctx->rewriter.dst_arena, (IntLiteral) { .value_u32 = *preallocated, .width = IntTy32 });
                cnst->type = cnst->payload.constant.value->type;
                *preallocated += required_space;

                register_processed(&ctx->rewriter, old, cnst);
                return cnst;
            }
            SHADY_FALLTHROUGH
        }
        case Constant_TAG:
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            recreate_decl_body_identity(&ctx->rewriter, old, new);
            return new;
        }
        case Block_TAG: return handle_block(ctx, old);
        case PtrType_TAG: {
            if (is_as_emulated(ctx, old->payload.ptr_type.address_space))
                return int32_type(ctx->rewriter.dst_arena);

            return recreate_node_identity(&ctx->rewriter, old);
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

void update_base_stack_ptrs(Context* ctx) {
    Node* per_thread_stack_ptr = (Node*) find_decl(ctx, "stack_ptr");
    assert(per_thread_stack_ptr && per_thread_stack_ptr->tag == GlobalVariable_TAG);
    per_thread_stack_ptr->payload.global_variable.init = int_literal(ctx->rewriter.dst_arena, (IntLiteral) { .value_u32 = ctx->preallocated_private_memory, .width = IntTy32});
    Node* subgroup_stack_ptr = (Node*)  find_decl(ctx, "uniform_stack_ptr");
    assert(subgroup_stack_ptr && subgroup_stack_ptr->tag == GlobalVariable_TAG);
    subgroup_stack_ptr->payload.global_variable.init = int_literal(ctx->rewriter.dst_arena, (IntLiteral) { .value_u32 = ctx->preallocated_subgroup_memory, .width = IntTy32});
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* lower_physical_ptrs(CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* new_decls_list = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    const Type* stack_base_element = int32_type(dst_arena);
    const Type* stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = NULL,
    });

    // TODO add a @Synthetic annotation to tag those
    Nodes annotations = nodes(dst_arena, 0, NULL);

    const Type* wrapped_type = record_type(dst_arena, (RecordType) {
        .members = nodes(dst_arena, 1, (const Node* []) { stack_arr_type }),
        .special = DecorateBlock,
        .names = strings(dst_arena, 0, NULL)
    });

    Node* physical_private_buffer = global_var(dst_arena, annotations, wrapped_type, "physical_private_buffer", AsGlobalLogical);
    Node* physical_subgroup_buffer = global_var(dst_arena, annotations, wrapped_type, "physical_subgroup_buffer", AsGlobalLogical);

    append_list(const Node*, new_decls_list, physical_private_buffer);
    append_list(const Node*, new_decls_list, physical_subgroup_buffer);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) process_node,
            .processed = done,
        },

        .config = config,
        .src_program = src_program,

        .preallocated_private_memory = 0,
        .preallocated_subgroup_memory = 0,

        .physical_private_buffer = physical_private_buffer,
        .physical_subgroup_buffer = physical_subgroup_buffer,

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

    update_base_stack_ptrs(&ctx);

    destroy_list(new_decls_list);

    destroy_dict(done);
    return rewritten;
}
