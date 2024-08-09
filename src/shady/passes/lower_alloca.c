#include "pass.h"

#include "../visit.h"
#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "util.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const CompilerConfig* config;
    struct Dict* prepared_offsets;
    const Node* base_stack_addr_on_entry;
    const Node* stack_size_on_entry;
    size_t num_slots;
    const Node* frame_size;

    const Type* stack_ptr_t;
} Context;

typedef struct {
    Visitor visitor;
    Context* context;
    BodyBuilder* bb;
    Node* nom_t;
    size_t num_slots;
    struct List* members;
    struct Dict* prepared_offsets;
} VContext;

typedef struct {
    size_t i;
    const Node* offset;
    const Type* type;
    AddressSpace as;
} StackSlot;

static void search_operand_for_alloca(VContext* vctx, const Node* node) {
    IrArena* a = vctx->context->rewriter.dst_arena;
    switch (node->tag) {
        case StackAlloc_TAG: {
            const Type* element_type = rewrite_node(&vctx->context->rewriter, node->payload.stack_alloc.type);
            assert(is_data_type(element_type));
            const Node* slot_offset = gen_primop_e(vctx->bb, offset_of_op, singleton(type_decl_ref_helper(a, vctx->nom_t)), singleton(int32_literal(a, entries_count_list(vctx->members))));
            append_list(const Type*, vctx->members, element_type);

            StackSlot slot = { vctx->num_slots, slot_offset, element_type, AsPrivate };
            insert_dict(const Node*, StackSlot, vctx->prepared_offsets, node, slot);

            vctx->num_slots++;

            return;
        }
        default: break;
    }

    not_alloca:
    visit_node_operands(&vctx->visitor, IGNORE_ABSTRACTIONS_MASK, node);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    Module* m = r->dst_module;
    switch (node->tag) {
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            if (!node->payload.fun.body)
                return fun;

            Context ctx2 = *ctx;
            ctx2.disable_lowering = lookup_annotation_with_string_payload(node, "DisablePass", "setup_stack_frames") || ctx->config->per_thread_stack_size == 0;
            if (ctx2.disable_lowering) {
                set_abstraction_body(fun, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
                return fun;
            }

            BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(fun));
            ctx2.prepared_offsets = new_dict(const Node*, StackSlot, (HashFn) hash_node, (CmpFn) compare_node);
            ctx2.base_stack_addr_on_entry = gen_get_stack_base_addr(bb);
            ctx2.stack_size_on_entry = gen_get_stack_size(bb);
            set_value_name((Node*) ctx2.stack_size_on_entry, "stack_size_before_alloca");

            Node* nom_t = nominal_type(m, empty(a), format_string_arena(a->arena, "%s_stack_frame", get_abstraction_name(node)));
            VContext vctx = {
                .visitor = {
                    .visit_node_fn = (VisitNodeFn) search_operand_for_alloca,
                },
                .context = &ctx2,
                .bb = bb,
                .nom_t = nom_t,
                .num_slots = 0,
                .members = new_list(const Node*),
                .prepared_offsets = ctx2.prepared_offsets,
            };
            search_operand_for_alloca(&vctx, node->payload.fun.body);
            visit_function_rpo(&vctx.visitor, node);

            vctx.nom_t->payload.nom_type.body = record_type(a, (RecordType) {
                .members = nodes(a, vctx.num_slots, read_list(const Node*, vctx.members)),
                .names = strings(a, 0, NULL),
                .special = 0
            });
            destroy_list(vctx.members);
            ctx2.num_slots = vctx.num_slots;
            ctx2.frame_size = gen_primop_e(bb, size_of_op, singleton(type_decl_ref_helper(a, vctx.nom_t)), empty(a));
            ctx2.frame_size = convert_int_extend_according_to_src_t(bb, ctx->stack_ptr_t, ctx2.frame_size);

            // make sure to use the new mem from then on
            register_processed(r, get_abstraction_mem(node), bb_mem(bb));
            set_abstraction_body(fun, finish_body(bb, rewrite_node(&ctx2.rewriter, get_abstraction_body(node))));

            destroy_dict(ctx2.prepared_offsets);
            return fun;
        }
        case StackAlloc_TAG: {
            if (!ctx->disable_lowering) {
                StackSlot* found_slot = find_value_dict(const Node*, StackSlot, ctx->prepared_offsets, node);
                if (!found_slot) {
                    error_print("lower_alloca: failed to find a stack offset for ");
                    log_node(ERROR, node);
                    error_print(", most likely this means this alloca was not found in the first block of a function.\n");
                    log_module(DEBUG, ctx->config, ctx->rewriter.src_module);
                    error_die();
                }

                BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, node->payload.stack_alloc.mem));
                if (!ctx->stack_size_on_entry) {
                    //String tmp_name = format_string_arena(a->arena, "stack_ptr_before_alloca_%s", get_abstraction_name(fun));
                    assert(false);
                }

                //const Node* lea_instr = prim_op_helper(a, lea_op, empty(a), mk_nodes(a, rewrite_node(&ctx->rewriter, first(node->payload.prim_op.operands)), found_slot->offset));
                const Node* converted_offset = convert_int_extend_according_to_dst_t(bb, ctx->stack_ptr_t, found_slot->offset);
                const Node* lea_instr = lea(a, (Lea) { .ptr = ctx->base_stack_addr_on_entry, .offset = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, ctx->stack_size_on_entry, converted_offset)), .indices = empty(a) });
                const Node* slot = first(bind_instruction_named(bb, lea_instr, (String []) { format_string_arena(a->arena, "stack_slot_%d", found_slot->i) }));
                const Node* ptr_t = ptr_type(a, (PtrType) { .pointed_type = found_slot->type, .address_space = found_slot->as });
                slot = gen_reinterpret_cast(bb, ptr_t, slot);
                //bool last = found_slot->i == ctx->num_slots - 1;
                //if (last) {
                const Node* updated_stack_ptr = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, ctx->stack_size_on_entry, ctx->frame_size));
                gen_set_stack_size(bb, updated_stack_ptr);
                //}

                return yield_values_and_wrap_in_block(bb, singleton(slot));
            }
            break;
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_alloca(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .stack_ptr_t = int_type(a, (Int) { .is_signed = false, .width = IntTy32 }),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
