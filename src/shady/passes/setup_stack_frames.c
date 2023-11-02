#include "passes.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "util.h"

#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const CompilerConfig* config;
    const Node* entry_base_stack_ptr;
    const Node* entry_stack_offset;
} Context;

typedef struct {
    Visitor visitor;
    Context* context;
    BodyBuilder* bb;
    Node* nom_t;
    struct List* members;
} VContext;

static void search_operand_for_alloca(VContext* vctx, const Node* node) {
    IrArena* a = vctx->context->rewriter.dst_arena;
    AddressSpace as;

    if (node->tag == PrimOp_TAG) {
        switch (node->payload.prim_op.op) {
            case alloca_op: as = AsPrivatePhysical; break;
            case alloca_subgroup_op: as = AsSubgroupPhysical; break;
            default: goto not_alloca;
        }

        const Type* element_type = rewrite_node(&vctx->context->rewriter, node->payload.prim_op.type_arguments.nodes[0]);
        assert(is_data_type(element_type));
        const Node* slot_offset = gen_primop_e(vctx->bb, offset_of_op, singleton(type_decl_ref_helper(a, vctx->nom_t)), singleton(int32_literal(a, entries_count_list(vctx->members))));
        append_list(const Type*, vctx->members, element_type);

        const Node* slot = first(bind_instruction_named(vctx->bb, prim_op(a, (PrimOp) {
            .op = lea_op,
            .operands = mk_nodes(a, vctx->context->entry_base_stack_ptr, slot_offset) }), (String []) {format_string_arena(a->arena, "stack_slot_%d", entries_count_list(vctx->members)) }));

        const Node* ptr_t = ptr_type(a, (PtrType) { .pointed_type = element_type, .address_space = as });
        slot = gen_reinterpret_cast(vctx->bb, ptr_t, slot);

        register_processed(&vctx->context->rewriter, node, quote_helper(a, singleton(slot)));
        return;
    }

    not_alloca:
    visit_node_operands(&vctx->visitor, IGNORE_ABSTRACTIONS_MASK, node);
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    switch (node->tag) {
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            Context ctx2 = *ctx;
            ctx2.disable_lowering = lookup_annotation_with_string_payload(node, "DisablePass", "setup_stack_frames");

            BodyBuilder* bb = begin_body(a);
            if (!ctx2.disable_lowering) {
                Node* nom_t = nominal_type(m, empty(a), format_string_arena(a->arena, "%s_stack_frame", get_abstraction_name(node)));
                ctx2.entry_stack_offset = first(bind_instruction_named(bb, prim_op(a, (PrimOp) { .op = get_stack_pointer_op } ), (String []) {format_string_arena(a->arena, "saved_stack_ptr_entering_%s", get_abstraction_name(fun)) }));
                ctx2.entry_base_stack_ptr = gen_primop_ce(bb, get_stack_base_op, 0, NULL);
                VContext vctx = {
                    .visitor = {
                        .visit_node_fn = (VisitNodeFn) search_operand_for_alloca,
                    },
                    .context = &ctx2,
                    .bb = bb,
                    .nom_t = nom_t,
                    .members = new_list(const Node*),
                };
                if (node->payload.fun.body) {
                    search_operand_for_alloca(&vctx, node->payload.fun.body);
                    visit_function_rpo(&vctx.visitor, node);
                }
                vctx.nom_t->payload.nom_type.body = record_type(a, (RecordType) {
                    .members = nodes(a, entries_count_list(vctx.members), read_list(const Node*, vctx.members)),
                    .names = strings(a, 0, NULL),
                    .special = 0
                });
                destroy_list(vctx.members);

                const Node* frame_size = gen_primop_e(bb, size_of_op, singleton(type_decl_ref_helper(a, nom_t)), empty(a));
                frame_size = convert_int_extend_according_to_src_t(bb, get_unqualified_type(ctx2.entry_stack_offset->type), frame_size);
                const Node* updated_stack_ptr = gen_primop_e(bb, add_op, empty(a), mk_nodes(a, ctx2.entry_stack_offset, frame_size));
                gen_primop(bb, set_stack_pointer_op, empty(a), singleton(updated_stack_ptr));
            }
            if (node->payload.fun.body)
                fun->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
            else
                cancel_body(bb);
            return fun;
        }
        case Return_TAG: {
            BodyBuilder* bb = begin_body(a);
            if (!ctx->disable_lowering) {
                assert(ctx->entry_stack_offset);
                // Restore SP before calling exit
                bind_instruction(bb, prim_op(a, (PrimOp) {
                    .op = set_stack_pointer_op,
                    .operands = nodes(a, 1, (const Node* []) {ctx->entry_stack_offset })
                }));
            }
            return finish_body(bb, recreate_node_identity(&ctx->rewriter, node));
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

Module* setup_stack_frames(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
