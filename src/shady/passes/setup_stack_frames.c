#include "passes.h"

#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"
#include "log.h"
#include "portability.h"

#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"

#include "list.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    CompilerConfig* config;
    const Node* entry_base_stack_ptr;
    const Node* entry_stack_offset;
    size_t total_size;
} Context;

typedef struct {
    Visitor visitor;
    Context* context;
    BodyBuilder* builder;
} VContext;

static void collect_allocas(VContext* vctx, const Node* node) {
    IrArena* arena = vctx->context->rewriter.dst_arena;
    AddressSpace as;

    if (node->tag == PrimOp_TAG) {
        switch (node->payload.prim_op.op) {
            case alloca_op: as = AsPrivatePhysical; break;
            case alloca_subgroup_op: as = AsSubgroupPhysical; break;
            default: goto not_alloca;
        }

        const Type* element_type = rewrite_node(&vctx->context->rewriter, node->payload.prim_op.type_arguments.nodes[0]);
        const Node* element_size = gen_primop_e(vctx->builder, size_of_op, singleton(element_type), empty(arena));

        const Node* slot = first(bind_instruction_named(vctx->builder, prim_op(arena, (PrimOp) {
            .op = lea_op,
            .operands = mk_nodes(arena, vctx->context->entry_base_stack_ptr, element_size) }), (String []) { "stack_slot" }));
        const Node* ptr_t = ptr_type(arena, (PtrType) { .pointed_type = element_type, .address_space = as });
        slot = gen_reinterpret_cast(vctx->builder, ptr_t, slot);

        register_processed(&vctx->context->rewriter, node, quote_single(arena, slot));
        return;
    }

    not_alloca:
    visit_children(&vctx->visitor, node);
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            Context ctx2 = *ctx;
            ctx2.disable_lowering = lookup_annotation_with_string_payload(node, "DisablePass", "setup_stack_frames");

            BodyBuilder* bb = begin_body(arena);
            if (!ctx2.disable_lowering) {
                ctx2.entry_stack_offset = first(bind_instruction_named(bb, prim_op(arena, (PrimOp) { .op = get_stack_pointer_op } ), (String []) { format_string(arena, "saved_stack_ptr_entering_%s", get_abstraction_name(fun)) }));
                ctx2.entry_base_stack_ptr = gen_primop_ce(bb, get_stack_base_op, 0, NULL);
                VContext vctx = {
                    .visitor = {
                        .visit_fn = (VisitFn) collect_allocas,
                        .visit_fn_scope_rpo = true,
                    },
                    .context = &ctx2,
                    .builder = bb,
                };
                if (node->payload.fun.body)
                        visit_children(&vctx.visitor, node->payload.fun.body);
            }
            if (node->payload.fun.body)
                fun->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
            else
                cancel_body(bb);
            return fun;
        }
        case Return_TAG: {
            BodyBuilder* bb = begin_body(arena);
            if (!ctx->disable_lowering) {
                assert(ctx->entry_stack_offset);
                // Restore SP before calling exit
                bind_instruction(bb, prim_op(arena, (PrimOp) {
                    .op = set_stack_pointer_op,
                    .operands = nodes(arena, 1, (const Node* []) { ctx->entry_stack_offset })
                }));
            }
            return finish_body(bb, recreate_node_identity(&ctx->rewriter, node));
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void  setup_stack_frames(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
