#include "pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

typedef struct {
    Rewriter rewriter;
    struct Dict* map;
} Context;

static const Node* make_nullptr(Context* ctx, const Type* t) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Node** found = find_value_dict(const Type*, const Node*, ctx->map, t);
    if (found)
        return *found;

    BodyBuilder* bb = begin_body(a);
    const Node* nul = gen_reinterpret_cast(bb, t, uint64_literal(a, 0));
    Node* decl = constant(ctx->rewriter.dst_module, singleton(annotation(a, (Annotation) {
        .name = "Generated",
    })), t, format_string_interned(a, "nullptr_%s", name_type_safe(a, t)));
    decl->payload.constant.instruction = yield_values_and_wrap_in_block(bb, singleton(nul));
    const Node* ref = ref_decl_helper(a, decl);
    insert_dict(const Type*, const Node*, ctx->map, t, ref);
    return ref;
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case NullPtr_TAG: {
            const Type* t = rewrite_node(r, node->payload.null_ptr.ptr_type);
            assert(t->tag == PtrType_TAG);
            return make_nullptr(ctx, t);
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Module* lower_nullptr(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .map = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.map);
    return dst;
}
