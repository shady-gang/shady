#include "shady/pass.h"

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
    const Node** found = shd_dict_find_value(const Type*, const Node*, ctx->map, t);
    if (found)
        return *found;

    BodyBuilder* bb = begin_block_pure(a);
    const Node* nul = gen_reinterpret_cast(bb, t, shd_uint64_literal(a, 0));
    Node* decl = constant(ctx->rewriter.dst_module, shd_singleton(annotation(a, (Annotation) {
        .name = "Generated",
    })), t, shd_fmt_string_irarena(a, "nullptr_%s", name_type_safe(a, t)));
    decl->payload.constant.value = yield_values_and_wrap_in_compound_instruction(bb, shd_singleton(nul));
    const Node* ref = ref_decl_helper(a, decl);
    shd_dict_insert(const Type*, const Node*, ctx->map, t, ref);
    return ref;
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case NullPtr_TAG: {
            const Type* t = shd_rewrite_node(r, node->payload.null_ptr.ptr_type);
            assert(t->tag == PtrType_TAG);
            return make_nullptr(ctx, t);
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

Module* lower_nullptr(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(get_module_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .map = shd_new_dict(const Node*, Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.map);
    return dst;
}
