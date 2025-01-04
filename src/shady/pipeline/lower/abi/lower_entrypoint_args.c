#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/function.h"
#include "shady/ir/debug.h"
#include "shady/ir/decl.h"
#include "shady/ir/annotation.h"
#include "shady/ir/mem.h"

#include "portability.h"
#include "log.h"
#include "util.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static Node* rewrite_entry_point_fun(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    Nodes annotations = shd_rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
    Node* fun = function(ctx->rewriter.dst_module, shd_empty(a), node->payload.fun.name, annotations, shd_empty(a));

    shd_register_processed(&ctx->rewriter, node, fun);

    return fun;
}

static const Node* generate_arg_struct_type(Rewriter* rewriter, Nodes params) {
    IrArena* a = rewriter->dst_arena;

    LARRAY(const Node*, types, params.count);
    LARRAY(String, names, params.count);

    for (int i = 0; i < params.count; ++i) {
        const Type* type = shd_rewrite_node(rewriter, params.nodes[i]->type);

        if (!shd_deconstruct_qualified_type(&type))
            shd_error("EntryPoint parameters must be uniform");

        types[i] = type;
        names[i] = shd_get_value_name_safe(params.nodes[i]);
    }

    return record_type(a, (RecordType) {
        .members = shd_nodes(a, params.count, types),
        .names = shd_strings(a, params.count, names)
    });
}

static const Node* generate_arg_struct(Rewriter* rewriter, const Node* old_entry_point, const Node* new_entry_point) {
    IrArena* a = rewriter->dst_arena;

    Nodes annotations = mk_nodes(a, annotation_value(a, (AnnotationValue) { .name = "EntryPointArgs", .value = fn_addr_helper(a, new_entry_point) }));
    const Node* type = generate_arg_struct_type(rewriter, old_entry_point->payload.fun.params);
    String name = shd_fmt_string_irarena(a, "__%s_args", old_entry_point->payload.fun.name);
    Node* var = global_variable_helper(rewriter->dst_module, annotations, type, name, AsExternal, false);

    return var;
}

static const Node* rewrite_body(Context* ctx, const Node* old_entry_point, const Node* new, const Node* arg_struct) {
    IrArena* a = ctx->rewriter.dst_arena;

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(new));

    Nodes params = old_entry_point->payload.fun.params;

    for (int i = 0; i < params.count; ++i) {
        const Node* addr = lea_helper(a, arg_struct, shd_int32_literal(a, 0), shd_singleton(shd_int32_literal(a, i)));
        const Node* val = shd_bld_load(bb, addr);
        shd_register_processed(&ctx->rewriter, params.nodes[i], val);
    }

    shd_register_processed(&ctx->rewriter, shd_get_abstraction_mem(old_entry_point), shd_bld_mem(bb));
    return shd_bld_finish(bb, shd_rewrite_node(&ctx->rewriter, old_entry_point->payload.fun.body));
}

static const Node* process(Context* ctx, const Node* node) {
    switch (node->tag) {
        case Function_TAG:
            if (shd_lookup_annotation(node, "EntryPoint") && node->payload.fun.params.count > 0) {
                Node* new_entry_point = rewrite_entry_point_fun(ctx, node);
                const Node* arg_struct = generate_arg_struct(&ctx->rewriter, node, new_entry_point);
                shd_set_abstraction_body(new_entry_point, rewrite_body(ctx, node, new_entry_point, arg_struct));
                return new_entry_point;
            }
            break;
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_lower_entrypoint_args(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
