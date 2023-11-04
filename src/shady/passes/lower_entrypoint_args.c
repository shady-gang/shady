#include "passes.h"

#include "portability.h"
#include "log.h"
#include "util.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"
#include "../transform/memory_layout.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static Node* rewrite_entry_point_fun(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
    Node* fun = function(ctx->rewriter.dst_module, empty(a), node->payload.fun.name, annotations, empty(a));

    register_processed(&ctx->rewriter, node, fun);

    return fun;
}

static const Node* generate_arg_struct_type(Rewriter* rewriter, Nodes params) {
    IrArena* a = rewriter->dst_arena;

    LARRAY(const Node*, types, params.count);
    LARRAY(String, names, params.count);

    for (int i = 0; i < params.count; ++i) {
        const Type* type = rewrite_node(rewriter, params.nodes[i]->type);

        if (!deconstruct_qualified_type(&type))
            error("EntryPoint parameters must be uniform");

        types[i] = type;
        names[i] = params.nodes[i]->payload.var.name;
    }

    return record_type(a, (RecordType) {
        .members = nodes(a, params.count, types),
        .names = strings(a, params.count, names)
    });
}

static const Node* generate_arg_struct(Rewriter* rewriter, const Node* old_entry_point, const Node* new_entry_point) {
    IrArena* a = rewriter->dst_arena;

    Nodes annotations = mk_nodes(a, annotation_value(a, (AnnotationValue) { .name = "EntryPointArgs", .value = fn_addr_helper(a, new_entry_point) }));
    const Node* type = generate_arg_struct_type(rewriter, old_entry_point->payload.fun.params);
    String name = format_string_arena(a->arena, "__%s_args", old_entry_point->payload.fun.name);
    Node* var = global_var(rewriter->dst_module, annotations, type, name, AsExternal);

    return ref_decl_helper(a, var);
}

static const Node* rewrite_body(Context* ctx, const Node* old_entry_point, const Node* arg_struct) {
    IrArena* a = ctx->rewriter.dst_arena;

    BodyBuilder* bb = begin_body(a);

    Nodes params = old_entry_point->payload.fun.params;

    for (int i = 0; i < params.count; ++i) {
        const Node* addr = gen_lea(bb, arg_struct, int32_literal(a, 0), singleton(int32_literal(a, i)));
        const Node* val = gen_load(bb, addr);
        register_processed(&ctx->rewriter, params.nodes[i], val);
    }

    return finish_body(bb, rewrite_node(&ctx->rewriter, old_entry_point->payload.fun.body));
}

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Function_TAG:
            if (lookup_annotation(node, "EntryPoint") && node->payload.fun.params.count > 0) {
                Node* new_entry_point = rewrite_entry_point_fun(ctx, node);
                const Node* arg_struct = generate_arg_struct(&ctx->rewriter, node, new_entry_point);
                new_entry_point->payload.fun.body = rewrite_body(ctx, node, arg_struct);
                return new_entry_point;
            }
            break;
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_entrypoint_args(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
