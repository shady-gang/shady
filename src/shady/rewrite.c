#include "rewrite.h"

#include "log.h"
#include "ir_private.h"
#include "portability.h"
#include "type.h"

#include "dict.h"

#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Rewriter create_rewriter(Module* src, Module* dst, RewriteFn fn) {
    return (Rewriter) {
        .src_arena = src->arena,
        .dst_arena = dst->arena,
        .src_module = src,
        .dst_module = dst,
        .rewrite_fn = fn,
        .processed = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
}

Rewriter create_importer(Module* src, Module* dst) {
    return create_rewriter(src, dst, recreate_node_identity);
}

static const Node* recreate_node_substitutions_only(Rewriter* rewriter, const Node* node) {
    assert(rewriter->dst_arena == rewriter->src_arena);
    const Node* found = rewriter->processed ? search_processed(rewriter, node) : NULL;
    if (found)
        return found;

    if (is_declaration(node))
        return node;
    if (node->tag == Variable_TAG)
        return node;
    return recreate_node_identity(rewriter, node);
}

Rewriter create_substituter(Module* module) {
    return create_rewriter(module, module, recreate_node_substitutions_only);
}

void destroy_rewriter(Rewriter* r) {
    assert(r->processed);
    destroy_dict(r->processed);
}

const Node* rewrite_node(Rewriter* rewriter, const Node* node) {
    assert(rewriter->rewrite_fn);
    if (node)
        return rewriter->rewrite_fn(rewriter, node);
    else
        return NULL;
}

Nodes rewrite_nodes(Rewriter* rewriter, Nodes old_nodes) {
    size_t count = old_nodes.count;
    LARRAY(const Node*, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = rewrite_node(rewriter, old_nodes.nodes[i]);
    return nodes(rewriter->dst_arena, count, arr);
}

const Node* search_processed(const Rewriter* ctx, const Node* old) {
    assert(ctx->processed && "this rewriter has no processed cache");
    const Node** found = find_value_dict(const Node*, const Node*, ctx->processed, old);
    return found ? *found : NULL;
}

const Node* find_processed(const Rewriter* ctx, const Node* old) {
    const Node* found = search_processed(ctx, old);
    assert(found && "this node was supposed to have been processed before");
    return found;
}

void register_processed(Rewriter* ctx, const Node* old, const Node* new) {
    assert(old->arena == ctx->src_arena);
    assert(new->arena == ctx->dst_arena);
#ifndef NDEBUG
    const Node* found = search_processed(ctx, old);
    if (found) {
        error_print("Trying to replace ");
        error_node(old);
        error_print(" with ");
        error_node(new);
        error_print(" but there was already ");
        error_node(found);
        error_print("\n");
        error("The same node got processed twice !");
    }
#endif
    assert(ctx->processed && "this rewriter has no processed cache");
    bool r = insert_dict_and_get_result(const Node*, const Node*, ctx->processed, old, new);
    assert(r);
}

void register_processed_list(Rewriter* rewriter, Nodes old, Nodes new) {
    assert(old.count == new.count);
    for (size_t i = 0; i < old.count; i++)
        register_processed(rewriter, old.nodes[i], new.nodes[i]);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

#pragma GCC diagnostic error "-Wswitch"

Nodes rewrite_nodes_generic(Rewriter* rewriter, RewriteFn fn, Nodes values) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = fn(rewriter, values.nodes[i]);
    return nodes(rewriter->dst_arena, values.count, arr);
}

#include "rewrite_default.h"

void rewrite_module(Rewriter* rewriter) {
    Nodes old_decls = get_module_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        rewrite_decl(rewriter, old_decls.nodes[i]);
    }
}

const Node* recreate_variable(Rewriter* rewriter, const Node* old) {
    assert(old->tag == Variable_TAG);
    return var(rewriter->dst_arena, rewrite_type(rewriter, old->payload.var.type), old->payload.var.name);
}

Nodes recreate_variables(Rewriter* rewriter, Nodes old) {
    LARRAY(const Node*, nvars, old.count);
    for (size_t i = 0; i < old.count; i++)
        nvars[i] = recreate_variable(rewriter, old.nodes[i]);
    return nodes(rewriter->dst_arena, old.count, nvars);
}

Node* recreate_decl_header_identity(Rewriter* rewriter, const Node* old) {
    Node* new = NULL;
    switch (old->tag) {
        case GlobalVariable_TAG: {
            new = global_var(rewriter->dst_module,
                             rewrite_nodes_generic(rewriter, rewrite_annotation, old->payload.global_variable.annotations),
                             rewrite_type(rewriter, old->payload.global_variable.type),
                             old->payload.global_variable.name,
                             old->payload.global_variable.address_space);
            break;
        }
        case Constant_TAG: {
            new = constant(rewriter->dst_module,
                           rewrite_nodes_generic(rewriter, rewrite_annotation, old->payload.constant.annotations),
                           rewrite_type(rewriter, old->payload.constant.type_hint),
                           old->payload.constant.name);
            break;
        }
        case Function_TAG: {
            Nodes new_params = recreate_variables(rewriter, old->payload.fun.params);
            new = function(rewriter->dst_module, new_params, old->payload.fun.name, rewrite_nodes(rewriter, old->payload.fun.annotations), rewrite_nodes(rewriter, old->payload.fun.return_types));
            assert(new && new->tag == Function_TAG);
            register_processed_list(rewriter, old->payload.fun.params, new->payload.fun.params);
            break;
        }
        default: error("not a decl");
    }
    assert(new);
    register_processed(rewriter, old, new);
    return new;
}

void recreate_decl_body_identity(Rewriter* rewriter, const Node* old, Node* new) {
    assert(is_declaration(new) && is_declaration(old));
    switch (old->tag) {
        case GlobalVariable_TAG: {
            new->payload.global_variable.init = rewrite_value(rewriter, old->payload.global_variable.init);
            break;
        }
        case Constant_TAG: {
            new->payload.constant.value     = rewrite_value(rewriter, old->payload.constant.value);
            // TODO check type now ?
            break;
        }
        case Function_TAG: {
            assert(new->payload.fun.body == NULL);
            new->payload.fun.body = rewrite_terminator(rewriter, old->payload.fun.body);
            break;
        }
        default: error("not a decl");
    }
}

const Node* recreate_node_identity(Rewriter* rewriter, const Node* node) {
    if (node == NULL)
        return NULL;

    // TODO redundant ?
    const Node* already_done_before = rewriter->processed ? search_processed(rewriter, node) : NULL;
    if (already_done_before)
        return already_done_before;

    IrArena* arena = rewriter->dst_arena;
    #include "rewrite_cases.h"
    assert(false);
}
