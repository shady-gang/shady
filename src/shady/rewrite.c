#include "rewrite.h"

#include "log.h"
#include "ir_private.h"
#include "portability.h"
#include "type.h"

#include "dict.h"

#include <assert.h>
#include <string.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Rewriter create_rewriter_base(Module* src, Module* dst) {
    return (Rewriter) {
        .src_arena = src->arena,
        .dst_arena = dst->arena,
        .src_module = src,
        .dst_module = dst,
        .config = {
            .search_map = true,
            .write_map = true,
        },
        .map = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .own_decls = true,
        .decls_map = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .parent = NULL,
    };
}

Rewriter create_node_rewriter(Module* src, Module* dst, RewriteNodeFn fn) {
    Rewriter r = create_rewriter_base(src, dst);
    r.rewrite_fn = fn;
    return r;
}

Rewriter create_op_rewriter(Module* src, Module* dst, RewriteOpFn fn) {
    Rewriter r = create_rewriter_base(src, dst);
    r.config.write_map = false;
    r.rewrite_op_fn = fn;
    return r;
}

void destroy_rewriter(Rewriter* r) {
    assert(r->map);
    shd_destroy_dict(r->map);
    if (r->own_decls)
        shd_destroy_dict(r->decls_map);
}

Rewriter create_importer(Module* src, Module* dst) {
    return create_node_rewriter(src, dst, recreate_node_identity);
}

Rewriter create_children_rewriter(Rewriter* parent) {
    Rewriter r = *parent;
    r.map = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    r.parent = parent;
    r.own_decls = false;
    return r;
}

Rewriter create_decl_rewriter(Rewriter* parent) {
    Rewriter r = *parent;
    r.map = shd_new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    r.own_decls = false;
    return r;
}

static bool should_memoize(const Node* node) {
    if (is_declaration(node))
        return false;
    if (node->tag == BasicBlock_TAG)
        return false;
    return true;
}

const Node* rewrite_node_with_fn(Rewriter* rewriter, const Node* node, RewriteNodeFn fn) {
    assert(rewriter->rewrite_fn);
    if (!node)
        return NULL;
    const Node** found = NULL;
    if (rewriter->config.search_map) {
        found = search_processed(rewriter, node);
    }
    if (found)
        return *found;

    const Node* rewritten = fn(rewriter, node);
    // assert(rewriter->dst_arena == rewritten->arena);
    if (is_declaration(node))
        return rewritten;
    if (rewriter->config.write_map && should_memoize(node)) {
        register_processed(rewriter, node, rewritten);
    }
    return rewritten;
}

Nodes rewrite_nodes_with_fn(Rewriter* rewriter, Nodes values, RewriteNodeFn fn) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = rewrite_node_with_fn(rewriter, values.nodes[i], fn);
    return shd_nodes(rewriter->dst_arena, values.count, arr);
}

const Node* rewrite_node(Rewriter* rewriter, const Node* node) {
    assert(rewriter->rewrite_fn);
    return rewrite_node_with_fn(rewriter, node, rewriter->rewrite_fn);
}

Nodes rewrite_nodes(Rewriter* rewriter, Nodes old_nodes) {
    assert(rewriter->rewrite_fn);
    return rewrite_nodes_with_fn(rewriter, old_nodes, rewriter->rewrite_fn);
}

const Node* rewrite_op_with_fn(Rewriter* rewriter, NodeClass class, String op_name, const Node* node, RewriteOpFn fn) {
    assert(rewriter->rewrite_op_fn);
    if (!node)
        return NULL;
    const Node** found = NULL;
    if (rewriter->config.search_map) {
        found = search_processed(rewriter, node);
    }
    if (found)
        return *found;

    const Node* rewritten = fn(rewriter, class, op_name, node);
    if (is_declaration(node))
        return rewritten;
    if (rewriter->config.write_map && should_memoize(node)) {
        register_processed(rewriter, node, rewritten);
    }
    return rewritten;
}

Nodes rewrite_ops_with_fn(Rewriter* rewriter, NodeClass class, String op_name, Nodes values, RewriteOpFn fn) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = rewrite_op_with_fn(rewriter, class, op_name, values.nodes[i], fn);
    return shd_nodes(rewriter->dst_arena, values.count, arr);
}

const Node* rewrite_op(Rewriter* rewriter, NodeClass class, String op_name, const Node* node) {
    assert(rewriter->rewrite_op_fn);
    return rewrite_op_with_fn(rewriter, class, op_name, node, rewriter->rewrite_op_fn);
}

Nodes rewrite_ops(Rewriter* rewriter, NodeClass class, String op_name, Nodes old_nodes) {
    assert(rewriter->rewrite_op_fn);
    return rewrite_ops_with_fn(rewriter, class, op_name, old_nodes, rewriter->rewrite_op_fn);
}

static const Node* rewrite_op_helper(Rewriter* rewriter, NodeClass class, String op_name, const Node* node) {
    if (rewriter->rewrite_op_fn)
        return rewrite_op_with_fn(rewriter, class, op_name, node, rewriter->rewrite_op_fn);
    assert(rewriter->rewrite_fn);
    return rewrite_node_with_fn(rewriter, node, rewriter->rewrite_fn);
}

static Nodes rewrite_ops_helper(Rewriter* rewriter, NodeClass class, String op_name, Nodes old_nodes) {
    if (rewriter->rewrite_op_fn)
        return rewrite_ops_with_fn(rewriter, class, op_name, old_nodes, rewriter->rewrite_op_fn);
    assert(rewriter->rewrite_fn);
    return rewrite_nodes_with_fn(rewriter, old_nodes, rewriter->rewrite_fn);
}

static const Node** search_processed_(const Rewriter* ctx, const Node* old, bool deep) {
    if (is_declaration(old)) {
        const Node** found = shd_dict_find_value(const Node*, const Node*, ctx->decls_map, old);
        return found ? found : NULL;
    }

    while (ctx) {
        assert(ctx->map && "this rewriter has no processed cache");
        const Node** found = shd_dict_find_value(const Node*, const Node*, ctx->map, old);
        if (found)
            return found;
        if (deep)
            ctx = ctx->parent;
        else
            ctx = NULL;
    }
    return NULL;
}

const Node** search_processed(const Rewriter* ctx, const Node* old) {
    return search_processed_(ctx, old, true);
}

const Node* find_processed(const Rewriter* ctx, const Node* old) {
    const Node** found = search_processed(ctx, old);
    assert(found && "this node was supposed to have been processed before");
    return *found;
}

void register_processed(Rewriter* ctx, const Node* old, const Node* new) {
    assert(old->arena == ctx->src_arena);
    assert(new ? new->arena == ctx->dst_arena : true);
#ifndef NDEBUG
    const Node** found = search_processed_(ctx, old, false);
    if (found) {
        // this can happen and is typically harmless
        // ie: when rewriting a jump into a loop, the outer jump cannot be finished until the loop body is rebuilt
        // and therefore the back-edge jump inside the loop will be rebuilt while the outer one isn't done.
        // as long as there is no conflict, this is correct, but this might hide perf hazards if we fail to cache things
        if (*found == new)
            return;
        shd_error_print("Trying to replace ");
        shd_log_node(ERROR, old);
        shd_error_print(" with ");
        shd_log_node(ERROR, new);
        shd_error_print(" but there was already ");
        if (*found)
            shd_log_node(ERROR, *found);
        else
            shd_log_fmt(ERROR, "NULL");
        shd_error_print("\n");
        shd_error("The same node got processed twice !");
    }
#endif
    struct Dict* map = is_declaration(old) ? ctx->decls_map : ctx->map;
    assert(map && "this rewriter has no processed cache");
    bool r = shd_dict_insert_get_result(const Node*, const Node*, map, old, new);
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

#include "rewrite_generated.c"

void rewrite_module(Rewriter* rewriter) {
    assert(rewriter->dst_module != rewriter->src_module);
    Nodes old_decls = get_module_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        if (!lookup_annotation(old_decls.nodes[i], "Exported")) continue;
        rewrite_op_helper(rewriter, NcDeclaration, "decl", old_decls.nodes[i]);
    }
}

const Node* recreate_param(Rewriter* rewriter, const Node* old) {
    assert(old->tag == Param_TAG);
    return param(rewriter->dst_arena, rewrite_op_helper(rewriter, NcType, "type", old->payload.param.type), old->payload.param.name);
}

Nodes recreate_params(Rewriter* rewriter, Nodes oparams) {
    LARRAY(const Node*, nparams, oparams.count);
    for (size_t i = 0; i < oparams.count; i++) {
        nparams[i] = recreate_param(rewriter, oparams.nodes[i]);
        assert(nparams[i]->tag == Param_TAG);
    }
    return shd_nodes(rewriter->dst_arena, oparams.count, nparams);
}

Node* recreate_decl_header_identity(Rewriter* rewriter, const Node* old) {
    Node* new = NULL;
    switch (is_declaration(old)) {
        case GlobalVariable_TAG: {
            Nodes new_annotations = rewrite_ops_helper(rewriter, NcAnnotation, "annotations", old->payload.global_variable.annotations);
            const Node* ntype = rewrite_op_helper(rewriter, NcType, "type", old->payload.global_variable.type);
            new = global_var(rewriter->dst_module,
                             new_annotations,
                             ntype,
                             old->payload.global_variable.name,
                             old->payload.global_variable.address_space);
            break;
        }
        case Constant_TAG: {
            Nodes new_annotations = rewrite_ops_helper(rewriter, NcAnnotation, "annotations", old->payload.constant.annotations);
            const Node* ntype = rewrite_op_helper(rewriter, NcType, "type_hint", old->payload.constant.type_hint);
            new = constant(rewriter->dst_module,
                           new_annotations,
                           ntype,
                           old->payload.constant.name);
            break;
        }
        case Function_TAG: {
            Nodes new_annotations = rewrite_ops_helper(rewriter, NcAnnotation, "annotations", old->payload.fun.annotations);
            Nodes new_params = recreate_params(rewriter, old->payload.fun.params);
            Nodes nyield_types = rewrite_ops_helper(rewriter, NcType, "return_types", old->payload.fun.return_types);
            new = function(rewriter->dst_module, new_params, old->payload.fun.name, new_annotations, nyield_types);
            assert(new && new->tag == Function_TAG);
            register_processed_list(rewriter, old->payload.fun.params, new->payload.fun.params);
            break;
        }
        case NominalType_TAG: {
            Nodes new_annotations = rewrite_ops_helper(rewriter, NcAnnotation, "annotations", old->payload.nom_type.annotations);
            new = nominal_type(rewriter->dst_module, new_annotations, old->payload.nom_type.name);
            break;
        }
        case NotADeclaration: shd_error("not a decl");
    }
    assert(new);
    register_processed(rewriter, old, new);
    return new;
}

void recreate_decl_body_identity(Rewriter* rewriter, const Node* old, Node* new) {
    assert(is_declaration(new));
    switch (is_declaration(old)) {
        case GlobalVariable_TAG: {
            new->payload.global_variable.init = rewrite_op_helper(rewriter, NcValue, "init", old->payload.global_variable.init);
            break;
        }
        case Constant_TAG: {
            new->payload.constant.value = rewrite_op_helper(rewriter, NcValue, "value", old->payload.constant.value);
            // TODO check type now ?
            break;
        }
        case Function_TAG: {
            assert(new->payload.fun.body == NULL);
            set_abstraction_body(new, rewrite_op_helper(rewriter, NcTerminator, "body", old->payload.fun.body));
            break;
        }
        case NominalType_TAG: {
            new->payload.nom_type.body = rewrite_op_helper(rewriter, NcType, "body", old->payload.nom_type.body);
            break;
        }
        case NotADeclaration: shd_error("not a decl");
    }
}

const Node* recreate_node_identity(Rewriter* rewriter, const Node* node) {
    if (node == NULL)
        return NULL;

    assert(node->arena == rewriter->src_arena);
    IrArena* arena = rewriter->dst_arena;
    if (can_be_default_rewritten(node->tag))
        return recreate_node_identity_generated(rewriter, node);

    switch (node->tag) {
        default:   assert(false);
        case Function_TAG:
        case Constant_TAG:
        case GlobalVariable_TAG:
        case NominalType_TAG: {
            Node* new = recreate_decl_header_identity(rewriter, node);
            recreate_decl_body_identity(rewriter, node, new);
            return new;
        }
        case Param_TAG:
            shd_log_fmt(ERROR, "Can't rewrite: ");
            shd_log_node(ERROR, node);
            shd_log_fmt(ERROR, ", params should be rewritten by the abstraction rewrite logic");
            shd_error_die();
        case BasicBlock_TAG: {
            Nodes params = recreate_params(rewriter, node->payload.basic_block.params);
            register_processed_list(rewriter, node->payload.basic_block.params, params);
            Node* bb = basic_block(arena, params, node->payload.basic_block.name);
            register_processed(rewriter, node, bb);
            const Node* nterminator = rewrite_op_helper(rewriter, NcTerminator, "body", node->payload.basic_block.body);
            set_abstraction_body(bb, nterminator);
            return bb;
        }
    }
    assert(false);
}

void dump_rewriter_map(Rewriter* r) {
    size_t i = 0;
    const Node* src, *dst;
    while (shd_dict_iter(r->map, &i, &src, &dst)) {
        shd_log_node(ERROR, src);
        shd_log_fmt(ERROR, " -> ");
        shd_log_node(ERROR, dst);
        shd_log_fmt(ERROR, "\n");
    }
}