#include "shady/rewrite.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

typedef struct MaskedEntry_ {
    NodeClass mask;
    const Node* new;
    struct MaskedEntry_* next;
} MaskedEntry;

static struct Dict* create_dict(bool use_masks) {
    if (use_masks)
        return shd_new_dict(const Node*, MaskedEntry*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);
    return shd_new_dict(const Node*, const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);
}

static Rewriter shd_create_rewriter_base(Module* src, Module* dst, bool use_masks) {
    return (Rewriter) {
        .src_arena = src->arena,
        .dst_arena = dst->arena,
        .src_module = src,
        .dst_module = dst,
        .map = create_dict(use_masks),
        .own_decls = true,
        .decls_map = create_dict(use_masks),
        .parent = NULL,
        .arena = shd_new_arena(),
    };
}

Rewriter shd_create_node_rewriter(Module* src, Module* dst, RewriteNodeFn fn) {
    Rewriter r = shd_create_rewriter_base(src, dst, false);
    r.rewrite_fn = fn;
    return r;
}

Rewriter shd_create_op_rewriter(Module* src, Module* dst, RewriteOpFn fn) {
    Rewriter r = shd_create_rewriter_base(src, dst, true);
    // r.config.write_map = false;
    r.rewrite_op_fn = fn;
    return r;
}

void shd_destroy_rewriter(Rewriter* r) {
    assert(r->map);
    shd_destroy_arena(r->arena);
    shd_destroy_dict(r->map);
    if (r->own_decls)
        shd_destroy_dict(r->decls_map);
}

Rewriter shd_create_importer(Module* src, Module* dst) {
    return shd_create_node_rewriter(src, dst, shd_recreate_node);
}

Rewriter shd_create_children_rewriter(Rewriter* parent) {
    Rewriter r = *parent;
    r.map = create_dict(r.rewrite_op_fn);
    r.arena = shd_new_arena();
    r.parent = parent;
    r.own_decls = false;
    return r;
}

static const Node** search_in_map(struct Dict* map, const Node* key, bool use_mask, NodeClass mask) {
    if (use_mask) {
        MaskedEntry** found = shd_dict_find_value(const Node*, MaskedEntry*, map, key);
        MaskedEntry* entry = found ? *found : NULL;
        while (entry) {
            if (entry->mask & mask)
                return &entry->new;
            entry = entry->next;
        }
        return NULL;
    } else {
        const Node** found = shd_dict_find_value(const Node*, const Node*, map, key);
        return found;
    }
}

static const Node** search_processed_internal(const Rewriter* ctx, const Node* old, NodeClass mask, bool deep) {
    if (is_declaration(old)) {
        const Node** found = search_in_map(ctx->decls_map, old, ctx->rewrite_op_fn, mask);
        return found ? found : NULL;
    }

    while (ctx) {
        assert(ctx->map && "this rewriter has no processed cache");
        const Node** found = search_in_map(ctx->map, old, ctx->rewrite_op_fn, mask);
        if (found)
            return found;
        if (deep)
            ctx = ctx->parent;
        else
            ctx = NULL;
    }
    return NULL;
}

const Node** shd_search_processed(const Rewriter* ctx, const Node* old) {
    return search_processed_internal(ctx, old, ~0, true);
}

const Node** shd_search_processed_mask(const Rewriter* ctx, const Node* old, NodeClass mask) {
    return search_processed_internal(ctx, old, mask, true);
}

const Node* shd_rewrite_node_with_fn(Rewriter* rewriter, const Node* node, RewriteNodeFn fn) {
    assert(rewriter->rewrite_fn);
    if (!node)
        return NULL;
    const Node** found = shd_search_processed(rewriter, node);
    if (found)
        return *found;

    const Node* rewritten = fn(rewriter, node);
    // assert(rewriter->dst_arena == rewritten->arena);
    if (is_declaration(node))
        return rewritten;
    shd_register_processed(rewriter, node, rewritten);
    return rewritten;
}

Nodes shd_rewrite_nodes_with_fn(Rewriter* rewriter, Nodes values, RewriteNodeFn fn) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = shd_rewrite_node_with_fn(rewriter, values.nodes[i], fn);
    return shd_nodes(rewriter->dst_arena, values.count, arr);
}

const Node* shd_rewrite_node(Rewriter* rewriter, const Node* node) {
    assert(rewriter->rewrite_fn);
    return shd_rewrite_node_with_fn(rewriter, node, rewriter->rewrite_fn);
}

Nodes shd_rewrite_nodes(Rewriter* rewriter, Nodes old_nodes) {
    assert(rewriter->rewrite_fn);
    return shd_rewrite_nodes_with_fn(rewriter, old_nodes, rewriter->rewrite_fn);
}

const Node* shd_rewrite_op_with_fn(Rewriter* rewriter, NodeClass class, String op_name, const Node* node, RewriteOpFn fn) {
    assert(rewriter->rewrite_op_fn);
    if (!node)
        return NULL;
    const Node** found = shd_search_processed_mask(rewriter, node, class);
    if (found)
        return *found;

    OpRewriteResult result = fn(rewriter, class, op_name, node);
    if (!result.mask)
        result.mask = shd_get_node_class_from_tag(result.result->tag);
    shd_register_processed_mask(rewriter, node, result.result, result.mask);
    return result.result;
}

Nodes shd_rewrite_ops_with_fn(Rewriter* rewriter, NodeClass class, String op_name, Nodes values, RewriteOpFn fn) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = shd_rewrite_op_with_fn(rewriter, class, op_name, values.nodes[i], fn);
    return shd_nodes(rewriter->dst_arena, values.count, arr);
}

const Node* shd_rewrite_op(Rewriter* rewriter, NodeClass class, String op_name, const Node* node) {
    assert(rewriter->rewrite_op_fn);
    return shd_rewrite_op_with_fn(rewriter, class, op_name, node, rewriter->rewrite_op_fn);
}

Nodes shd_rewrite_ops(Rewriter* rewriter, NodeClass class, String op_name, Nodes old_nodes) {
    assert(rewriter->rewrite_op_fn);
    return shd_rewrite_ops_with_fn(rewriter, class, op_name, old_nodes, rewriter->rewrite_op_fn);
}

static const Node* rewrite_op_helper(Rewriter* rewriter, NodeClass class, String op_name, const Node* node) {
    if (rewriter->rewrite_op_fn)
        return shd_rewrite_op_with_fn(rewriter, class, op_name, node, rewriter->rewrite_op_fn);
    assert(rewriter->rewrite_fn);
    return shd_rewrite_node_with_fn(rewriter, node, rewriter->rewrite_fn);
}

static Nodes rewrite_ops_helper(Rewriter* rewriter, NodeClass class, String op_name, Nodes old_nodes) {
    if (rewriter->rewrite_op_fn)
        return shd_rewrite_ops_with_fn(rewriter, class, op_name, old_nodes, rewriter->rewrite_op_fn);
    assert(rewriter->rewrite_fn);
    return shd_rewrite_nodes_with_fn(rewriter, old_nodes, rewriter->rewrite_fn);
}

void shd_register_processed_mask(Rewriter* ctx, const Node* old, const Node* new, NodeClass mask) {
    assert(old->arena == ctx->src_arena);
    assert(new ? new->arena == ctx->dst_arena : true);
#ifndef NDEBUG
    // In debug mode, we run this extra check so we can provide nice diagnostics
    const Node** found = search_processed_internal(ctx, old, mask, false);
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
    if (ctx->rewrite_op_fn) {
        MaskedEntry** found = shd_dict_find_value(const Node*, MaskedEntry*, map, old);
        MaskedEntry* entry = found ? *found : NULL;
        if (!entry) {
            entry = shd_arena_alloc(ctx->arena, sizeof(MaskedEntry));
            bool r = shd_dict_insert_get_result(const Node*, MaskedEntry*, map, old, entry);
            assert(r);
        } else {
            MaskedEntry* tail = entry;
            while (tail->next)
                tail = tail->next;
            entry = shd_arena_alloc(ctx->arena, sizeof(MaskedEntry));
            tail->next = entry;
        }
        *entry = (MaskedEntry) {
            .mask = mask,
            .new = new,
            .next = NULL
        };
    } else {
        bool r = shd_dict_insert_get_result(const Node*, const Node*, map, old, new);
        assert(r);
    }
}

void shd_register_processed(Rewriter* ctx, const Node* old, const Node* new) {
    return shd_register_processed_mask(ctx, old, new, ~0);
}

void shd_register_processed_list(Rewriter* rewriter, Nodes old, Nodes new) {
    assert(old.count == new.count);
    for (size_t i = 0; i < old.count; i++)
        shd_register_processed(rewriter, old.nodes[i], new.nodes[i]);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

#pragma GCC diagnostic error "-Wswitch"

#include "rewrite_generated.c"

void shd_rewrite_module(Rewriter* rewriter) {
    assert(rewriter->dst_module != rewriter->src_module);
    Nodes old_decls = shd_module_get_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        if (!shd_lookup_annotation(old_decls.nodes[i], "Exported") && !shd_lookup_annotation(old_decls.nodes[i], "EntryPoint") && !shd_lookup_annotation(old_decls.nodes[i], "Internal")) continue;
        rewrite_op_helper(rewriter, NcDeclaration, "decl", old_decls.nodes[i]);
    }
}

const Node* shd_recreate_param(Rewriter* rewriter, const Node* old) {
    assert(old->tag == Param_TAG);
    return param_helper(rewriter->dst_arena, rewrite_op_helper(rewriter, NcType, "type", old->payload.param.type), old->payload.param.name);
}

Nodes shd_recreate_params(Rewriter* rewriter, Nodes oparams) {
    LARRAY(const Node*, nparams, oparams.count);
    for (size_t i = 0; i < oparams.count; i++) {
        nparams[i] = shd_recreate_param(rewriter, oparams.nodes[i]);
        assert(nparams[i]->tag == Param_TAG);
    }
    return shd_nodes(rewriter->dst_arena, oparams.count, nparams);
}

Node* shd_recreate_node_head(Rewriter* rewriter, const Node* old) {
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
            Nodes new_params = shd_recreate_params(rewriter, old->payload.fun.params);
            Nodes nyield_types = rewrite_ops_helper(rewriter, NcType, "return_types", old->payload.fun.return_types);
            new = function(rewriter->dst_module, new_params, old->payload.fun.name, new_annotations, nyield_types);
            assert(new && new->tag == Function_TAG);
            shd_register_processed_list(rewriter, old->payload.fun.params, new->payload.fun.params);
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
    shd_register_processed_mask(rewriter, old, new, shd_get_node_class_from_tag(new->tag));
    return new;
}

void shd_recreate_node_body(Rewriter* rewriter, const Node* old, Node* new) {
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
            shd_set_abstraction_body(new, rewrite_op_helper(rewriter, NcTerminator, "body", old->payload.fun.body));
            break;
        }
        case NominalType_TAG: {
            new->payload.nom_type.body = rewrite_op_helper(rewriter, NcType, "body", old->payload.nom_type.body);
            break;
        }
        case NotADeclaration: shd_error("not a decl");
    }
}

const Node* shd_recreate_node(Rewriter* rewriter, const Node* node) {
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
            Node* new = shd_recreate_node_head(rewriter, node);
            shd_recreate_node_body(rewriter, node, new);
            return new;
        }
        case Param_TAG:
            shd_log_fmt(ERROR, "Can't rewrite: ");
            shd_log_node(ERROR, node);
            shd_log_fmt(ERROR, ", params should be rewritten by the abstraction rewrite logic");
            shd_error_die();
        case BasicBlock_TAG: {
            Nodes params = shd_recreate_params(rewriter, node->payload.basic_block.params);
            shd_register_processed_list(rewriter, node->payload.basic_block.params, params);
            Node* bb = basic_block(arena, params, node->payload.basic_block.name);
            shd_register_processed_mask(rewriter, node, bb, NcBasic_block | NcAbstraction);
            const Node* nterminator = rewrite_op_helper(rewriter, NcTerminator, "body", node->payload.basic_block.body);
            shd_set_abstraction_body(bb, nterminator);
            return bb;
        }
    }
    assert(false);
}

void shd_dump_rewriter_map(Rewriter* r) {
    size_t i = 0;
    const Node* src, *dst;
    while (shd_dict_iter(r->map, &i, &src, &dst)) {
        shd_log_node(ERROR, src);
        shd_log_fmt(ERROR, " -> ");
        shd_log_node(ERROR, dst);
        shd_log_fmt(ERROR, "\n");
    }
}