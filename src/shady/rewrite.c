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
}

Rewriter shd_create_importer(Module* src, Module* dst) {
    return shd_create_node_rewriter(src, dst, shd_recreate_node);
}

Rewriter shd_create_children_rewriter(Rewriter* parent) {
    Rewriter r = *parent;
    r.map = create_dict(r.rewrite_op_fn);
    r.arena = shd_new_arena();
    r.parent = parent;
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

static const Node** search_processed_internal(const Rewriter* r, const Node* old, NodeClass mask, bool deep) {
    while (r) {
        assert(r->map && "this rewriter has no processed cache");
        const Node** found = search_in_map(r->map, old, r->rewrite_op_fn, mask);
        if (found)
            return found;
        assert(r != r->parent && "impossible loop constructed, somehow");
        if (deep)
            r = r->parent;
        else
            r = NULL;
    }
    return NULL;
}

const Node** shd_search_processed(const Rewriter* r, const Node* old) {
    return search_processed_internal(r, old, ~0, true);
}

const Node** shd_search_processed_mask(const Rewriter* r, const Node* old, NodeClass mask) {
    return search_processed_internal(r, old, mask, true);
}

const Node* shd_rewrite_node_with_fn(Rewriter* r, const Node* old, RewriteNodeFn fn) {
    assert(r->rewrite_fn);
    if (!old)
        return NULL;
    const Node** found = shd_search_processed(r, old);
    if (found)
        return *found;

    const Node* rewritten = fn(r, old);
    shd_register_processed(r, old, rewritten);
    return rewritten;
}

Nodes shd_rewrite_nodes_with_fn(Rewriter* r, Nodes old, RewriteNodeFn fn) {
    LARRAY(const Node*, arr, old.count);
    for (size_t i = 0; i < old.count; i++)
        arr[i] = shd_rewrite_node_with_fn(r, old.nodes[i], fn);
    return shd_nodes(r->dst_arena, old.count, arr);
}

const Node* shd_rewrite_node(Rewriter* r, const Node* old) {
    assert(r->rewrite_fn);
    return shd_rewrite_node_with_fn(r, old, r->rewrite_fn);
}

Nodes shd_rewrite_nodes(Rewriter* r, Nodes old) {
    assert(r->rewrite_fn);
    return shd_rewrite_nodes_with_fn(r, old, r->rewrite_fn);
}

const Node* shd_rewrite_op_with_fn(Rewriter* r, NodeClass class, String op_name, const Node* old, RewriteOpFn fn) {
    assert(r->rewrite_op_fn);
    if (!old)
        return NULL;
    const Node** found = shd_search_processed_mask(r, old, class);
    if (found)
        return *found;

    OpRewriteResult result = fn(r, class, op_name, old);
    if (!result.mask)
        result.mask = shd_get_node_class_from_tag(result.result->tag);
    shd_register_processed_mask(r, old, result.result, result.mask);
    return result.result;
}

Nodes shd_rewrite_ops_with_fn(Rewriter* r, NodeClass class, String op_name, Nodes old, RewriteOpFn fn) {
    LARRAY(const Node*, arr, old.count);
    for (size_t i = 0; i < old.count; i++)
        arr[i] = shd_rewrite_op_with_fn(r, class, op_name, old.nodes[i], fn);
    return shd_nodes(r->dst_arena, old.count, arr);
}

const Node* shd_rewrite_op(Rewriter* r, NodeClass class, String op_name, const Node* old) {
    assert(r->rewrite_op_fn);
    return shd_rewrite_op_with_fn(r, class, op_name, old, r->rewrite_op_fn);
}

Nodes shd_rewrite_ops(Rewriter* r, NodeClass class, String op_name, Nodes old) {
    assert(r->rewrite_op_fn);
    return shd_rewrite_ops_with_fn(r, class, op_name, old, r->rewrite_op_fn);
}

static const Node* rewrite_op_helper(Rewriter* r, NodeClass class, String op_name, const Node* old) {
    if (r->rewrite_op_fn)
        return shd_rewrite_op_with_fn(r, class, op_name, old, r->rewrite_op_fn);
    assert(r->rewrite_fn);
    return shd_rewrite_node_with_fn(r, old, r->rewrite_fn);
}

static Nodes rewrite_ops_helper(Rewriter* r, NodeClass class, String op_name, Nodes old) {
    if (r->rewrite_op_fn)
        return shd_rewrite_ops_with_fn(r, class, op_name, old, r->rewrite_op_fn);
    assert(r->rewrite_fn);
    return shd_rewrite_nodes_with_fn(r, old, r->rewrite_fn);
}

void shd_register_processed_mask(Rewriter* r, const Node* old, const Node* new, NodeClass mask) {
    assert(old->arena == r->src_arena);
    assert(new ? new->arena == r->dst_arena : true);
#ifndef NDEBUG
    // In debug mode, we run this extra check so we can provide nice diagnostics
    const Node** found = search_processed_internal(r, old, mask, false);
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
    struct Dict* map = r->map;
    assert(map && "this rewriter has no processed cache");
    if (r->rewrite_op_fn) {
        MaskedEntry** found = shd_dict_find_value(const Node*, MaskedEntry*, map, old);
        MaskedEntry* entry = found ? *found : NULL;
        if (!entry) {
            entry = shd_arena_alloc(r->arena, sizeof(MaskedEntry));
            bool r = shd_dict_insert_get_result(const Node*, MaskedEntry*, map, old, entry);
            assert(r);
        } else {
            MaskedEntry* tail = entry;
            while (tail->next)
                tail = tail->next;
            entry = shd_arena_alloc(r->arena, sizeof(MaskedEntry));
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

void shd_register_processed(Rewriter* r, const Node* old, const Node* new) {
    return shd_register_processed_mask(r, old, new, ~0);
}

void shd_register_processed_list(Rewriter* r, Nodes old, Nodes new) {
    assert(old.count == new.count);
    for (size_t i = 0; i < old.count; i++)
        shd_register_processed(r, old.nodes[i], new.nodes[i]);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

#pragma GCC diagnostic error "-Wswitch"

#include "rewrite_generated.c"

void shd_rewrite_module(Rewriter* r) {
    assert(r->dst_module != r->src_module);
    Nodes old_decls = shd_module_get_declarations(r->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        if (!shd_lookup_annotation(old_decls.nodes[i], "Exported") && !shd_lookup_annotation(old_decls.nodes[i], "EntryPoint") && !shd_lookup_annotation(old_decls.nodes[i], "Internal")) continue;
        rewrite_op_helper(r, NcDeclaration, "decl", old_decls.nodes[i]);
    }
}

const Node* shd_recreate_param(Rewriter* r, const Node* oparam) {
    assert(oparam->tag == Param_TAG);
    return param_helper(r->dst_arena, rewrite_op_helper(r, NcType, "type", oparam->payload.param.type), oparam->payload.param.name);
}

Nodes shd_recreate_params(Rewriter* r, Nodes oparams) {
    LARRAY(const Node*, nparams, oparams.count);
    for (size_t i = 0; i < oparams.count; i++) {
        nparams[i] = shd_recreate_param(r, oparams.nodes[i]);
        assert(nparams[i]->tag == Param_TAG);
    }
    return shd_nodes(r->dst_arena, oparams.count, nparams);
}

Node* shd_recreate_node_head(Rewriter* r, const Node* old) {
    Node* new = NULL;
    switch (is_declaration(old)) {
        case GlobalVariable_TAG: {
            Nodes new_annotations = rewrite_ops_helper(r, NcAnnotation, "annotations", old->payload.global_variable.annotations);
            const Node* ntype = rewrite_op_helper(r, NcType, "type", old->payload.global_variable.type);
            new = global_var(r->dst_module,
                             new_annotations,
                             ntype,
                             old->payload.global_variable.name,
                             old->payload.global_variable.address_space);
            break;
        }
        case Constant_TAG: {
            Nodes new_annotations = rewrite_ops_helper(r, NcAnnotation, "annotations", old->payload.constant.annotations);
            const Node* ntype = rewrite_op_helper(r, NcType, "type_hint", old->payload.constant.type_hint);
            new = constant(r->dst_module,
                           new_annotations,
                           ntype,
                           old->payload.constant.name);
            break;
        }
        case Function_TAG: {
            Nodes new_annotations = rewrite_ops_helper(r, NcAnnotation, "annotations", old->payload.fun.annotations);
            Nodes new_params = shd_recreate_params(r, old->payload.fun.params);
            Nodes nyield_types = rewrite_ops_helper(r, NcType, "return_types", old->payload.fun.return_types);
            new = function(r->dst_module, new_params, old->payload.fun.name, new_annotations, nyield_types);
            assert(new && new->tag == Function_TAG);
            shd_register_processed_list(r, old->payload.fun.params, new->payload.fun.params);
            break;
        }
        case NominalType_TAG: {
            Nodes new_annotations = rewrite_ops_helper(r, NcAnnotation, "annotations", old->payload.nom_type.annotations);
            new = nominal_type(r->dst_module, new_annotations, old->payload.nom_type.name);
            break;
        }
        case NotADeclaration: shd_error("not a decl");
    }
    assert(new);
    shd_register_processed_mask(shd_get_top_rewriter(r), old, new, shd_get_node_class_from_tag(new->tag));
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

Rewriter* shd_get_top_rewriter(Rewriter* r) {
    while (r->parent)
        r = r->parent;
    return r;
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