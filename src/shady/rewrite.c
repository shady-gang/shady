#include "shady/rewrite.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

#define bool int

KeyHash shd_hash_node(const Node** pnode);
bool shd_compare_node(const Node** pa, const Node** pb);

typedef enum {
    MASK
} RewriteRuleType;

typedef struct OpRewriteResultRule_ {
    RewriteRuleType type;
    struct OpRewriteResultRule_* next;
    const Node* new;
} OpRewriteResultRule;

typedef struct {
    OpRewriteResultRule base;
    NodeClass mask;
} OpRewriteResultMaskRule;

struct OpRewriteResult_ {
    size_t uuid;
    Rewriter* r;
    OpRewriteResultRule* first_rule;
    const Node* defaultResult;
    int empty;
};

OpRewriteResult* shd_new_rewrite_result_none(Rewriter* r) {
    OpRewriteResult* result = shd_arena_alloc(r->arena, sizeof(OpRewriteResult));
    *result = (OpRewriteResult) {
        .uuid = (size_t) result,
        .r = r,
        .first_rule = NULL,
        .empty = true,
        .defaultResult = NULL,
    };
    return result;
}

OpRewriteResult* shd_new_rewrite_result(Rewriter* r, const Node* defaultResult) {
    OpRewriteResult* result = shd_new_rewrite_result_none(r);
    result->defaultResult = defaultResult;
    result->empty = false;
    return result;
}

void add_rule(OpRewriteResult* result, OpRewriteResultRule* rule) {
    OpRewriteResultRule** last_rule = &result->first_rule;
    while (*last_rule) {
        last_rule = &(*last_rule)->next;
    }
    *last_rule = rule;
}

void shd_rewrite_result_add_mask_rule(OpRewriteResult* result, NodeClass mask, const Node* new) {
    OpRewriteResultMaskRule* rule = shd_arena_alloc(result->r->arena, sizeof(OpRewriteResultMaskRule));
    *rule = (OpRewriteResultMaskRule) {
        .base = {
            .type = MASK,
            .next = NULL,
            .new = new,
        },
        .mask = mask
    };
    add_rule(result, &rule->base);
}

static void init_dicts(Rewriter* r) {
    r->map = shd_new_dict(const Node*, OpRewriteResult*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node);
}

static Rewriter shd_create_rewriter_base(Module* src, Module* dst) {
    Rewriter r = {
        .src_arena = src->arena,
        .dst_arena = dst->arena,
        .src_module = src,
        .dst_module = dst,
        .parent = NULL,
        .arena = shd_new_arena(),
        .select_rewriter_fn = shd_default_rewriter_selector,
    };
    init_dicts(&r);
    return r;
}

Rewriter shd_create_node_rewriter(Module* src, Module* dst, RewriteNodeFn fn) {
    Rewriter r = shd_create_rewriter_base(src, dst);
    r.rewrite_fn = fn;
    return r;
}

Rewriter shd_create_op_rewriter(Module* src, Module* dst, RewriteOpFn fn) {
    Rewriter r = shd_create_rewriter_base(src, dst);
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
    init_dicts(&r);
    r.arena = shd_new_arena();
    r.parent = parent;
    return r;
}

static bool rule_match(const OpRewriteResultRule* rule, const Node* node, NodeClass use_class) {
    switch (rule->type) {
        case MASK: {
            OpRewriteResultMaskRule* mask_rule = (OpRewriteResultMaskRule*) rule;
            if ((mask_rule->mask & use_class) != 0)
                return true;
            break;
        }
        default: assert(false);
    }
    return false;
}

static const Node* apply_rule(const OpRewriteResult* result, const Node* old, bool* found_something, NodeClass use_class) {
    OpRewriteResultRule* rule = result->first_rule;
    while (rule) {
        if (rule_match(rule, old, use_class)) {
            *found_something = true;
            return rule->new;
        }
        rule = rule->next;
    }
    if (!result->empty)
        *found_something = true;
    return result->defaultResult;
}

static const OpRewriteResult* search_in_map(struct Dict* map, const Node* key) {
    OpRewriteResult** found = shd_dict_find_value(const Node*, OpRewriteResult*, map, key);
    if (!found) return NULL;
    return *found;
}

static const Node* shd_search_processed_canary(const Rewriter* r, const Node* old, bool* found_something, NodeClass mask) {
    while (r) {
        assert(r->map && "this rewriter has no processed cache");
        const OpRewriteResult* entry = search_in_map(r->map, old);
        if (entry) {
            const Node* result = apply_rule(entry, old, found_something, mask);
            if (found_something) return result;
        }
        assert(r != r->parent && "impossible loop constructed, somehow");
        r = r->parent;
    }
    return NULL;
}

const Node* shd_search_processed_by_use_class(const Rewriter* r, const Node* old, NodeClass mask) {
    int b;
    return shd_search_processed_canary(r, old, &b, mask);
}

const Node* shd_search_processed(const Rewriter* r, const Node* old) {
    return shd_search_processed_by_use_class(r, old, 0);
}

static bool should_rewrite_at_top_level(const Node* n) {
    switch (n->tag) {
        case Function_TAG:
        case NominalType_TAG:
        case GlobalVariable_TAG:
        case Constant_TAG:
        case BuiltinRef_TAG: return true;
        default: return false;
    }
}

Rewriter* shd_default_rewriter_selector(Rewriter* r, const Node* n) {
    if (should_rewrite_at_top_level(n))
        return shd_get_top_rewriter(r);
    return r;
}

static Nodes rewrite_ops_helper(Rewriter* r, NodeClass class, String op_name, Nodes old);

const Node* shd_rewrite_node_with_fn(Rewriter* r, const Node* old, RewriteNodeFn fn) {
    if (!old)
        return NULL;
    r = r->select_rewriter_fn(r, old);
    assert(r->rewrite_fn);

    bool found_something = false;
    const Node* found = shd_search_processed_canary(r, old, &found_something, 0);
    if (found_something) return found;
    assert(!shd_search_processed(r, old));

    Node* rewritten = (Node*) fn(r, old);
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
    if (!old)
        return NULL;
    r = r->select_rewriter_fn(r, old);

    assert(r->rewrite_op_fn);
    bool found_something = false;
    const Node* found = shd_search_processed_canary(r, old, &found_something, class);
    if (found_something) return found;

    OpRewriteResult* result = fn(r, class, op_name, old);
    shd_register_processed_result(r, old, result);
    return apply_rule(result, old, &found_something, class);
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

void shd_register_processed_result(Rewriter* r, const Node* old, const OpRewriteResult* result) {
    assert(old->arena == r->src_arena);
    assert(result->defaultResult ? result->defaultResult->arena == r->dst_arena : true);
#ifndef NDEBUG
    // In debug mode, we run this extra check so we can provide nice diagnostics
    const OpRewriteResult* found = search_in_map(r->map, old);
    // result->empty => !result->defaultResult
    assert(!result->defaultResult || !result->empty);
    if (found) {
        // this can happen and is typically harmless
        // ie: when rewriting a jump into a loop, the outer jump cannot be finished until the loop body is rebuilt
        // and therefore the back-edge jump inside the loop will be rebuilt while the outer one isn't done.
        // as long as there is no conflict, this is correct, but this might hide perf hazards if we fail to cache things
        if (found->uuid == result->uuid)
            return;
        if (found->first_rule == result->first_rule && found->defaultResult == result->defaultResult)
            return;
        shd_error_print("Trying to replace ");
        shd_log_node(ERROR, old);
        shd_error_print(" with ");
        //shd_log_node(ERROR, new);
        shd_error_print(" but there was already ");
        // if (*found)
        //     shd_log_node(ERROR, *found);
        // else
        //     shd_log_fmt(ERROR, "NULL");
        shd_error_print("\n");
        shd_error("The same node got processed twice !");
    }
#endif
    bool inserted_ok = shd_dict_insert(const Node*, const OpRewriteResult*, r->map, old, result);
    assert(inserted_ok);
}

void shd_register_processed(Rewriter* r, const Node* old, const Node* new) {
    return shd_register_processed_result(r, old, shd_new_rewrite_result(r, new));
}

void shd_register_processed_list(Rewriter* r, Nodes old, Nodes new) {
    assert(old.count == new.count);
    for (size_t i = 0; i < old.count; i++)
        shd_register_processed(r, old.nodes[i], new.nodes[i]);
}

#pragma GCC diagnostic error "-Wswitch"

#include "rewrite_generated.c"

void shd_rewrite_module(Rewriter* r) {
    assert(r->dst_module != r->src_module);
    Nodes old_decls = shd_module_get_all_exported(r->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        // if (!shd_lookup_annotation(old_decls.nodes[i], "Exported") && !shd_lookup_annotation(old_decls.nodes[i], "EntryPoint") && !shd_lookup_annotation(old_decls.nodes[i], "Internal")) continue;
        const Node* ndecl = rewrite_op_helper(r, 0, "decl", old_decls.nodes[i]);
        if (!ndecl) continue;
        String exported_name = shd_get_exported_name(ndecl);
        if (exported_name) shd_module_add_export(r->dst_module, exported_name, ndecl);
    }
}

const Node* shd_recreate_param(Rewriter* r, const Node* oparam) {
    assert(oparam->tag == Param_TAG);
    const Node* nparam = param_helper(r->dst_arena, rewrite_op_helper(r, NcType, "type", oparam->payload.param.type));
    shd_rewrite_annotations(r, oparam, nparam);
    return nparam;
}

Nodes shd_recreate_params(Rewriter* r, Nodes oparams) {
    LARRAY(const Node*, nparams, oparams.count);
    for (size_t i = 0; i < oparams.count; i++) {
        nparams[i] = shd_recreate_param(r, oparams.nodes[i]);
        assert(nparams[i]->tag == Param_TAG);
    }
    return shd_nodes(r->dst_arena, oparams.count, nparams);
}

Function shd_rewrite_function_head_payload(Rewriter* r, Function old) {
    Function new = old;
    new.body = NULL;
    new.params = shd_recreate_params(r, old.params);
    new.return_types = rewrite_ops_helper(r, NcType, "return_types", old.return_types);
    return new;
}

GlobalVariable shd_rewrite_global_head_payload(Rewriter* r, GlobalVariable old) {
    GlobalVariable new = old;
    new.init = NULL;
    new.type = rewrite_op_helper(r, NcType, "type", old.type);
    return new;
}

Constant shd_rewrite_constant_head_payload(Rewriter* r, Constant old) {
    Constant new = old;
    new.value = NULL;
    new.type_hint = rewrite_op_helper(r, NcType, "type_hint", old.type_hint);
    return new;
}

NominalType shd_rewrite_nominal_type_head_payload(Rewriter* r, NominalType old) {
    NominalType new = old;
    new.body = NULL;
    return new;
}

BasicBlock shd_rewrite_basic_block_head_payload(Rewriter* r, BasicBlock old) {
    BasicBlock new = old;
    new.body = NULL;
    new.params = shd_recreate_params(r, old.params);
    return new;
}

Node* shd_recreate_node_head_(Rewriter* r, const Node* old) {
    Node* new = NULL;
    switch (is_declaration(old)) {
        case GlobalVariable_TAG: {
            new = shd_global_var(r->dst_module, shd_rewrite_global_head_payload(r, old->payload.global_variable));
            break;
        }
        case Constant_TAG: {
            new = shd_constant(r->dst_module, shd_rewrite_constant_head_payload(r, old->payload.constant));
            break;
        }
        case Function_TAG: {
            new = shd_function(r->dst_module, shd_rewrite_function_head_payload(r, old->payload.fun));
            shd_register_processed_list(r, old->payload.fun.params, new->payload.fun.params);
            break;
        }
        case NominalType_TAG: {
            new = shd_nominal_type(r->dst_module, shd_rewrite_nominal_type_head_payload(r, old->payload.nom_type));
            break;
        }
        default: shd_error("not a decl");
    }
    assert(new);
    return new;
}

Node* shd_recreate_node_head(Rewriter* r, const Node* old) {
    Node* new = shd_recreate_node_head_(r, old);
    shd_register_processed(r, old, new);
    shd_rewrite_annotations(r, old, new);
    return new;
}

void shd_recreate_node_body(Rewriter* r, const Node* old, Node* new) {
    assert(is_declaration(new));
    switch (is_declaration(old)) {
        case GlobalVariable_TAG: {
            new->payload.global_variable.init = rewrite_op_helper(r, NcValue, "init", old->payload.global_variable.init);
            break;
        }
        case Constant_TAG: {
            new->payload.constant.value = rewrite_op_helper(r, NcValue, "value", old->payload.constant.value);
            // TODO check type now ?
            break;
        }
        case Function_TAG: {
            assert(new->payload.fun.body == NULL);
            shd_set_abstraction_body(new, rewrite_op_helper(r, NcTerminator, "body", old->payload.fun.body));
            break;
        }
        case NominalType_TAG: {
            new->payload.nom_type.body = rewrite_op_helper(r, NcType, "body", old->payload.nom_type.body);
            break;
        }
        default: shd_error("not a decl");
    }
}

void shd_rewrite_annotations(Rewriter* r, const Node* old, Node* new) {
    if (old == new)
        return;
    if (old->annotations.count)
        new->annotations = shd_concat_nodes(r->dst_arena, new->annotations, rewrite_ops_helper(r, NcAnnotation, "annotations", old->annotations));
    assert(new->annotations.count < 256);
}

const Node* shd_recreate_node(Rewriter* r, const Node* old) {
    if (old == NULL)
        return NULL;

    assert(old->arena == r->src_arena);
    IrArena* arena = r->dst_arena;
    if (can_be_default_rewritten(old->tag)) {
        Node* new = (Node*) recreate_node_identity_generated(r, old);
        shd_rewrite_annotations(r, old, new);
        return new;
    }

    switch (old->tag) {
        default: assert(false);
        case Function_TAG:
        case Constant_TAG:
        case GlobalVariable_TAG:
        case NominalType_TAG: {
            Node* new = shd_recreate_node_head(r, old);
            shd_recreate_node_body(r, old, new);
            return new;
        }
        case Param_TAG: {
            shd_log_fmt(ERROR, "Can't rewrite: ");
            shd_log_node(ERROR, old);
            shd_log_fmt(ERROR, ", params should be rewritten by the abstraction rewrite logic");
            shd_error_die();
        }
        case BasicBlock_TAG: {
            BasicBlock payload = old->payload.basic_block;
            Node* new = shd_basic_block(arena, shd_rewrite_basic_block_head_payload(r, payload));
            shd_rewrite_annotations(r, old, new);
            shd_register_processed_list(r, payload.params, get_abstraction_params(new));
            shd_register_processed(r, old, new);
            shd_set_abstraction_body(new, rewrite_op_helper(r, NcTerminator, "body", payload.body));
            return new;
        }
    }
    SHADY_UNREACHABLE;
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