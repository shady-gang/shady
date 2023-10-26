#include "rewrite.h"

#include "log.h"
#include "ir_private.h"
#include "portability.h"
#include "type.h"

#include "dict.h"

#include <assert.h>

#pragma GCC diagnostic error "-Wswitch-enum"

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Rewriter create_rewriter(Module* src, Module* dst, RewriteFn fn) {
    return (Rewriter) {
        .src_arena = src->arena,
        .dst_arena = dst->arena,
        .src_module = src,
        .dst_module = dst,
        .rewrite_fn = fn,
        .rewrite_field_type = {
            .rewrite_type = fn,
            .rewrite_value = fn,
            .rewrite_instruction = fn,
            .rewrite_terminator = fn,
            .rewrite_decl = fn,
            .rewrite_anon_lambda = fn,
            .rewrite_basic_block = fn,
            .rewrite_annotation = fn,
        },
        .config = {
            .search_map = true,
            //.write_map = true,
            .rebind_let = dst->arena->config.check_types,
            .fold_quote = true,
        },
        .map = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .decls_map = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
    };
}

void destroy_rewriter(Rewriter* r) {
    assert(r->map);
    destroy_dict(r->map);
    destroy_dict(r->decls_map);
}

const Node* rewrite_node_with_fn(Rewriter* rewriter, const Node* node, RewriteFn fn) {
    assert(rewriter->rewrite_fn);
    if (!node)
        return NULL;
    const Node* found = NULL;
    if (rewriter->config.search_map) {
        found = search_processed(rewriter, node);
    }
    if (found)
        return found;

    const Node* rewritten = fn(rewriter, node);
    if (is_declaration(node))
        return rewritten;
    if (rewriter->config.write_map) {
        register_processed(rewriter, node, rewritten);
    }
    return rewritten;
}

const Node* rewrite_node(Rewriter* rewriter, const Node* node) {
    return rewrite_node_with_fn(rewriter, node, rewriter->rewrite_fn);
}

Nodes rewrite_nodes(Rewriter* rewriter, Nodes old_nodes) {
    size_t count = old_nodes.count;
    LARRAY(const Node*, arr, count);
    for (size_t i = 0; i < count; i++)
        arr[i] = rewrite_node(rewriter, old_nodes.nodes[i]);
    return nodes(rewriter->dst_arena, count, arr);
}

Nodes rewrite_nodes_with_fn(Rewriter* rewriter, Nodes values, RewriteFn fn) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = rewrite_node_with_fn(rewriter, values.nodes[i], fn);
    return nodes(rewriter->dst_arena, values.count, arr);
}

const Node* search_processed(const Rewriter* ctx, const Node* old) {
    struct Dict* map = is_declaration(old) ? ctx->decls_map : ctx->map;
    assert(map && "this rewriter has no processed cache");
    const Node** found = find_value_dict(const Node*, const Node*, map, old);
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
        log_node(ERROR, old);
        error_print(" with ");
        log_node(ERROR, new);
        error_print(" but there was already ");
        log_node(ERROR, found);
        error_print("\n");
        error("The same node got processed twice !");
    }
#endif
    struct Dict* map = is_declaration(old) ? ctx->decls_map : ctx->map;
    assert(map && "this rewriter has no processed cache");
    bool r = insert_dict_and_get_result(const Node*, const Node*, map, old, new);
    assert(r);
}

void register_processed_list(Rewriter* rewriter, Nodes old, Nodes new) {
    assert(old.count == new.count);
    for (size_t i = 0; i < old.count; i++)
        register_processed(rewriter, old.nodes[i], new.nodes[i]);
}

void clear_processed_non_decls(Rewriter* rewriter) {
    clear_dict(rewriter->map);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

#pragma GCC diagnostic error "-Wswitch"

#define rewrite_type rewriter->rewrite_field_type.rewrite_type
#define rewrite_value rewriter->rewrite_field_type.rewrite_value
#define rewrite_instruction rewriter->rewrite_field_type.rewrite_instruction
#define rewrite_terminator rewriter->rewrite_field_type.rewrite_terminator
#define rewrite_decl rewriter->rewrite_field_type.rewrite_decl
#define rewrite_anon_lambda rewriter->rewrite_field_type.rewrite_anon_lambda
#define rewrite_basic_block rewriter->rewrite_field_type.rewrite_basic_block
#define rewrite_annotation rewriter->rewrite_field_type.rewrite_annotation

void rewrite_module(Rewriter* rewriter) {
    Nodes old_decls = get_module_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        if (old_decls.nodes[i]->tag == NominalType_TAG) continue;
        rewrite_node_with_fn(rewriter, old_decls.nodes[i], rewrite_decl);
    }
}

const Node* recreate_variable(Rewriter* rewriter, const Node* old) {
    assert(old->tag == Variable_TAG);
    return var(rewriter->dst_arena, rewrite_node_with_fn(rewriter, old->payload.var.type, rewrite_type), old->payload.var.name);
}

Nodes recreate_variables(Rewriter* rewriter, Nodes old) {
    LARRAY(const Node*, nvars, old.count);
    for (size_t i = 0; i < old.count; i++) {
        if (rewriter->config.process_variables)
            nvars[i] = rewrite_node(rewriter, old.nodes[i]);
        else
            nvars[i] = recreate_variable(rewriter, old.nodes[i]);
        assert(nvars[i]->tag == Variable_TAG);
    }
    return nodes(rewriter->dst_arena, old.count, nvars);
}

Node* recreate_decl_header_identity(Rewriter* rewriter, const Node* old) {
    Node* new = NULL;
    switch (is_declaration(old)) {
        case GlobalVariable_TAG: {
            Nodes new_annotations = rewrite_nodes_with_fn(rewriter, old->payload.global_variable.annotations, rewrite_annotation);
            const Node* ntype = rewrite_node_with_fn(rewriter, old->payload.global_variable.type, rewrite_type);
            new = global_var(rewriter->dst_module,
                             new_annotations,
                             ntype,
                             old->payload.global_variable.name,
                             old->payload.global_variable.address_space);
            break;
        }
        case Constant_TAG: {
            Nodes new_annotations = rewrite_nodes_with_fn(rewriter, old->payload.constant.annotations, rewrite_annotation);
            const Node* ntype = rewrite_node_with_fn(rewriter, old->payload.constant.type_hint, rewrite_type);
            new = constant(rewriter->dst_module,
                           new_annotations,
                           ntype,
                           old->payload.constant.name);
            break;
        }
        case Function_TAG: {
            Nodes new_annotations = rewrite_nodes_with_fn(rewriter, old->payload.fun.annotations, rewrite_annotation);
            Nodes new_params = recreate_variables(rewriter, old->payload.fun.params);
            Nodes nyield_types = rewrite_nodes_with_fn(rewriter, old->payload.fun.return_types, rewrite_type);
            new = function(rewriter->dst_module, new_params, old->payload.fun.name, new_annotations, nyield_types);
            assert(new && new->tag == Function_TAG);
            register_processed_list(rewriter, old->payload.fun.params, new->payload.fun.params);
            break;
        }
        case NominalType_TAG: {
            Nodes new_annotations = rewrite_nodes_with_fn(rewriter, old->payload.nom_type.annotations, rewrite_annotation);
            new = nominal_type(rewriter->dst_module, new_annotations, old->payload.nom_type.name);
            break;
        }
        case NotADecl: error("not a decl");
    }
    assert(new);
    register_processed(rewriter, old, new);
    return new;
}

void recreate_decl_body_identity(Rewriter* rewriter, const Node* old, Node* new) {
    assert(is_declaration(new));
    switch (is_declaration(old)) {
        case GlobalVariable_TAG: {
            new->payload.global_variable.init = rewrite_node_with_fn(rewriter, old->payload.global_variable.init, rewrite_value);
            break;
        }
        case Constant_TAG: {
            new->payload.constant.value = rewrite_node_with_fn(rewriter, old->payload.constant.value, rewrite_value);
            // TODO check type now ?
            break;
        }
        case Function_TAG: {
            assert(new->payload.fun.body == NULL);
            new->payload.fun.body = rewrite_node_with_fn(rewriter, old->payload.fun.body, rewrite_terminator);
            break;
        }
        case NominalType_TAG: {
            new->payload.nom_type.body = rewrite_node_with_fn(rewriter, old->payload.nom_type.body, rewrite_type);
            break;
        }
        case NotADecl: error("not a decl");
    }
}

static const Node* rebind_results(Rewriter* rewriter, const Node* ninstruction, const Node* olam) {
    assert(olam->tag == AnonLambda_TAG);
    Nodes oparams = olam->payload.anon_lam.params;
    Nodes ntypes = unwrap_multiple_yield_types(rewriter->dst_arena, ninstruction->type);
    assert(ntypes.count == oparams.count);
    LARRAY(const Node*, new_params, oparams.count);
    for (size_t i = 0; i < oparams.count; i++) {
        new_params[i] = var(rewriter->dst_arena, ntypes.nodes[i], oparams.nodes[i]->payload.var.name);
        register_processed(rewriter, oparams.nodes[i], new_params[i]);
    }
    const Node* nbody = rewrite_node(rewriter, olam->payload.anon_lam.body);
    const Node* tail = lambda(rewriter->dst_arena, nodes(rewriter->dst_arena, oparams.count, new_params), nbody);
    return tail;
}

const Node* recreate_node_identity(Rewriter* rewriter, const Node* node) {
    if (node == NULL)
        return NULL;

    assert(node->arena == rewriter->src_arena);

    IrArena* arena = rewriter->dst_arena;
    #define REWRITE_FIELD_SCRATCH(t, n)
    #define REWRITE_FIELD_POD(t, n) .n = old_payload.n,
    #define REWRITE_FIELD_TYPE(t, n) .n = rewrite_node_with_fn(rewriter, old_payload.n, rewrite_type),
    #define REWRITE_FIELD_TYPES(t, n) .n = rewrite_nodes_with_fn(rewriter, old_payload.n, rewrite_type),
    #define REWRITE_FIELD_VALUE(t, n) .n = rewrite_node_with_fn(rewriter, old_payload.n, rewrite_value),
    #define REWRITE_FIELD_VALUES(t, n) .n = rewrite_nodes_with_fn(rewriter, old_payload.n, rewrite_value),
    #define REWRITE_FIELD_INSTRUCTION(t, n) .n = rewrite_node_with_fn(rewriter, old_payload.n, rewrite_instruction),
    #define REWRITE_FIELD_TERMINATOR(t, n) .n = rewrite_node_with_fn(rewriter, old_payload.n, rewrite_terminator),
    #define REWRITE_FIELD_TERMINATORS(t, n) .n = rewrite_nodes_with_fn(rewriter, old_payload.n, rewrite_terminator),
    #define REWRITE_FIELD_DECL(t, n) .n = rewrite_node_with_fn(rewriter, old_payload.n, rewrite_decl),
    #define REWRITE_FIELD_ANON_LAMBDA(t, n) .n = rewrite_node_with_fn(rewriter, old_payload.n, rewrite_anon_lambda),
    #define REWRITE_FIELD_ANON_LAMBDAS(t, n) .n = rewrite_nodes_with_fn(rewriter, old_payload.n, rewrite_anon_lambda),
    #define REWRITE_FIELD_BASIC_BLOCK(t, n) .n = rewrite_node_with_fn(rewriter, old_payload.n, rewrite_basic_block),
    #define REWRITE_FIELD_BASIC_BLOCKS(t, n) .n = rewrite_nodes_with_fn(rewriter, old_payload.n, rewrite_basic_block),
    #define REWRITE_FIELD_STRING(t, n) .n = string(arena, old_payload.n),
    #define REWRITE_FIELD_STRINGS(t, n) .n = import_strings(arena, old_payload.n),
    #define REWRITE_FIELD_ANNOTATIONS(t, n) .n = rewrite_nodes_with_fn(rewriter, old_payload.n, rewrite_annotation),
    switch (node->tag) {
        case InvalidNode_TAG:   assert(false);
        #define REWRITE_FIELD(hash, ft, t, n) REWRITE_FIELD_##ft(t, n)
        #define REWRITE_NODE_0_0(StructName, short_name)
        #define REWRITE_NODE_0_1(StructName, short_name) case StructName##_TAG: return short_name(arena);
        #define REWRITE_NODE_1_0(StructName, short_name)
        #define REWRITE_NODE_1_1(StructName, short_name) case StructName##_TAG: { StructName old_payload = node->payload.short_name; return short_name(arena, (StructName) { StructName##_Fields(REWRITE_FIELD) }); }
        #define REWRITE_NODE(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) REWRITE_NODE_##has_payload##_##autogen_ctor(StructName, short_name)
        NODES(REWRITE_NODE)
        case Function_TAG:
        case Constant_TAG:
        case GlobalVariable_TAG:
        case NominalType_TAG: {
            Node* new = recreate_decl_header_identity(rewriter, node);
            recreate_decl_body_identity(rewriter, node, new);
            return new;
        }
        case Variable_TAG: error("variables should be recreated as part of decl handling");
        case Let_TAG: {
            const Node* instruction = rewrite_node_with_fn(rewriter, node->payload.let.instruction, rewrite_instruction);
            if (arena->config.allow_fold && rewriter->config.fold_quote && instruction->tag == PrimOp_TAG && instruction->payload.prim_op.op == quote_op) {
                Nodes old_params = node->payload.let.tail->payload.anon_lam.params;
                Nodes new_args = instruction->payload.prim_op.operands;
                assert(old_params.count == new_args.count);
                register_processed_list(rewriter, old_params, new_args);
                return rewrite_node(rewriter, node->payload.let.tail->payload.anon_lam.body);
            }
            const Node* tail;
            if (rewriter->config.rebind_let)
                tail = rebind_results(rewriter, instruction, node->payload.let.tail);
            else
                tail = rewrite_node_with_fn(rewriter, node->payload.let.tail, rewrite_anon_lambda);
            return let(arena, instruction, tail);
        }
        case LetMut_TAG: error("De-sugar this by hand")
        case AnonLambda_TAG: {
            Nodes params = recreate_variables(rewriter, node->payload.anon_lam.params);
            register_processed_list(rewriter, node->payload.anon_lam.params, params);
            const Node* nterminator = rewrite_node_with_fn(rewriter, node->payload.anon_lam.body, rewrite_terminator);
            const Node* nlam = lambda(rewriter->dst_arena, params, nterminator);
            // register_processed(rewriter, node, nlam);
            return nlam;
        }
        case BasicBlock_TAG: {
            Nodes params = recreate_variables(rewriter, node->payload.basic_block.params);
            register_processed_list(rewriter, node->payload.basic_block.params, params);
            const Node* fn = rewrite_node_with_fn(rewriter, node->payload.basic_block.fn, rewrite_decl);
            Node* bb = basic_block(arena, (Node*) fn, params, node->payload.basic_block.name);
            register_processed(rewriter, node, bb);
            const Node* nterminator = rewrite_node_with_fn(rewriter, node->payload.basic_block.body, rewrite_terminator);
            bb->payload.basic_block.body = nterminator;
            return bb;
        }
    }
    assert(false);
}
