#include "rewrite.h"

#include "log.h"
#include "ir_private.h"
#include "portability.h"
#include "type.h"

#include "dict.h"

#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

//RewriteNodeFn rewrite_only_substitute;
const Node* rewrite_only_substitute(Rewriter*, const Node*);

Rewriter create_rewriter(Module* src, Module* dst, RewriteNodeFn fn) {
    return (Rewriter) {
        .src_arena = src->arena,
        .dst_arena = dst->arena,
        .src_module = src,
        .dst_module = dst,
        .rewrite_fn = fn,
        .config = {
            .search_map = true,
            //.write_map = true,
            .rebind_let = false,
            .fold_quote = true,
        },
        .map = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .decls_map = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
    };
}

Rewriter create_substituter(IrArena* a) {
    return (Rewriter) {
        .src_arena = a,
        .dst_arena = a,
        .src_module = NULL,
        .dst_module = NULL,
        .rewrite_fn = rewrite_only_substitute,
        .config = {
            .search_map = true,
            //.write_map = true,
            .rebind_let = false,
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

Rewriter create_importer(Module* src, Module* dst) {
    return create_rewriter(src, dst, recreate_node_identity);
}

Module* rebuild_module(Module* src) {
    IrArena* a = get_module_arena(src);
    Module* dst = new_module(a, get_module_name(src));
    Rewriter r = create_importer(src, dst);
    rewrite_module(&r);
    destroy_rewriter(&r);
    return dst;
}

const Node* rewrite_node_with_fn(Rewriter* rewriter, const Node* node, RewriteNodeFn fn) {
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

Nodes rewrite_nodes_with_fn(Rewriter* rewriter, Nodes values, RewriteNodeFn fn) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = rewrite_node_with_fn(rewriter, values.nodes[i], fn);
    return nodes(rewriter->dst_arena, values.count, arr);
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
    const Node* found = NULL;
    if (rewriter->config.search_map) {
        found = search_processed(rewriter, node);
    }
    if (found)
        return found;

    const Node* rewritten = fn(rewriter, class, op_name, node);
    if (is_declaration(node))
        return rewritten;
    if (rewriter->config.write_map) {
        register_processed(rewriter, node, rewritten);
    }
    return rewritten;
}

Nodes rewrite_ops_with_fn(Rewriter* rewriter, NodeClass class, String op_name, Nodes values, RewriteOpFn fn) {
    LARRAY(const Node*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = rewrite_op_with_fn(rewriter, class, op_name, values.nodes[i], fn);
    return nodes(rewriter->dst_arena, values.count, arr);
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

#include "rewrite_generated.c"

void rewrite_module(Rewriter* rewriter) {
    Nodes old_decls = get_module_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        if (old_decls.nodes[i]->tag == NominalType_TAG) continue;
        rewrite_op_helper(rewriter, NcDeclaration, "decl", old_decls.nodes[i]);
    }
}

const Node* recreate_variable(Rewriter* rewriter, const Node* old) {
    assert(old->tag == Variable_TAG);
    return var(rewriter->dst_arena, rewrite_op_helper(rewriter, NcType, "type", old->payload.var.type), old->payload.var.name);
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

Node* clone_bb_head(Rewriter* r, const Node* bb) {
    assert(bb && bb->tag == BasicBlock_TAG);
    Nodes nparams = recreate_variables(r, get_abstraction_params(bb));
    return basic_block(r->dst_arena, (Node*) bb->payload.basic_block.fn, nparams, get_abstraction_name(bb));
}

Node* recreate_decl_header_identity(Rewriter* rewriter, const Node* old) {
    Node* new = NULL;
    assert(rewriter->dst_module && "Cannot recreate decls in a substituter");
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
            Nodes new_params = recreate_variables(rewriter, old->payload.fun.params);
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
        case NotADeclaration: error("not a decl");
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
            new->payload.constant.instruction = rewrite_op_helper(rewriter, NcInstruction, "instruction", old->payload.constant.instruction);
            // TODO check type now ?
            break;
        }
        case Function_TAG: {
            assert(new->payload.fun.body == NULL);
            new->payload.fun.body = rewrite_op_helper(rewriter, NcTerminator, "body", old->payload.fun.body);
            break;
        }
        case NominalType_TAG: {
            new->payload.nom_type.body = rewrite_op_helper(rewriter, NcType, "body", old->payload.nom_type.body);
            break;
        }
        case NotADeclaration: error("not a decl");
    }
}

const Node* rebind_let(Rewriter* rewriter, const Node* ninstruction, const Node* olam) {
    assert(olam->tag == Case_TAG);
    Nodes oparams = olam->payload.case_.params;
    Nodes ntypes = unwrap_multiple_yield_types(rewriter->dst_arena, ninstruction->type);
    assert(ntypes.count == oparams.count);
    LARRAY(const Node*, new_params, oparams.count);
    for (size_t i = 0; i < oparams.count; i++) {
        new_params[i] = var(rewriter->dst_arena, ntypes.nodes[i], oparams.nodes[i]->payload.var.name);
        register_processed(rewriter, oparams.nodes[i], new_params[i]);
    }
    const Node* nbody = rewrite_node(rewriter, olam->payload.case_.body);
    const Node* tail = case_(rewriter->dst_arena, nodes(rewriter->dst_arena, oparams.count, new_params), nbody);
    return tail;
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
        case Variable_TAG: error("variables should be recreated as part of decl handling");
        /*case Let_TAG: {
            const Node* instruction = rewrite_op_helper(rewriter, NcInstruction, "instruction", node->payload.let.instruction);
            if (arena->config.allow_fold && rewriter->config.fold_quote && instruction->tag == PrimOp_TAG && instruction->payload.prim_op.op == quote_op) {
                Nodes old_params = node->payload.let.tail->payload.case_.params;
                Nodes new_args = instruction->payload.prim_op.operands;
                assert(old_params.count == new_args.count);
                register_processed_list(rewriter, old_params, new_args);
                for (size_t i = 0; i < old_params.count; i++) {
                    String old_name = get_value_name(old_params.nodes[i]);
                    if (!old_name) continue;
                    const Node* new_arg = new_args.nodes[i];
                    if (new_arg->tag == Variable_TAG && !get_value_name(new_arg)) {
                        set_variable_name((Node*) new_arg, old_name);
                    }
                }
                return rewrite_op_helper(rewriter, NcTerminator, "body", node->payload.let.tail->payload.case_.body);
            }
            const Node* tail;
            if (rewriter->config.rebind_let)
                tail = rebind_let(rewriter, instruction, node->payload.let.tail);
            else
                tail = rewrite_op_helper(rewriter, NcCase, "tail", node->payload.let.tail);
            return let(arena, instruction, tail);
        }*/
        case LetMut_TAG: error("De-sugar this by hand")
        case Case_TAG: {
            Nodes params = recreate_variables(rewriter, node->payload.case_.params);
            register_processed_list(rewriter, node->payload.case_.params, params);
            const Node* nterminator = rewrite_op_helper(rewriter, NcTerminator, "body", node->payload.case_.body);
            const Node* nlam = case_(rewriter->dst_arena, params, nterminator);
            // register_processed(rewriter, node, nlam);
            return nlam;
        }
        case BasicBlock_TAG: {
            Nodes params = recreate_variables(rewriter, node->payload.basic_block.params);
            register_processed_list(rewriter, node->payload.basic_block.params, params);
            const Node* fn = rewrite_op_helper(rewriter, NcDeclaration, "fn", node->payload.basic_block.fn);
            Node* bb = basic_block(arena, (Node*) fn, params, node->payload.basic_block.name);
            register_processed(rewriter, node, bb);
            const Node* nterminator = rewrite_op_helper(rewriter, NcTerminator, "body", node->payload.basic_block.body);
            bb->payload.basic_block.body = nterminator;
            return bb;
        }
    }
    assert(false);
}

const Node* rewrite_only_substitute(Rewriter* r, const Node* n) {
    if (is_declaration(n))
        return n;
    return recreate_node_identity(r, n);
    /*switch (n->tag) {
        case Variable_TAG: return n;
    }*/
}
