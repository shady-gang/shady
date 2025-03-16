#include "shady/pass.h"
#include "shady/ir/annotation.h"
#include "shady/ir/cast.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/ext.h"
#include "shady/ir/type.h"
#include "shady/ir/composite.h"
#include "shady/ir/function.h"
#include "shady/ir/debug.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

#include "spirv/unified1/spirv.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    struct Dict* cache;
} Context;

typedef struct {
    String iset;
    Op opcode;
    Nodes params;
} SubgroupOp;

typedef struct {
    const Type* t;
    SubgroupOp op;
} Key;

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

KeyHash shd_hash_nodes(Nodes* nodes);
bool shd_compare_nodes(Nodes* a, Nodes* b);

KeyHash shd_hash_node(const Node** pnode);
bool shd_compare_node(const Node** pa, const Node** pb);

static KeyHash hash_key(Key* key) {
    return shd_hash_node(&key->t) ^ shd_hash_string(&key->op.iset) ^ shd_hash(&key->op.opcode, sizeof(Op)) & shd_hash_nodes(&key->op.params);
}

static bool compare_key(Key* a, Key* b) {
    if (a == b)
        return true;
    if (!!a != !!b)
        return false;
    return shd_compare_node(&a->t, &b->t) && strcmp(a->op.iset, b->op.iset) == 0 && a->op.opcode == b->op.opcode && shd_compare_nodes(&a->op.params, &b->op.params);
}

static bool is_extended_type(SHADY_UNUSED IrArena* a, const Type* t, bool allow_vectors) {
    switch (t->tag) {
        case Int_TAG: return true;
        // TODO allow 16-bit floats specifically !
        case Float_TAG: return true;
        case PackType_TAG:
            if (allow_vectors)
                return is_extended_type(a, t->payload.pack_type.element_type, false);
            return false;
        default: return false;
    }
}

static bool is_supported_natively(Context* ctx, SubgroupOp op, const Type* element_type) {
    IrArena* a = ctx->rewriter.dst_arena;
    if (element_type->tag == Int_TAG && element_type->payload.int_type.width <= IntTy32) {
        return true;
    } else if (element_type->tag == Float_TAG /* TODO is it */) {
        return true;
    } else if (element_type->tag == Bool_TAG) {
        return true;
    } else if (!ctx->config->lower.emulate_subgroup_ops_extended_types && is_extended_type(a, element_type, true)) {
        return true;
    }

    return false;
}

static const Node* rebuild_op(Context* ctx, BodyBuilder* bb, SubgroupOp, const Node* src, bool);

static const Node* rebuild_op_deconstruct(Context* ctx, BodyBuilder* bb, const Type* t, SubgroupOp op, const Node* param) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* original_t = t;
    t = shd_get_maybe_nominal_type_body(t);
    switch (is_type(t)) {
        case Type_ArrType_TAG:
        case Type_RecordType_TAG: {
            assert(t->payload.record_type.special == 0);
            Nodes element_types = shd_get_composite_type_element_types(t);
            LARRAY(const Node*, elements, element_types.count);
            for (size_t i = 0; i < element_types.count; i++) {
                const Node* e = shd_extract_helper(a, param, shd_singleton(shd_uint32_literal(a, i)));
                elements[i] = rebuild_op(ctx, bb, op, e, false);
            }
            return composite_helper(a, original_t, shd_nodes(a, element_types.count, elements));
        }
        case Type_Int_TAG: {
            if (t->payload.int_type.width == IntTy64) {
                const Node* hi = prim_op_helper(a, rshift_logical_op, shd_empty(a), mk_nodes(a, param, shd_int32_literal(a, 32)));
                hi = shd_bld_convert_int_zero_extend(bb, shd_int32_type(a), hi);
                const Node* lo = shd_bld_convert_int_zero_extend(bb, shd_int32_type(a), param);
                hi = rebuild_op(ctx, bb, op, hi, false);
                lo = rebuild_op(ctx, bb, op, lo, false);
                const Node* it = int_type(a, (Int) { .width = IntTy64, .is_signed = t->payload.int_type.is_signed });
                hi = shd_bld_convert_int_zero_extend(bb, it, hi);
                lo = shd_bld_convert_int_zero_extend(bb, it, lo);
                hi = prim_op_helper(a, lshift_op, shd_empty(a), mk_nodes(a, hi, shd_int32_literal(a, 32)));
                return prim_op_helper(a, or_op, shd_empty(a), mk_nodes(a, lo, hi));
            }
            break;
        }
        case Type_PtrType_TAG: {
            param = shd_bld_bitcast(bb, shd_uint64_type(a), param);
            return shd_bld_bitcast(bb, t, rebuild_op_deconstruct(ctx, bb, shd_uint64_type(a), op, param));
        }
        default: break;
    }
    return rebuild_op(ctx, bb, op, param, true);
}
static const Node* rebuild_op(Context* ctx, BodyBuilder* bb, SubgroupOp op, const Node* src, bool error_if_not_native) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    const Node* src_t = shd_get_unqualified_type(src->type);

    Key key = {
        .t = src_t,
        .op = op,
    };

    if (is_supported_natively(ctx, op, src_t)) {
        if (strcmp("shady.primop", op.iset) == 0)
            return prim_op_helper(a, op.opcode, shd_empty(a), shd_singleton(src));
        if (strcmp("shady.scope_cast", op.iset) == 0)
            return scope_cast_helper(a, shd_resolve_to_int_literal(shd_first(op.params))->value, src);
        return shd_bld_ext_instruction(bb, op.iset, op.opcode, qualified_type_helper(a, ShdScopeSubgroup, src_t), shd_nodes_append(a, op.params, src));
    } else if (error_if_not_native) {
        shd_log_fmt(ERROR, "subgroup_first emulation is not supported for ");
        shd_log_node(ERROR, src_t);
        shd_log_fmt(ERROR, ".\n");
        shd_error_die();
    }

    // if (shd_resolve_to_int_literal(scope)->value != SpvScopeSubgroup)
    //     shd_error("TODO")

    if (shd_bld_mem(bb) == NULL) {
        return rebuild_op_deconstruct(ctx, bb, src_t, op, src);
    }

    Node* fn = NULL;
    Node** found = shd_dict_find_value(Key, Node*, ctx->cache, key);
    if (found)
        fn = *found;
    else {
        const Node* src_param = param_helper(a, qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, src_t));
        shd_set_debug_name(src_param, "src");
        fn = function_helper(m, shd_singleton(src_param), shd_singleton(qualified_type_helper(a, ShdScopeSubgroup, src_t)));
        shd_set_debug_name(fn, shd_fmt_string_irarena(a, "%s_%d_%s", op.iset, op.opcode, shd_get_type_name(a, src_t)));
        shd_add_annotation_named(fn, "Leaf");
        shd_add_annotation_named(fn, "Generated");
        shd_dict_insert(Key, Node*, ctx->cache, key, fn);

        BodyBuilder* fn_bb = shd_bld_begin(a, shd_get_abstraction_mem(fn));
        const Node* result = rebuild_op_deconstruct(ctx, fn_bb, src_t, op, src_param);
        shd_set_abstraction_body(fn, shd_bld_finish(fn_bb, fn_ret(a, (Return) {
            .args = shd_singleton(result),
            .mem = shd_bld_mem(fn_bb),
        })));
    }

    return shd_first(shd_bld_call(bb, fn, shd_singleton(src)));
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case ExtInstr_TAG: {
            ExtInstr payload = node->payload.ext_instr;
            if (strcmp(payload.set, "spirv.core") == 0 && payload.opcode == SpvOpGroupNonUniformBroadcastFirst) {
                BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
                SubgroupOp op = {
                    .iset = payload.set,
                    .opcode = payload.opcode,
                    .params = shd_singleton(shd_rewrite_node(r, payload.operands.nodes[0])),
                };
                return shd_bld_to_instr_yield_values(bb, shd_singleton(
                    rebuild_op(ctx, bb, op, shd_rewrite_node(r, payload.operands.nodes[1]), false)));
            }
            break;
        }
        case ScopeCast_TAG: {
            ScopeCast payload = node->payload.scope_cast;
            BodyBuilder* bb = shd_bld_begin_pure(a);
            SubgroupOp op = {
                .iset = "shady.scope_cast",
                .opcode = 0,
                .params = shd_singleton(shd_uint64_literal(a, payload.scope)),
            };
            return shd_bld_to_instr_yield_value(bb,rebuild_op(ctx, bb, op, shd_rewrite_node(r, payload.src), false));
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_lower_subgroup_ops(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    assert(!config->lower.emulate_subgroup_ops && "TODO");
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .cache =  shd_new_dict(Key, Node*, (HashFn) hash_key, (CmpFn) compare_key)
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.cache);
    return dst;
}
