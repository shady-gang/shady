#include "shady/pass.h"
#include "shady/ir/memory_layout.h"

#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

#include <spirv/unified1/spirv.h>

#include <string.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    struct Dict* fns;
} Context;

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

static bool is_supported_natively(Context* ctx, const Type* element_type) {
    IrArena* a = ctx->rewriter.dst_arena;
    if (element_type->tag == Int_TAG && element_type->payload.int_type.width == IntTy32) {
        return true;
    } else if (!ctx->config->lower.emulate_subgroup_ops_extended_types && is_extended_type(a, element_type, true)) {
        return true;
    }

    return false;
}

static const Node* build_subgroup_first(Context* ctx, BodyBuilder* bb, const Node* scope, const Node* src);

static const Node* generate(Context* ctx, BodyBuilder* bb, const Node* scope, const Node* t, const Node* param) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* original_t = t;
    t = get_maybe_nominal_type_body(t);
    switch (is_type(t)) {
        case Type_ArrType_TAG:
        case Type_RecordType_TAG: {
            assert(t->payload.record_type.special == 0);
            Nodes element_types = get_composite_type_element_types(t);
            LARRAY(const Node*, elements, element_types.count);
            for (size_t i = 0; i < element_types.count; i++) {
                const Node* e = gen_extract(bb, param, shd_singleton(shd_uint32_literal(a, i)));
                elements[i] = build_subgroup_first(ctx, bb, scope, e);
            }
            return composite_helper(a, original_t, shd_nodes(a, element_types.count, elements));
        }
        case Type_Int_TAG: {
            if (t->payload.int_type.width == IntTy64) {
                const Node* hi = gen_primop_e(bb, rshift_logical_op, shd_empty(a), mk_nodes(a, param, shd_int32_literal(a, 32)));
                hi = convert_int_zero_extend(bb, shd_int32_type(a), hi);
                const Node* lo = convert_int_zero_extend(bb, shd_int32_type(a), param);
                hi = build_subgroup_first(ctx, bb, scope, hi);
                lo = build_subgroup_first(ctx, bb, scope, lo);
                const Node* it = int_type(a, (Int) { .width = IntTy64, .is_signed = t->payload.int_type.is_signed });
                hi = convert_int_zero_extend(bb, it, hi);
                lo = convert_int_zero_extend(bb, it, lo);
                hi = gen_primop_e(bb, lshift_op, shd_empty(a), mk_nodes(a, hi, shd_int32_literal(a, 32)));
                return gen_primop_e(bb, or_op, shd_empty(a), mk_nodes(a, lo, hi));
            }
            break;
        }
        case Type_PtrType_TAG: {
            param = gen_reinterpret_cast(bb, shd_uint64_type(a), param);
            return gen_reinterpret_cast(bb, t, generate(ctx, bb, scope, shd_uint64_type(a), param));
        }
        default: break;
    }
    return NULL;
}

static void build_fn_body(Context* ctx, Node* fn, const Node* scope, const Node* param, const Type* t) {
    IrArena* a = ctx->rewriter.dst_arena;
    BodyBuilder* bb = begin_body_with_mem(a, get_abstraction_mem(fn));
    const Node* result = generate(ctx, bb, scope, t, param);
    if (result) {
        set_abstraction_body(fn, finish_body(bb, fn_ret(a, (Return) {
            .args = shd_singleton(result),
            .mem = bb_mem(bb),
        })));
        return;
    }

    shd_log_fmt(ERROR, "subgroup_first emulation is not supported for ");
    shd_log_node(ERROR, t);
    shd_log_fmt(ERROR, ".\n");
    shd_error_die();
}

static const Node* build_subgroup_first(Context* ctx, BodyBuilder* bb, const Node* scope, const Node* src) {
    IrArena* a = ctx->rewriter.dst_arena;
    Module* m = ctx->rewriter.dst_module;
    const Node* t = get_unqualified_type(src->type);
    if (is_supported_natively(ctx, t))
        return gen_ext_instruction(bb, "spirv.core", SpvOpGroupNonUniformBroadcastFirst, shd_as_qualified_type(t, true), mk_nodes(a, scope, src));

    if (resolve_to_int_literal(scope)->value != SpvScopeSubgroup)
        shd_error("TODO")

    Node* fn = NULL;
    Node** found = shd_dict_find_value(const Node*, Node*, ctx->fns, t);
    if (found)
        fn = *found;
    else {
        const Node* src_param = param(a, shd_as_qualified_type(t, false), "src");
        fn = function(m, shd_singleton(src_param), shd_fmt_string_irarena(a, "subgroup_first_%s", name_type_safe(a, t)),
                      mk_nodes(a, annotation(a, (Annotation) { .name = "Generated"}), annotation(a, (Annotation) { .name = "Leaf" })), shd_singleton(
                        shd_as_qualified_type(t, true)));
        shd_dict_insert(const Node*, Node*, ctx->fns, t, fn);
        build_fn_body(ctx, fn, scope, src_param, t);
    }

    return shd_first(gen_call(bb, fn_addr_helper(a, fn), shd_singleton(src)));
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case ExtInstr_TAG: {
            ExtInstr payload = node->payload.ext_instr;
            if (strcmp(payload.set, "spirv.core") == 0 && payload.opcode == SpvOpGroupNonUniformBroadcastFirst) {
                BodyBuilder* bb = begin_body_with_mem(a, shd_rewrite_node(r, payload.mem));
                return yield_values_and_wrap_in_block(bb, shd_singleton(
                    build_subgroup_first(ctx, bb, shd_rewrite_node(r, payload.operands.nodes[0]), shd_rewrite_node(r, payload.operands.nodes[1]))));
            }
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

Module* lower_subgroup_ops(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(get_module_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    assert(!config->lower.emulate_subgroup_ops && "TODO");
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .fns =  shd_new_dict(const Node*, Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node)
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.fns);
    return dst;
}
