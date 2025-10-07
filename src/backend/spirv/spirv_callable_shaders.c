#include "shady/pass.h"
#include "shady/ir/memory_layout.h"
#include "shady/ir/decl.h"
#include "shady/ir/annotation.h"
#include "shady/ir/function.h"
#include "shady/ir/ext.h"
#include "shady/analysis/uses.h"
#include "shady/dict.h"

#include "portability.h"
#include "log.h"

#include "spirv/unified1/spirv.h"

typedef struct {
    Rewriter rewriter;
    const UsesMap* uses;
    const TargetConfig* target;
    uint64_t* sbt_index;
    Node2Node signature2type;
    Node2Node signature2calleevar;
    Node2Node signature2callervar;
} Context;

static bool is_indirectly_called(Context* ctx, const Node* f) {
    const Use* use = shd_get_first_use(ctx->uses, f);
    for (; use; use = use->next_use) {
        if (use->user->tag == FnAddr_TAG)
            return true;
    }
    return false;
}

static const Type* get_type_for_signature(Context* ctx, const Node* fnt) {
    assert(fnt && fnt->tag == FnType_TAG);
    assert(fnt->arena == ctx->rewriter.src_arena);
    const Type* found = shd_node2node_find(ctx->signature2type, fnt);
    if (found) return found;
    Rewriter* r = &ctx->rewriter;
    IrArena* a = ctx->rewriter.dst_arena;
    FnType payload = fnt->payload.fn_type;

    Nodes param_types = payload.param_types;
    Nodes ret_types = payload.return_types;

    LARRAY(const Type*, members, param_types.count + ret_types.count);
    size_t j = 0;
    for (size_t i = 0; i < param_types.count; i++) {
        members[j++] = shd_rewrite_node(r, shd_get_unqualified_type(param_types.nodes[i]));
    }
    for (size_t i = 0; i < ret_types.count; i++) {
        members[j++] = shd_rewrite_node(r, shd_get_unqualified_type(ret_types.nodes[i]));
    }
    const Type* rec = shd_struct_type_with_members(a, 0, shd_nodes(a, j, members));
    shd_set_debug_name(rec, shd_fmt_string_irarena(a, "%s_io_t", shd_get_type_name(a, fnt)));
    shd_node2node_insert(ctx->signature2type, fnt, rec);
    return rec;
}

static const Type* get_callee_var_for_signature(Context* ctx, const Node* fnt) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* found = shd_node2node_find(ctx->signature2calleevar, fnt);
    if (found) return found;
    const Type* rec = get_type_for_signature(ctx, fnt);
    Node* var = shd_global_var(ctx->rewriter.dst_module, (GlobalVariable) {
        .type = rec,
        .is_ref = true,
        .address_space = AsIncomingCallableDataKHR,
    });
    shd_set_debug_name(var, shd_fmt_string_irarena(a, "%s_callee_var", shd_get_type_name(a, fnt)));
    shd_node2node_insert(ctx->signature2calleevar, fnt, var);
    return var;
}

static const Type* get_caller_var_for_signature(Context* ctx, const Node* fnt) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* found = shd_node2node_find(ctx->signature2callervar, fnt);
    if (found) return found;
    const Type* rec = get_type_for_signature(ctx, fnt);
    Node* var = shd_global_var(ctx->rewriter.dst_module, (GlobalVariable) {
        .type = rec,
        .is_ref = true,
        .address_space = AsCallableDataKHR,
    });
    shd_set_debug_name(var, shd_fmt_string_irarena(a, "%s_caller_var", shd_get_type_name(a, fnt)));
    shd_node2node_insert(ctx->signature2callervar, fnt, var);
    return var;
}

static Nodes rewrite_call(Context* ctx, BodyBuilder* bb, const Node* ocallee, const Nodes old_args) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    const Type* ofnt = shd_get_unqualified_type(ocallee->type);
    IrArena* oa = ofnt->arena;
    assert(ofnt->tag == PtrType_TAG);
    shd_deconstruct_pointer_type(&ofnt);
    assert(ofnt->tag == FnType_TAG);

    const Node* var = get_caller_var_for_signature(ctx, ofnt);
    for (size_t i = 0; i < old_args.count; i++) {
        const Node* arg = shd_rewrite_node(r, old_args.nodes[i]);
        shd_bld_store(bb, ptr_composite_element_helper(a, var, shd_uint32_literal(a, i)), arg);
    }

    const Node* execute_callable_op = shd_make_ext_spv_op(a, "spirv.core", SpvOpExecuteCallableKHR, false, NULL, 2);
    shd_bld_add_instruction(bb, ext_instr_helper(a, shd_bld_mem(bb), execute_callable_op, mk_nodes(a, shd_rewrite_node(r, ocallee), var)));

    size_t num_results = ofnt->payload.fn_type.return_types.count;
    LARRAY(const Node*, results, num_results);
    for (size_t i = 0; i < num_results; i++) {
        results[i] = shd_bld_load(bb, ptr_composite_element_helper(a, var, shd_uint32_literal(a, old_args.count + i)));
    }
    return shd_nodes(a, num_results, results);
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    const TargetConfig* target = &shd_get_arena_config(a)->target;

    switch (node->tag) {
        case FnAddr_TAG: {
            FnAddr payload = node->payload.fn_addr;
            if (shd_lookup_annotation(payload.fn, "Leaf"))
                break;
            uint64_t index = (*ctx->sbt_index)++;
            const Node* new = int_literal(a, (IntLiteral) {
                .is_signed = false,
                .width = ctx->target->memory.fn_ptr_size,
                .value = index
            });
            shd_register_processed(r, node, new);
            shd_rewrite_node(r, payload.fn);
            return new;
        }
        case PtrType_TAG: {
            const Node* pointee = node->payload.ptr_type.pointed_type;
            if (pointee->tag == FnType_TAG && !ctx->target->capabilities.native_tailcalls)
                return int_type_helper(a, ctx->target->memory.fn_ptr_size, false);
            break;
        }
        case Function_TAG: {
            //if (!is_indirectly_called(ctx, node))
            //    break;
            if (shd_lookup_annotation(node, "Leaf"))
                break;

            Node* new = (Node*) shd_recreate_node(r, node);
            assert(new && new->tag == Function_TAG);

            IrArena* oa = node->arena;

            Node* wrapper = function_helper(r->dst_module, shd_empty(a), shd_empty(a));
            shd_add_annotation(wrapper, annotation_value_helper(a, "EntryPoint", string_lit_helper(a, "Callable")));
            const Node* sbt_index = shd_rewrite_node(r, fn_addr_helper(oa, node));
            shd_add_annotation(wrapper, annotation_value_helper(a, "SBTIndex", sbt_index));
            shd_module_add_export(r->dst_module, shd_fmt_string_irarena(a, "callee_%d", shd_get_int_value(sbt_index, false)), wrapper);

            const Node* var = get_callee_var_for_signature(ctx, node->type);

            BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(wrapper));
            Nodes params = get_abstraction_params(new);
            LARRAY(const Node*, args, params.count);
            for (size_t i = 0; i < params.count; i++) {
                args[i] = shd_bld_load(bb, ptr_composite_element_helper(a, var, shd_uint32_literal(a, i)));
            }
            Nodes results = shd_bld_call(bb, new, shd_nodes(a, params.count, args));
            for (size_t i = 0; i < results.count; i++) {
                shd_bld_store(bb, ptr_composite_element_helper(a, var, shd_uint32_literal(a, params.count + i)), results.nodes[i]);
            }
            shd_set_abstraction_body(wrapper, shd_bld_return(bb, shd_empty(a)));

            return new;
        }
        case Call_TAG: {
            Call payload = node->payload.call;
            if (shd_lookup_annotation(payload.callee, "Leaf"))
                break;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes results = rewrite_call(ctx, bb, fn_addr_helper(payload.callee->arena, payload.callee), payload.args);
            return shd_bld_to_instr_yield_values(bb, results);
        }
        case IndirectCall_TAG: {
            IndirectCall payload = node->payload.indirect_call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes results = rewrite_call(ctx, bb, payload.callee, payload.args);
            return shd_bld_to_instr_yield_values(bb, results);
        }
        case IndirectTailCall_TAG: {
            IndirectTailCall payload = node->payload.indirect_tail_call;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes results = rewrite_call(ctx, bb, payload.callee, payload.args);
            return shd_bld_return(bb, results);
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* shd_lower_to_callable_shaders(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    uint64_t num_callables = 0;
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .sbt_index = &num_callables,
        .target = &aconfig.target,
        .uses = shd_new_uses_map_module(src, NcType),
        .signature2type = shd_new_node2node(),
        .signature2calleevar = shd_new_node2node(),
        .signature2callervar = shd_new_node2node(),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_uses_map(ctx.uses);
    shd_destroy_node2node(ctx.signature2type);
    shd_destroy_node2node(ctx.signature2calleevar);
    shd_destroy_node2node(ctx.signature2callervar);

    const Node* ep = shd_module_get_exported(dst, aconfig.target.entry_point);
    assert(ep);
    shd_add_annotation(ep, annotation_value_helper(a, "NumCallables", shd_uint32_literal(a, num_callables)));
    return dst;
}
