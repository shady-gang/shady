#include "shady/pass.h"
#include "shady/dict.h"
#include "shady/ir/cast.h"
#include "shady/ir/memory_layout.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "util.h"
#include "shady/config.h"

#include <assert.h>
#include <string.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;

    struct Dict* push;
    struct Dict* pop;

    const Node* stack;
    const Node* stack_pointer;
    const Node* max_stack_pointer;
} Context;

static const Node* gen_fn(Context* ctx, const Type* element_type, bool push) {
    Node2Node cache = push ? ctx->push : ctx->pop;

    const Node* found = shd_node2node_find(cache, element_type);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    const Type* qualified_t = qualified_type(a, (QualifiedType) { .scope = shd_get_arena_config(a)->target.scopes.bottom, .type = element_type });

    const Node* value_param = NULL;
    if (push) {
        value_param = param_helper(a, qualified_t);
        shd_set_debug_name(value_param, "value");
    }
    Nodes params = push ? shd_singleton(value_param) : shd_empty(a);
    Nodes return_ts = push ? shd_empty(a) : shd_singleton(qualified_t);
    String name = shd_format_string_arena(a->arena, "generated_%s_%s", push ? "push" : "pop", shd_get_type_name(a, element_type));
    Node* fun = function_helper(ctx->rewriter.dst_module, params, return_ts);
    shd_add_annotation_named(fun, "Generated");
    shd_add_annotation_named(fun, "Leaf");
    shd_node2node_insert(cache, element_type, fun);

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));

    const Node* element_size = size_of_helper(a, element_type);
    element_size = shd_bld_conversion(bb, shd_uint32_type(a), element_size);

    // TODO somehow annotate the uniform guys as uniform
    const Node* stack_pointer = ctx->stack_pointer;
    const Node* stack = ctx->stack;

    const Node* stack_size = shd_bld_load(bb, stack_pointer);

    if (!push) // for pop, we decrease the stack size first
        stack_size = prim_op_helper(a, sub_op, mk_nodes(a, stack_size, element_size));

    const Node* addr = lea_helper(a, ctx->stack, shd_int32_literal(a, 0), shd_singleton(stack_size));
    assert(shd_get_unqualified_type(addr->type)->tag == PtrType_TAG);
    AddressSpace addr_space = shd_get_unqualified_type(addr->type)->payload.ptr_type.address_space;

    addr = shd_bld_bitcast(bb, ptr_type(a, (PtrType) { .address_space = addr_space, .pointed_type = element_type }), addr);

    const Node* popped_value = NULL;
    if (push)
        shd_bld_store(bb, addr, value_param);
    else
        popped_value = shd_bld_load(bb, addr);

    if (push) // for push, we increase the stack size after the store
        stack_size = prim_op_helper(a, add_op, mk_nodes(a, stack_size, element_size));

    if (ctx->max_stack_pointer) {
        const Node* old_max_stack_size = shd_bld_load(bb, ctx->max_stack_pointer);
        const Node* new_max_stack_size = prim_op_helper(a, max_op, mk_nodes(a, old_max_stack_size, stack_size));
        shd_bld_store(bb, ctx->max_stack_pointer, new_max_stack_size);
    }

    // store updated stack size
    shd_bld_store(bb, stack_pointer, stack_size);
    if (ctx->config->printf_trace.stack_size) {
        shd_bld_debug_printf(bb, name, shd_empty(a));
        shd_bld_debug_printf(bb, "stack size after: %d\n", shd_singleton(stack_size));
    }

    if (push) {
        shd_set_abstraction_body(fun, shd_bld_return(bb, shd_empty(a)));
    } else {
        assert(popped_value);
        shd_set_abstraction_body(fun, shd_bld_return(bb, shd_singleton(popped_value)));
    }
    return fun;
}

static const Node* process_node(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case GetStackSize_TAG: {
            assert(ctx->stack);
            GetStackSize payload = old->payload.get_stack_size;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            const Node* sp = shd_bld_load(bb, ctx->stack_pointer);
            return shd_bld_to_instr_yield_values(bb, shd_singleton(sp));
        }
        case SetStackSize_TAG: {
            assert(ctx->stack);
            SetStackSize payload = old->payload.set_stack_size;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            const Node* new_stack_size = shd_rewrite_node(r, old->payload.set_stack_size.value);
            shd_bld_store(bb, ctx->stack_pointer, new_stack_size);

            if (ctx->max_stack_pointer) {
                const Node* old_max_stack_size = shd_bld_load(bb, ctx->max_stack_pointer);
                const Node* new_max_stack_size = prim_op_helper(a, max_op, mk_nodes(a, old_max_stack_size, new_stack_size));
                shd_bld_store(bb, ctx->max_stack_pointer, new_max_stack_size);
            }

            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        case GetStackBaseAddr_TAG: {
            return bit_cast_helper(a, shd_rewrite_node(r, shd_get_unqualified_type(old->type)), ctx->stack);
        }
        case PushStack_TAG:{
            assert(ctx->stack);
            PushStack payload = old->payload.push_stack;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            const Type* element_type = shd_rewrite_node(&ctx->rewriter, shd_get_unqualified_type(old->payload.push_stack.value->type));

            bool push = true;

            const Node* fn = gen_fn(ctx, element_type, push);
            Nodes args = shd_singleton(shd_rewrite_node(&ctx->rewriter, old->payload.push_stack.value));
            shd_bld_call(bb, fn, args);

            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        case PopStack_TAG: {
            assert(ctx->stack);
            PopStack payload = old->payload.pop_stack;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            const Type* element_type = shd_rewrite_node(&ctx->rewriter, old->payload.pop_stack.type);

            bool push = false;

            const Node* fn = gen_fn(ctx, element_type, push);
            Nodes results = shd_bld_call(bb, fn, shd_empty(a));

            assert(results.count == 1);
            return shd_bld_to_instr_yield_values(bb, results);
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

Module* shd_pass_lower_stack(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),

        .config = config,

        .push = shd_new_node2node(),
        .pop = shd_new_node2node(),
    };

    if (config->per_thread_stack_size > 0) {
        const Type* stack_base_element = shd_uint8_type(a);
        const Type* stack_arr_type = arr_type(a, (ArrType) {
                .element_type = stack_base_element,
                .size = shd_uint32_literal(a, config->per_thread_stack_size),
        });
        const Type* stack_counter_t = shd_uint32_type(a);

        // Arrays for the stacks
        Node* stack_decl = global_variable_helper(dst, stack_arr_type, AsPrivate);
        shd_set_debug_name(stack_decl, "stack");
        shd_add_annotation_named(stack_decl, "Generated");
        if (config->lower.force_stack_in_scratch)
            shd_add_annotation_named(stack_decl, "AllocateInScratchMemory");
        ctx.stack = stack_decl;

        // Pointers into those arrays
        // Node* stack_ptr_decl = global_variable_helper(dst, annotations, stack_counter_t, "stack_ptr", AsPrivate);
        Node* stack_ptr_decl = shd_global_var(dst, (GlobalVariable) {
            .type = stack_counter_t,
            .address_space = AsPrivate,
            .is_ref = true
        });
        shd_set_debug_name(stack_ptr_decl, "stack_ptr");
        shd_add_annotation_named(stack_ptr_decl, "Generated");
        stack_ptr_decl->payload.global_variable.init = shd_uint32_literal(a, 0);
        ctx.stack_pointer = stack_ptr_decl;

        if (config->printf_trace.max_stack_size) {
            Node* max_stack_size_var = shd_global_var(dst, (GlobalVariable) {
                .type = stack_counter_t,
                .address_space = AsPrivate,
                .is_ref = true
            });
            shd_set_debug_name(stack_ptr_decl, "max_stack_ptr");
            shd_add_annotation_named(max_stack_size_var, "Generated");
            max_stack_size_var->payload.global_variable.init = shd_uint32_literal(a, 0);
            ctx.max_stack_pointer = max_stack_size_var;

            const Node* old = shd_module_get_fini_fn(src);
            Node* new;
            BodyBuilder* bb = shd_bld_begin_fn_rewrite(&ctx.rewriter, old, &new);
            shd_bld_debug_printf(bb, "max_stack_size: %d\n", mk_nodes(a, shd_bld_load(bb, max_stack_size_var)));
            shd_bld_finish_fn_rewrite(&ctx.rewriter, old, new, bb);
        }
    }

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);

    shd_destroy_node2node(ctx.push);
    shd_destroy_node2node(ctx.pop);
    return dst;
}
