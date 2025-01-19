#include "shady/pass.h"
#include "shady/ir/cast.h"
#include "shady/ir/memory_layout.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "util.h"

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
} Context;

static const Node* gen_fn(Context* ctx, const Type* element_type, bool push) {
    struct Dict* cache = push ? ctx->push : ctx->pop;

    const Node** found = shd_dict_find_value(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* a = ctx->rewriter.dst_arena;
    const Type* qualified_t = qualified_type(a, (QualifiedType) { .is_uniform = false, .type = element_type });

    const Node* value_param = NULL;
    if (push) {
        value_param = param_helper(a, qualified_t);
        shd_set_debug_name(value_param, "value");
    }
    Nodes params = push ? shd_singleton(value_param) : shd_empty(a);
    Nodes return_ts = push ? shd_empty(a) : shd_singleton(qualified_t);
    String name = shd_format_string_arena(a->arena, "generated_%s_%s", push ? "push" : "pop", shd_get_type_name(a, element_type));
    Node* fun = function_helper(ctx->rewriter.dst_module, params, name, return_ts);
    shd_add_annotation_named(fun, "Generated");
    shd_add_annotation_named(fun, "Leaf");
    shd_dict_insert(const Node*, Node*, cache, element_type, fun);

    BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));

    const Node* element_size = prim_op_helper(a, size_of_op, shd_singleton(element_type), shd_empty(a));
    element_size = shd_bld_conversion(bb, shd_uint32_type(a), element_size);

    // TODO somehow annotate the uniform guys as uniform
    const Node* stack_pointer = ctx->stack_pointer;
    const Node* stack = ctx->stack;

    const Node* stack_size = shd_bld_load(bb, stack_pointer);

    if (!push) // for pop, we decrease the stack size first
        stack_size = prim_op_helper(a, sub_op, shd_empty(a), mk_nodes(a, stack_size, element_size));

    const Node* addr = lea_helper(a, ctx->stack, shd_int32_literal(a, 0), shd_singleton(stack_size));
    assert(shd_get_unqualified_type(addr->type)->tag == PtrType_TAG);
    AddressSpace addr_space = shd_get_unqualified_type(addr->type)->payload.ptr_type.address_space;

    addr = shd_bld_reinterpret_cast(bb, ptr_type(a, (PtrType) { .address_space = addr_space, .pointed_type = element_type }), addr);

    const Node* popped_value = NULL;
    if (push)
        shd_bld_store(bb, addr, value_param);
    else
        popped_value = shd_bld_load(bb, addr);

    if (push) // for push, we increase the stack size after the store
        stack_size = prim_op_helper(a, add_op, shd_empty(a), mk_nodes(a, stack_size, element_size));

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
            const Node* val = shd_rewrite_node(r, old->payload.set_stack_size.value);
            shd_bld_store(bb, ctx->stack_pointer, val);
            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        case GetStackBaseAddr_TAG: {
            return prim_op_helper(a, reinterpret_op, shd_singleton(shd_rewrite_node(r, shd_get_unqualified_type(old->type))), mk_nodes(a, ctx->stack));
        }
        case PushStack_TAG:{
            assert(ctx->stack);
            PushStack payload = old->payload.push_stack;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            const Type* element_type = shd_rewrite_node(&ctx->rewriter, shd_get_unqualified_type(old->payload.push_stack.value->type));

            bool push = true;

            const Node* fn = gen_fn(ctx, element_type, push);
            Nodes args = shd_singleton(shd_rewrite_node(&ctx->rewriter, old->payload.push_stack.value));
            shd_bld_call(bb, fn_addr_helper(a, fn), args);

            return shd_bld_to_instr_yield_values(bb, shd_empty(a));
        }
        case PopStack_TAG: {
            assert(ctx->stack);
            PopStack payload = old->payload.pop_stack;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            const Type* element_type = shd_rewrite_node(&ctx->rewriter, old->payload.pop_stack.type);

            bool push = false;

            const Node* fn = gen_fn(ctx, element_type, push);
            Nodes results = shd_bld_call(bb, fn_addr_helper(a, fn), shd_empty(a));

            assert(results.count == 1);
            return shd_bld_to_instr_yield_values(bb, results);
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

Module* shd_pass_lower_stack(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),

        .config = config,

        .push = shd_new_dict(const Node*, Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .pop = shd_new_dict(const Node*, Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };

    if (config->per_thread_stack_size > 0) {
        const Type* stack_base_element = shd_uint8_type(a);
        const Type* stack_arr_type = arr_type(a, (ArrType) {
                .element_type = stack_base_element,
                .size = shd_uint32_literal(a, config->per_thread_stack_size),
        });
        const Type* stack_counter_t = shd_uint32_type(a);

        // Arrays for the stacks
        Node* stack_decl = global_variable_helper(dst, stack_arr_type, "stack", AsPrivate);
        shd_add_annotation_named(stack_decl, "Generated");

        // Pointers into those arrays
        // Node* stack_ptr_decl = global_variable_helper(dst, annotations, stack_counter_t, "stack_ptr", AsPrivate);
        Node* stack_ptr_decl = shd_global_var(dst, (GlobalVariable) {
            .type = stack_counter_t,
            .name = "stack_ptr",
            .address_space = AsPrivate,
            .is_ref = true
        });
        shd_add_annotation_named(stack_ptr_decl, "Generated");
        stack_ptr_decl->payload.global_variable.init = shd_uint32_literal(a, 0);

        ctx.stack = stack_decl;
        ctx.stack_pointer = stack_ptr_decl;
    }

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);

    shd_destroy_dict(ctx.push);
    shd_destroy_dict(ctx.pop);
    return dst;
}
