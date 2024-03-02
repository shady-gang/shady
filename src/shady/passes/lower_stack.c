#include "passes.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"
#include "util.h"

#include "../rewrite.h"
#include "../type.h"
#include "../ir_private.h"

#include "../transform/ir_gen_helpers.h"

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

    const Node** found = find_value_dict(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* a = ctx->rewriter.dst_arena;
    const Type* qualified_t = qualified_type(a, (QualifiedType) { .is_uniform = false, .type = element_type });

    const Node* param = push ? var(a, qualified_t, "value") : NULL;
    Nodes params = push ? singleton(param) : empty(a);
    Nodes return_ts = push ? empty(a) : singleton(qualified_t);
    String name = format_string_arena(a->arena, "generated_%s_%s", push ? "push" : "pop", name_type_safe(a, element_type));
    Node* fun = function(ctx->rewriter.dst_module, params, name, singleton(annotation(a, (Annotation) { .name = "Generated" })), return_ts);
    insert_dict(const Node*, Node*, cache, element_type, fun);

    BodyBuilder* bb = begin_body(a);

    const Node* element_size = gen_primop_e(bb, size_of_op, singleton(element_type), empty(a));
    element_size = gen_conversion(bb, uint32_type(a), element_size);

    // TODO somehow annotate the uniform guys as uniform
    const Node* stack_pointer = ctx->stack_pointer;
    const Node* stack = ctx->stack;

    const Node* stack_size = gen_load(bb, stack_pointer);

    if (!push) // for pop, we decrease the stack size first
        stack_size = gen_primop_ce(bb, sub_op, 2, (const Node* []) { stack_size, element_size});

    const Node* addr = gen_lea(bb, stack, stack_size, nodes(a, 1, (const Node* []) {uint32_literal(a, 0) }));
    assert(get_unqualified_type(addr->type)->tag == PtrType_TAG);
    AddressSpace addr_space = get_unqualified_type(addr->type)->payload.ptr_type.address_space;

    addr = gen_reinterpret_cast(bb, ptr_type(a, (PtrType) {.address_space = addr_space, .pointed_type = element_type}), addr);

    const Node* popped_value = NULL;
    if (push)
        gen_store(bb, addr, param);
    else
        popped_value = gen_primop_ce(bb, load_op, 1, (const Node* []) { addr });

    if (push) // for push, we increase the stack size after the store
        stack_size = gen_primop_ce(bb, add_op, 2, (const Node* []) { stack_size, element_size});

    // store updated stack size
    gen_store(bb, stack_pointer, stack_size);
    if (ctx->config->printf_trace.stack_size) {
        bind_instruction(bb, prim_op(a, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(a, string_lit(a, (StringLiteral) { .string = name })) }));
        bind_instruction(bb, prim_op(a, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(a, string_lit(a, (StringLiteral) { .string = "stack size after: %d\n" }), stack_size) }));
    }

    if (push) {
        fun->payload.fun.body = finish_body(bb, fn_ret(a, (Return) { .fn = fun, .args = empty(a) }));
    } else {
        assert(popped_value);
        fun->payload.fun.body = finish_body(bb, fn_ret(a, (Return) { .fn = fun, .args = singleton(popped_value) }));
    }
    return fun;
}

static const Node* process_node(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;

    if (old->tag == Function_TAG && strcmp(get_abstraction_name(old), "generated_init") == 0) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
        BodyBuilder* bb = begin_body(a);

        // Make sure to zero-init the stack pointers
        // TODO isn't this redundant with thoose things having an initial value already ?
        // is this an old forgotten workaround ?
        const Node* stack_pointer = ctx->stack_pointer;
        gen_store(bb, stack_pointer, uint32_literal(a, 0));
        new->payload.fun.body = finish_body(bb, rewrite_node(&ctx->rewriter, old->payload.fun.body));
        return new;
    }

    switch (old->tag) {
        case PrimOp_TAG: {
            const PrimOp* oprim_op = &old->payload.prim_op;
            switch (oprim_op->op) {
                case get_stack_pointer_op: {
                    BodyBuilder* bb = begin_body(a);
                    const Node* sp = gen_load(bb, ctx->stack_pointer);
                    return yield_values_and_wrap_in_block(bb, singleton(sp));
                }
                case set_stack_pointer_op: {
                    BodyBuilder* bb = begin_body(a);
                    const Node* val = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[0]);
                    gen_store(bb, ctx->stack_pointer, val);
                    return yield_values_and_wrap_in_block(bb, empty(a));
                }
                case get_stack_base_op: {
                    BodyBuilder* bb = begin_body(a);
                    const Node* stack_pointer = ctx->stack_pointer;
                    const Node* stack_size = gen_load(bb, stack_pointer);
                    const Node* stack_base_ptr = gen_lea(bb, ctx->stack, stack_size, empty(a));
                    if (ctx->config->printf_trace.stack_size) {
                        if (oprim_op->op == get_stack_base_op)
                            bind_instruction(bb, prim_op(a, (PrimOp) {.op = debug_printf_op, .operands = mk_nodes(a, string_lit(a, (StringLiteral) {.string = "trace: stack_size=%d\n"}), stack_size)}));
                        else
                            bind_instruction(bb, prim_op(a, (PrimOp) {.op = debug_printf_op, .operands = mk_nodes(a, string_lit(a, (StringLiteral) {.string = "trace: uniform stack_size=%d\n"}), stack_size)}));
                    }
                    return yield_values_and_wrap_in_block(bb, singleton(stack_base_ptr));
                }
                case push_stack_op:
                case pop_stack_op: {
                    BodyBuilder* bb = begin_body(a);
                    const Type* element_type = rewrite_node(&ctx->rewriter, first(oprim_op->type_arguments));

                    bool push = oprim_op->op == push_stack_op;

                    const Node* fn = gen_fn(ctx, element_type, push);
                    Nodes args = push ? singleton(rewrite_node(&ctx->rewriter, first(oprim_op->operands))) : empty(a);
                    Nodes results = bind_instruction(bb, call(a, (Call) { .callee = fn_addr_helper(a, fn), .args = args}));

                    if (push)
                        return yield_values_and_wrap_in_block(bb, empty(a));

                    assert(results.count == 1);
                    return yield_values_and_wrap_in_block(bb, results);
                }
                default: break;
            }
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Module* lower_stack(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    const Type* stack_base_element = uint8_type(a);
    const Type* stack_arr_type = arr_type(a, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(a, config->per_thread_stack_size),
    });
    const Type* stack_counter_t = uint32_type(a);

    Nodes annotations = mk_nodes(a, annotation(a, (Annotation) { .name = "Generated" }));

    // Arrays for the stacks
    Node* stack_decl = global_var(dst, annotations, stack_arr_type, "stack", AsPrivatePhysical);

    // Pointers into those arrays
    Node* stack_ptr_decl = global_var(dst, annotations, stack_counter_t, "stack_ptr", AsPrivateLogical);
    stack_ptr_decl->payload.global_variable.init = uint32_literal(a, 0);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process_node),

        .config = config,

        .push = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .pop = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),

        .stack = ref_decl_helper(a, stack_decl),
        .stack_pointer = ref_decl_helper(a, stack_ptr_decl),
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);

    destroy_dict(ctx.push);
    destroy_dict(ctx.pop);
    return dst;
}
