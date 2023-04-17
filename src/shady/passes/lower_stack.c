#include "passes.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"

#include "../transform/ir_gen_helpers.h"

#include "../rewrite.h"
#include "../type.h"

#include <assert.h>
#include <string.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    CompilerConfig* config;

    struct Dict* push_uniforms;
    struct Dict* push;
    struct Dict* pop_uniforms;
    struct Dict* pop;

    const Node* stack;
    const Node* stack_pointer;
    const Node* uniform_stack;
    const Node* uniform_stack_pointer;
} Context;

static const Node* gen_fn(Context* ctx, const Type* element_type, bool push, bool uniform) {
    struct Dict* cache = push ? (uniform ? ctx->push_uniforms : ctx->push) : (uniform ? ctx->pop_uniforms : ctx->pop);

    const Node** found = find_value_dict(const Node*, const Node*, cache, element_type);
    if (found)
        return *found;

    IrArena* arena = ctx->rewriter.dst_arena;
    const Type* qualified_t = qualified_type(arena, (QualifiedType) { .is_uniform = uniform, .type = element_type });

    const Node* param = push ? var(arena, qualified_t, "value") : NULL;
    Nodes params = push ? singleton(param) : empty(arena);
    Nodes return_ts = push ? empty(arena) : singleton(qualified_t);
    String name = format_string(arena, "generated_%s_%s_%s", push ? "push" : "pop", uniform ? "uniform" : "private", name_type_safe(arena, element_type));
    Node* fun = function(ctx->rewriter.dst_module, params, name, singleton(annotation(arena, (Annotation) { .name = "Generated" })), return_ts);
    insert_dict(const Node*, Node*, cache, element_type, fun);

    BodyBuilder* bb = begin_body(arena);

    const Node* element_size = gen_primop_e(bb, size_of_op, singleton(element_type), empty(arena));
    element_size = gen_conversion(bb, uint32_type(arena), element_size);

    // TODO somehow annotate the uniform guys as uniform
    const Node* stack_pointer = uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer;
    const Node* stack = uniform ? ctx->uniform_stack : ctx->stack;

    const Node* stack_size = gen_load(bb, stack_pointer);

    if (!push) // for pop, we decrease the stack size first
        stack_size = gen_primop_ce(bb, sub_op, 2, (const Node* []) { stack_size, element_size});

    const Node* addr = gen_lea(bb, stack, stack_size, nodes(arena, 1, (const Node* []) { uint32_literal(arena, 0) }));
    assert(get_unqualified_type(addr->type)->tag == PtrType_TAG);
    AddressSpace addr_space = get_unqualified_type(addr->type)->payload.ptr_type.address_space;

    addr = gen_reinterpret_cast(bb, ptr_type(arena, (PtrType) {.address_space = addr_space, .pointed_type = element_type}), addr);

    if (uniform) {
        assert(is_qualified_type_uniform(stack_pointer->type));
        assert(is_qualified_type_uniform(stack_size->type));
        assert(is_qualified_type_uniform(stack->type));
        assert(is_qualified_type_uniform(addr->type));
    }

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
        bind_instruction(bb, prim_op(arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(arena, string_lit(arena, (StringLiteral) { .string = name })) }));
        bind_instruction(bb, prim_op(arena, (PrimOp) { .op = debug_printf_op, .operands = mk_nodes(arena, string_lit(arena, (StringLiteral) { .string = "stack size after: %d\n" }), stack_size) }));
    }

    if (push) {
        fun->payload.fun.body = finish_body(bb, fn_ret(arena, (Return) { .fn = fun, .args = empty(arena) }));
    } else {
        assert(popped_value);
        fun->payload.fun.body = finish_body(bb, fn_ret(arena, (Return) { .fn = fun, .args = singleton(popped_value) }));
    }
    return fun;
}

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* old_instruction = node->payload.let.instruction;
    const Node* tail = rewrite_node(&ctx->rewriter, node->payload.let.tail);

    if (old_instruction->tag == PrimOp_TAG) {
        const PrimOp* oprim_op = &old_instruction->payload.prim_op;
        switch (oprim_op->op) {
            case get_stack_pointer_op:
            case get_stack_pointer_uniform_op: {
                BodyBuilder* bb = begin_body(arena);
                bool uniform = oprim_op->op == get_stack_pointer_uniform_op;
                const Node* sp = gen_load(bb, uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer);
                return finish_body(bb, let(arena, quote_single(arena, sp), tail));
            }
            case set_stack_pointer_op:
            case set_stack_pointer_uniform_op: {
                BodyBuilder* bb = begin_body(arena);
                bool uniform = oprim_op->op == set_stack_pointer_uniform_op;
                const Node* val = rewrite_node(&ctx->rewriter, oprim_op->operands.nodes[0]);
                gen_store(bb, uniform ? ctx->uniform_stack_pointer : ctx->stack_pointer, val);
                return finish_body(bb, let(arena, unit(arena), tail));
            }
            case get_stack_base_uniform_op:
            case get_stack_base_op: {
                BodyBuilder* bb = begin_body(arena);
                const Node* stack_pointer = oprim_op->op == get_stack_base_op ? ctx->stack_pointer : ctx->uniform_stack_pointer;
                const Node* stack_size = gen_load(bb, stack_pointer);
                const Node* stack_base_ptr = gen_lea(bb, oprim_op->op == get_stack_base_op ? ctx->stack : ctx->uniform_stack, stack_size, empty(arena));
                if (ctx->config->printf_trace.stack_size) {
                    if (oprim_op->op == get_stack_base_op)
                        bind_instruction(bb, prim_op(arena, (PrimOp) {.op = debug_printf_op, .operands = mk_nodes(arena, string_lit(arena, (StringLiteral) {.string = "trace: stack_size=%d\n"}), stack_size)}));
                    else
                        bind_instruction(bb, prim_op(arena, (PrimOp) {.op = debug_printf_op, .operands = mk_nodes(arena, string_lit(arena, (StringLiteral) {.string = "trace: uniform stack_size=%d\n"}), stack_size)}));
                }
                return finish_body(bb, let(arena, quote_single(arena, stack_base_ptr), tail));
            }
            case push_stack_op:
            case push_stack_uniform_op:
            case pop_stack_op:
            case pop_stack_uniform_op: {
                BodyBuilder* bb = begin_body(arena);
                const Type* element_type = rewrite_node(&ctx->rewriter, first(oprim_op->type_arguments));

                bool push = oprim_op->op == push_stack_op || oprim_op->op == push_stack_uniform_op;
                bool uniform = oprim_op->op == push_stack_uniform_op || oprim_op->op == pop_stack_uniform_op;

                const Node* fn = gen_fn(ctx, element_type, push, uniform);
                Nodes args = push ? singleton(rewrite_node(&ctx->rewriter, first(oprim_op->operands))) : empty(arena);
                Nodes results = bind_instruction(bb, call(arena, (Call) { .callee = fn_addr(arena, (FnAddr) { .fn = fn }), .args = args}));

                if (push)
                    return finish_body(bb, let(arena, unit(arena), tail));

                assert(results.count == 1);
                return finish_body(bb, let(arena, quote_single(arena, first(results)), tail));
            }
            default: break;
        }
    }

    return let(arena, rewrite_node(&ctx->rewriter, old_instruction), tail);
}

static const Node* process_node(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;

    if (old->tag == Function_TAG && strcmp(get_abstraction_name(old), "generated_init") == 0) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
        BodyBuilder* bb = begin_body(arena);

        // Make sure to zero-init the stack pointers
        // TODO isn't this redundant with thoose things having an initial value already ?
        // is this an old forgotten workarround ?
        const Node* uniform_stack_pointer = ctx->uniform_stack_pointer;
        gen_store(bb, uniform_stack_pointer, uint32_literal(arena, 0));
        const Node* stack_pointer = ctx->stack_pointer;
        gen_store(bb, stack_pointer, uint32_literal(arena, 0));
        new->payload.fun.body = finish_body(bb, rewrite_node(&ctx->rewriter, old->payload.fun.body));
        return new;
    }

    switch (old->tag) {
        case Let_TAG: return process_let(ctx, old);
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void lower_stack(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    IrArena* dst_arena = get_module_arena(dst);

    const Type* stack_base_element = uint8_type(dst_arena);
    const Type* stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_thread_stack_size),
    });
    const Type* uniform_stack_arr_type = arr_type(dst_arena, (ArrType) {
        .element_type = stack_base_element,
        .size = uint32_literal(dst_arena, config->per_subgroup_stack_size),
    });
    const Type* stack_counter_t = uint32_type(dst_arena);

    Nodes annotations = mk_nodes(dst_arena, annotation(dst_arena, (Annotation) { .name = "Generated" }));

    // Arrays for the stacks
    Node* stack_decl = global_var(dst, annotations, stack_arr_type, "stack", AsPrivatePhysical);
    Node* uniform_stack_decl = global_var(dst, annotations, uniform_stack_arr_type, "uniform_stack", AsSubgroupPhysical);

    // Pointers into those arrays
    Node* stack_ptr_decl = global_var(dst, annotations, stack_counter_t, "stack_ptr", AsPrivateLogical);
    stack_ptr_decl->payload.global_variable.init = uint32_literal(dst_arena, 0);
    Node* uniform_stack_ptr_decl = global_var(dst, annotations, stack_counter_t, "uniform_stack_ptr", AsSubgroupLogical);
    uniform_stack_ptr_decl->payload.global_variable.init = uint32_literal(dst_arena, 0);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),

        .config = config,

        .push_uniforms = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .push = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .pop_uniforms = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .pop = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node),

        .stack = ref_decl(dst_arena, (RefDecl) { .decl = stack_decl }),
        .stack_pointer = ref_decl(dst_arena, (RefDecl) { .decl = stack_ptr_decl }),
        .uniform_stack = ref_decl(dst_arena, (RefDecl) { .decl = uniform_stack_decl }),
        .uniform_stack_pointer = ref_decl(dst_arena, (RefDecl) { .decl = uniform_stack_ptr_decl }),
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);

    destroy_dict(ctx.push);
    destroy_dict(ctx.pop);
    destroy_dict(ctx.push_uniforms);
    destroy_dict(ctx.pop_uniforms);
}
