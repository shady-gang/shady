#include "ir_gen_helpers.h"

#include "list.h"
#include "portability.h"

#include "../ir_private.h"
#include "../type.h"
#include "../rewrite.h"

#include <string.h>
#include <assert.h>

Nodes gen_primop(BodyBuilder* bb, Op op, Nodes operands) {
    assert(bb->arena->config.check_types);
    const Node* instruction = prim_op(bb->arena, (PrimOp) { .op = op, .operands = operands });
    return bind_instruction(bb, instruction);
}

Nodes gen_primop_c(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]) {
    return gen_primop(bb, op, nodes(bb->arena, operands_count, operands));
}

const Node* gen_primop_ce(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]) {
    Nodes result = gen_primop_c(bb, op, operands_count, operands);
    assert(result.count == 1);
    return result.nodes[0];
}

const Node* gen_primop_e(BodyBuilder* bb, Op op, Nodes nodes) {
    Nodes result = gen_primop(bb, op, nodes);
    assert(result.count == 1);
    return result.nodes[0];
}

void gen_push_value_stack(BodyBuilder* instructions, const Node* value) {
    gen_primop_c(instructions, push_stack_op, 2, (const Node*[]) { extract_operand_type(value->type), value });
}

void gen_push_values_stack(BodyBuilder* instructions, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        gen_push_value_stack(instructions, value);
    }
}

const Node* gen_pop_value_stack(BodyBuilder* bb, const Type* type) {
    const Node* instruction = prim_op(bb->arena, (PrimOp) { .op = pop_stack_op, .type_arguments = nodes(bb->arena, 1, (const Node*[]) { type }) });
    return first(bind_instruction(bb, instruction));
}

const Node* gen_reinterpret_cast(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = reinterpret_op, .operands = singleton(src), .type_arguments = singleton(dst)})));
}

const Node* gen_merge_i32s_i64(BodyBuilder* bb, const Node* lo, const Node* hi) {
    // widen them
    lo = gen_reinterpret_cast(bb, int64_type(bb->arena), lo);
    hi = gen_reinterpret_cast(bb, int64_type(bb->arena), hi);
    // shift hi by 32
    hi = gen_primop_ce(bb, lshift_op, 2, (const Node* []) { hi, int64_literal(bb->arena, 32) });
    // Merge the two
    return gen_primop_ce(bb, or_op, 2, (const Node* []) { lo, hi });
}

const Node* gen_load(BodyBuilder* instructions, const Node* ptr) {
    return gen_primop_ce(instructions, load_op, 1, (const Node* []) { ptr });
}

void gen_store(BodyBuilder* instructions, const Node* ptr, const Node* value) {
    gen_primop_c(instructions, store_op, 2, (const Node* []) { ptr, value });
}

const Node* gen_lea(BodyBuilder* instructions, const Node* base, const Node* offset, Nodes selectors) {
    LARRAY(const Node*, ops, 2 + selectors.count);
    ops[0] = base;
    ops[1] = offset;
    for (size_t i = 0; i < selectors.count; i++)
        ops[2 + i] = selectors.nodes[i];
    return gen_primop_ce(instructions, lea_op, 2 + selectors.count, ops);
}

const Node* find_or_process_decl(Rewriter* rewriter, Module* mod, const char* name) {
    Nodes old_decls = get_module_declarations(mod);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (strcmp(get_decl_name(decl), name) == 0) {
            return rewrite_node(rewriter, decl);
        }
    }
    assert(false);
}

const Node* access_decl(Rewriter* rewriter, Module* mod, const char* name) {
    const Node* decl = find_or_process_decl(rewriter, mod, name);
    if (decl->tag == Function_TAG)
        return fn_addr(rewriter->dst_arena, (FnAddr) { .fn = decl });
    else
        return ref_decl(rewriter->dst_arena, (RefDecl) { .decl = decl });
}
