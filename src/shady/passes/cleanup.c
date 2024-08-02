#include "pass.h"

#include "../analysis/uses.h"
#include "../ir_private.h"
#include "../type.h"

#include "portability.h"
#include "log.h"

#pragma GCC diagnostic error "-Wswitch"

typedef struct {
    Rewriter rewriter;
    const UsesMap* map;
    bool* todo;
} Context;

static size_t count_calls(const UsesMap* map, const Node* bb) {
    size_t count = 0;
    const Use* use = get_first_use(map, bb);
    for (;use; use = use->next_use) {
        if (use->user->tag == Jump_TAG) {
            const Use* jump_use = get_first_use(map, use->user);
            for (; jump_use; jump_use = jump_use->next_use) {
                if (jump_use->operand_class == NcJump)
                    return SIZE_MAX; // you can never inline conditional jumps
                count++;
            }
        }
    }
    return count;
}

Nodes add_structured_construct(BodyBuilder* bb, Nodes params, Structured_constructTag tag, union NodesUnion payload);

static void reset_params(Nodes params) {
    for (size_t i = 0; i < params.count; i++)
        ((Node*) params.nodes[i])->payload.param.abs = NULL;
}

// eliminates blocks by "lifting" their contents out and replacing yield with the tail of the outer let
// In other words, we turn these patterns:
//
// let block {
//   let I in case(x) =>
//   let J in case(y) =>
//   let K in case(z) =>
//      ...
//   yield (x, y, z) }
// in case(a, b, c) => R
//
// into these:
//
// let I in case(x) =>
// let J in case(y) =>
// let K in case(z) =>
// ...
// R[a->x, b->y, c->z]
const Node* flatten_block(IrArena* arena, const Node* instruction, BodyBuilder* bb) {
    assert(instruction->tag == Block_TAG);
    // follow the terminator of the block until we hit a yield()
    const Node* const lam = instruction->payload.block.inside;
    assert(is_case(lam));
    const Node* terminator = get_abstraction_body(lam);
    while (true) {
        if (is_structured_construct(terminator)) {
            Nodes params = get_abstraction_params(get_structured_construct_tail(terminator));
            reset_params(params);
            add_structured_construct(bb, params, (Structured_constructTag) terminator->tag, terminator->payload);
            terminator = get_abstraction_body(get_structured_construct_tail(terminator));
            continue;
        }

        switch (is_terminator(terminator)) {
            case NotATerminator: assert(false);
            case Terminator_Let_TAG: {
                add_structured_construct(bb, empty(arena), (Structured_constructTag) NotAStructured_construct, terminator->payload);
                terminator = terminator->payload.let.in;
                continue;
            }
            case Terminator_BlockYield_TAG: {
                return maybe_tuple_helper(arena, terminator->payload.block_yield.args);
            }
            case Terminator_Return_TAG:
            case Terminator_TailCall_TAG: {
                return terminator;
            }
            // if we see anything else, give up
            default: {
                assert(false && "invalid block");
            }
        }
    }
}

static bool has_side_effects(const Node* instr) {
    bool side_effects = true;
    if (instr->tag == PrimOp_TAG)
        side_effects = has_primop_got_side_effects(instr->payload.prim_op.op);
    switch (instr->tag) {
        case Load_TAG: return false;
        default: break;
    }
    return side_effects;
}

const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    if (old->tag == Function_TAG || old->tag == Constant_TAG) {
        Context c = *ctx;
        c.map = create_uses_map(old, NcType | NcDeclaration);
        const Node* new = recreate_node_identity(&c.rewriter, old);
        destroy_uses_map(c.map);
        return new;
    }

    switch (old->tag) {
        case Let_TAG: {
            Let payload = old->payload.let;
            bool consumed = false;
            Nodes result_types = unwrap_multiple_yield_types(a, payload.instruction->type);
            for (size_t i = 0; i < result_types.count; i++) {
                const Use* use = get_first_use(ctx->map, extract_multiple_ret_types_helper(payload.instruction, i));
                assert(use);
                for (;use; use = use->next_use) {
                    if (use->user == old)
                        continue;
                    consumed = true;
                    break;
                }
                if (consumed)
                    break;
            }
            if (!consumed && !has_side_effects(payload.instruction) && ctx->rewriter.dst_arena) {
                debugvv_print("Cleanup: found an unused instruction: ");
                log_node(DEBUGVV, payload.instruction);
                debugvv_print("\n");
                *ctx->todo = true;
                return rewrite_node(&ctx->rewriter, payload.in);
            }

            BodyBuilder* bb = begin_body(a);
            const Node* oinstruction = old->payload.let.instruction;
            const Node* instruction;
            // optimization: fold blocks
            if (oinstruction->tag == Block_TAG) {
                *ctx->todo = true;
                instruction = flatten_block(a, recreate_node_identity(r, oinstruction), bb);
                register_processed(r, oinstruction, instruction);
                if (is_terminator(instruction))
                    return finish_body(bb, instruction);
            } else {
                instruction = rewrite_node(r, oinstruction);
                register_processed(r, oinstruction, instruction);
            }
            const Node* nlet = let(a, instruction, rewrite_node(r, old->payload.let.in));
            return finish_body(bb, nlet);
        }
        case BasicBlock_TAG: {
            size_t uses = count_calls(ctx->map, old);
            if (uses <= 1) {
                log_string(DEBUGVV, "Eliminating basic block '%s' since it's used only %d times.\n", get_abstraction_name(old), uses);
                return NULL;
            }
            break;
        }
        case Jump_TAG: {
            const Node* otarget = old->payload.jump.target;
            const Node* ntarget = rewrite_node(r, otarget);
            if (!ntarget) {
                // it's been inlined away! just steal the body
                Nodes nargs = rewrite_nodes(r, old->payload.jump.args);
                register_processed_list(r, get_abstraction_params(otarget), nargs);
                return rewrite_node(r, get_abstraction_body(otarget));
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

OptPass simplify;

bool simplify(SHADY_UNUSED const CompilerConfig* config, Module** m) {
    Module* src = *m;

    IrArena* a = get_module_arena(src);
    *m = new_module(a, get_module_name(*m));
    bool todo = false;
    Context ctx = { .todo = &todo };
    ctx.rewriter = create_node_rewriter(src, *m, (RewriteNodeFn) process),
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return todo;
}

OptPass opt_demote_alloca;
RewritePass import;

Module* cleanup(SHADY_UNUSED const CompilerConfig* config, Module* const src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    if (!aconfig.check_types)
        return src;
    bool todo;
    size_t r = 0;
    Module* m = src;
    do {
        debugv_print("Cleanup round %d\n", r);
        if (getenv("SHADY_DUMP_CLEAN_ROUNDS"))
            log_module(DEBUGVV, config, m);
        todo = false;
        todo |= opt_demote_alloca(config, &m);
        if (getenv("SHADY_DUMP_CLEAN_ROUNDS"))
            log_module(DEBUGVV, config, m);
        todo |= simplify(config, &m);
        r++;
    } while (todo);
    return import(config, m);
}
