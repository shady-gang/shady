#include "pass.h"

#include "../analysis/uses.h"
#include "../ir_private.h"

#include "portability.h"
#include "log.h"

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

void bind_variables2(BodyBuilder* bb, Nodes vars, const Node* instr);

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
        switch (is_terminator(terminator)) {
            case NotATerminator: assert(false);
            case Terminator_Let_TAG: {
                bind_variables2(bb, terminator->payload.let.variables, terminator->payload.let.instruction);
                terminator = get_abstraction_body(terminator->payload.let.tail);
                continue;
            }
            case Terminator_Yield_TAG: {
                return quote_helper(arena, terminator->payload.yield.args);
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

const Node* process(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
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
            bool side_effects = true;
            if (payload.instruction->tag == PrimOp_TAG)
                side_effects = has_primop_got_side_effects(payload.instruction->payload.prim_op.op);
            bool consumed = false;
            Nodes vars = payload.variables;
            for (size_t i = 0; i < vars.count; i++) {
                const Use* use = get_first_use(ctx->map, vars.nodes[i]);
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
            if (!consumed && !side_effects && ctx->rewriter.dst_arena) {
                debugvv_print("Cleanup: found an unused instruction: ");
                log_node(DEBUGVV, payload.instruction);
                debugvv_print("\n");
                *ctx->todo = true;
                return rewrite_node(&ctx->rewriter, get_abstraction_body(payload.tail));
            }

            BodyBuilder* bb = begin_body(a);
            const Node* instruction = rewrite_node(r, old->payload.let.instruction);
            // optimization: fold blocks
            if (instruction->tag == Block_TAG) {
                *ctx->todo = true;
                instruction = flatten_block(a, instruction, bb);
                if (is_terminator(instruction))
                    return finish_body(bb, instruction);
            }
            Nodes ovars = old->payload.let.variables;
            // optimization: eliminate unecessary quotes by rewriting variables into their values directly
            if (instruction->tag == PrimOp_TAG && instruction->payload.prim_op.op == quote_op) {
                *ctx->todo = true;
                register_processed_list(r, ovars, instruction->payload.prim_op.operands);
                return finish_body(bb, get_abstraction_body(rewrite_node(r, old->payload.let.tail)));
            }
            // rewrite variables now
            Nodes nvars = recreate_vars(a, ovars, instruction);
            register_processed_list(r, ovars, nvars);
            const Node* nlet = let(a, instruction, nvars, rewrite_node(r, old->payload.let.tail));
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
        todo = false;
        todo |= opt_demote_alloca(config, &m);
        todo |= simplify(config, &m);
        r++;
    } while (todo);
    return import(config, m);
}
