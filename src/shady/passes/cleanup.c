#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../analysis/uses.h"

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

const Node* process(Context* ctx, const Node* old) {
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
            const Node* tail_case = payload.tail;
            Nodes tail_params = get_abstraction_params(tail_case);
            for (size_t i = 0; i < tail_params.count; i++) {
                const Use* use = get_first_use(ctx->map, tail_params.nodes[i]);
                assert(use);
                for (;use; use = use->next_use) {
                    if (use->user == tail_case)
                        continue;
                    consumed = true;
                    break;
                }
                if (consumed)
                    break;
            }
            if (!consumed && !side_effects && ctx->rewriter.dst_arena) {
                debug_print("Cleanup: found an unused instruction: ");
                log_node(DEBUG, payload.instruction);
                debug_print("\n");
                *ctx->todo = true;
                return rewrite_node(&ctx->rewriter, get_abstraction_body(tail_case));
            }
            break;
        }
        case BasicBlock_TAG: {
            size_t uses = count_calls(ctx->map, old);
            if (uses <= 1) {
                log_string(DEBUGV, "Eliminating basic block '%s' since it's used only %d times.\n", get_abstraction_name(old), uses);
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

    return recreate_node_identity(&ctx->rewriter, old);;
}

Module* cleanup(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    if (!aconfig.check_types)
        return src;
    IrArena* a = new_ir_arena(aconfig);
    bool todo;
    Context ctx = { .todo = &todo };
    size_t r = 0;
    Module* m;
    do {
        debug_print("Cleanup round %d\n", r);
        todo = false;
        m = new_module(a, get_module_name(src));
        ctx.rewriter = create_rewriter(src, m, (RewriteNodeFn) process),
        rewrite_module(&ctx.rewriter);
        destroy_rewriter(&ctx.rewriter);
        src = m;
        r++;
    } while (todo);
    return m;
}
