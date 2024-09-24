#include "shady/pass.h"

#include "../analysis/uses.h"
#include "../analysis/leak.h"
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
    for (; use; use = use->next_use) {
        if (use->user->tag == Jump_TAG) {
            const Use* jump_use = get_first_use(map, use->user);
            for (; jump_use; jump_use = jump_use->next_use) {
                if (jump_use->operand_class == NcJump)
                    return SIZE_MAX; // you can never inline conditional jumps
                count++;
            }
        } else if (use->operand_class == NcBasic_block)
            return SIZE_MAX; // you can never inline basic blocks used for other purposes
    }
    return count;
}

static bool is_used_as_value(const UsesMap* map, const Node* instr) {
    const Use* use = get_first_use(map, instr);
    for (; use; use = use->next_use) {
        if (use->operand_class == NcValue)
            return true;
    }
    return false;
}

const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    if (old->tag == Function_TAG || old->tag == Constant_TAG) {
        Context c = *ctx;
        c.map = create_fn_uses_map(old, NcType | NcDeclaration);
        const Node* new = recreate_node_identity(&c.rewriter, old);
        destroy_uses_map(c.map);
        return new;
    }

    switch (old->tag) {
        case BasicBlock_TAG: {
            size_t uses = count_calls(ctx->map, old);
            if (uses <= 1 && a->config.optimisations.inline_single_use_bbs) {
                log_string(DEBUGVV, "Eliminating basic block '%s' since it's used only %d times.\n", get_abstraction_name_safe(old), uses);
                *ctx->todo = true;
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
                register_processed(r, get_abstraction_mem(otarget), rewrite_node(r, old->payload.jump.mem));
                return rewrite_node(r, get_abstraction_body(otarget));
            }
            break;
        }
        case Control_TAG: {
            Control payload = old->payload.control;
            if (is_control_static(ctx->map, old)) {
                const Node* control_inside = payload.inside;
                const Node* term = get_abstraction_body(control_inside);
                if (term->tag == Join_TAG) {
                    Join payload_join = term->payload.join;
                    if (payload_join.join_point == first(get_abstraction_params(control_inside))) {
                        // if we immediately consume the join point and it's never leaked, this control block does nothing and can be eliminated
                        register_processed(r, get_abstraction_mem(control_inside), rewrite_node(r, payload.mem));
                        register_processed(r, control_inside, NULL);
                        *ctx->todo = true;
                        return rewrite_node(r, term);
                    }
                }
            }
            break;
        }
        case Join_TAG: {
            Join payload = old->payload.join;
            const Node* control = get_control_for_jp(ctx->map, payload.join_point);
            if (control) {
                Control old_control_payload = control->payload.control;
                // there was a control but now there is not anymore - jump to the tail!
                if (rewrite_node(r, old_control_payload.inside) == NULL) {
                    return jump_helper(a, rewrite_node(r, old_control_payload.tail), rewrite_nodes(r, payload.args), rewrite_node(r, payload.mem));
                }
            }
            break;
        }
        case Load_TAG: {
            if (!is_used_as_value(ctx->map, old))
                return rewrite_node(r, old->payload.load.mem);
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
    ctx.rewriter = create_node_rewriter(src, *m, (RewriteNodeFn) process);
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return todo;
}

OptPass opt_demote_alloca;
OptPass opt_mem2reg;
RewritePass import;

Module* cleanup(const CompilerConfig* config, Module* const src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    if (!aconfig.check_types)
        return src;
    bool todo;
    size_t r = 0;
    Module* m = src;
    bool changed_at_all = false;
    do {
        todo = false;
        debugv_print("Cleanup round %d\n", r);

        APPLY_OPT(opt_demote_alloca);
        APPLY_OPT(opt_mem2reg);
        APPLY_OPT(simplify);

        changed_at_all |= todo;

        r++;
    } while (todo);
    if (changed_at_all)
        debugv_print("After %d rounds of cleanup:\n", r);
    return import(config, m);
}
