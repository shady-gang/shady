#include "shady/pass.h"
#include "shady/analysis/uses.h"

#include "analysis/leak.h"
#include "ir_private.h"

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
    const Use* use = shd_get_first_use(map, bb);
    for (; use; use = use->next_use) {
        if (use->user->tag == Jump_TAG) {
            const Use* jump_use = shd_get_first_use(map, use->user);
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
    const Use* use = shd_get_first_use(map, instr);
    for (; use; use = use->next_use) {
        if (use->operand_class == NcValue)
            return true;
    }
    return false;
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    if (old->tag == Function_TAG || old->tag == Constant_TAG) {
        Context c = *ctx;
        c.map = shd_new_uses_map_fn(old, NcType | NcFunction);
        const Node* new = shd_recreate_node(&c.rewriter, old);
        shd_destroy_uses_map(c.map);
        return new;
    }

    switch (old->tag) {
        case BasicBlock_TAG: {
            size_t uses = count_calls(ctx->map, old);
            if (uses <= 1 && a->config.optimisations.inline_single_use_bbs) {
                shd_log_fmt(DEBUGVV, "Eliminating basic block '%s' since it's used only %d times.\n", shd_get_node_name_safe(old), uses);
                *ctx->todo = true;
                return NULL;
            }
            break;
        }
        case Jump_TAG: {
            const Node* otarget = old->payload.jump.target;
            const Node* ntarget = shd_rewrite_node(r, otarget);
            if (!ntarget) {
                // it's been inlined away! just steal the body
                Nodes nargs = shd_rewrite_nodes(r, old->payload.jump.args);
                shd_register_processed_list(r, get_abstraction_params(otarget), nargs);
                shd_register_processed(r, shd_get_abstraction_mem(otarget), shd_rewrite_node(r, old->payload.jump.mem));
                return shd_rewrite_node(r, get_abstraction_body(otarget));
            }
            break;
        }
        case Control_TAG: {
            Control payload = old->payload.control;
            if (shd_is_control_static(ctx->map, old)) {
                const Node* control_inside = payload.inside;
                const Node* term = get_abstraction_body(control_inside);
                if (term->tag == Join_TAG) {
                    Join payload_join = term->payload.join;
                    if (payload_join.join_point == shd_first(get_abstraction_params(control_inside))) {
                        // if we immediately consume the join point and it's never leaked, this control block does nothing and can be eliminated
                        shd_register_processed(r, shd_get_abstraction_mem(control_inside), shd_rewrite_node(r, payload.mem));
                        shd_register_processed(r, control_inside, NULL);
                        *ctx->todo = true;
                        return shd_rewrite_node(r, term);
                    }
                }
            }
            const Use* use = shd_get_first_use(ctx->map, shd_first(get_abstraction_params(payload.inside)));
            bool used_at_all = false;
            for (;use; use = use->next_use) {
                if (use->user == payload.inside) {
                    continue;
                }

                used_at_all = true;
            }
            if (!used_at_all) {
                *ctx->todo = true;
                const Node* control_inside = payload.inside;
                shd_register_processed(r, shd_get_abstraction_mem(control_inside), shd_rewrite_node(r, payload.mem));
                shd_register_processed(r, control_inside, NULL);
                return shd_rewrite_node(r, get_abstraction_body(control_inside));
            }
            break;
        }
        case Join_TAG: {
            Join payload = old->payload.join;
            const Node* control = shd_get_control_for_jp(ctx->map, payload.join_point);
            if (control) {
                Control old_control_payload = control->payload.control;
                // there was a control but now there is not anymore - jump to the tail!
                if (shd_rewrite_node(r, old_control_payload.inside) == NULL) {
                    return jump_helper(a, shd_rewrite_node(r, payload.mem), shd_rewrite_node(r, old_control_payload.tail),
                                       shd_rewrite_nodes(r, payload.args));
                }
            }
            break;
        }
        case Load_TAG: {
            if (!is_used_as_value(ctx->map, old))
                return shd_rewrite_node(r, old->payload.load.mem);
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

OptPass shd_opt_simplify;

bool shd_opt_simplify(SHADY_UNUSED const CompilerConfig* config, Module** m) {
    Module* src = *m;

    IrArena* a = shd_module_get_arena(src);
    *m = shd_new_module(a, shd_module_get_name(*m));
    bool todo = false;
    Context ctx = { .todo = &todo };
    ctx.rewriter = shd_create_node_rewriter(src, *m, (RewriteNodeFn) process);
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return todo;
}

OptPass shd_opt_demote_alloca;
OptPass shd_opt_mem2reg;

Module* shd_cleanup(const CompilerConfig* config, SHADY_UNUSED void* unused, Module* const src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    if (!aconfig.check_types)
        return src;
    bool todo;
    size_t r = 0;
    Module* m = src;
    bool changed_at_all = false;
    do {
        todo = false;
        shd_log_fmt(DEBUGV, "Cleanup round %d\n", r);

        APPLY_OPT(shd_opt_demote_alloca);
        APPLY_OPT(shd_opt_mem2reg);
        APPLY_OPT(shd_opt_simplify);

        changed_at_all |= todo;

        r++;
    } while (todo);
    if (changed_at_all)
        shd_debugv_print("After %d rounds of cleanup:\n", r);

    return shd_import(config, m);
}
