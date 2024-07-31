#include "pass.h"

#include "portability.h"
#include "log.h"

typedef struct StackState_ StackState;
struct StackState_ {
    StackState* prev;
    enum { VALUE, MERGE } type;
    bool leaks;
    const Node* value;
    size_t count;
    const Node** values;
};

typedef struct {
    Rewriter rewriter;
    StackState* state;
} Context;

static void tag_leaks(Context* ctx) {
    StackState* s = ctx->state;
    while (s) {
        s->leaks = true;
        s = s->prev;
    }
}

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    StackState entry;
    Context child_ctx = *ctx;

    bool is_push = false;
    bool is_pop = false;

    switch (is_terminator(node)) {
        case Terminator_Unreachable_TAG: break;
        case Let_TAG: {
            const Node* old_instruction = node->payload.let.instruction;
            const Node* ntail = NULL;
            switch (is_instruction(old_instruction)) {
                case Instruction_PushStack_TAG: {
                    const Node* value = rewrite_node(&ctx->rewriter, old_instruction->payload.push_stack.value);
                    entry = (StackState) {
                        .prev = ctx->state,
                        .type = VALUE,
                        .value = value,
                        .leaks = false,
                    };
                    child_ctx.state = &entry;
                    is_push = true;
                    break;
                }
                case Instruction_PopStack_TAG: {
                    if (ctx->state) {
                        child_ctx.state = ctx->state->prev;
                        is_pop = true;
                    }
                    break;
                }
                case Instruction_Block_TAG:
                // Leaf calls and indirect calls are not analysed and so they are considered to leak the state
                // we also need to forget our information about the current state
                case Instruction_Call_TAG: {
                    tag_leaks(ctx);
                    child_ctx.state = NULL;
                    break;
                }
                default: break;
                case NotAnInstruction: assert(false);
            }

            ntail = rewrite_node(&child_ctx.rewriter, node->payload.let.tail);

            const Node* ninstruction = NULL;
            if (is_push && !child_ctx.state->leaks) {
                // replace stack pushes with no-ops
                ninstruction = quote_helper(a, empty(a));
            } else if (is_pop) {
                assert(ctx->state->type == VALUE);
                const Node* value = ctx->state->value;
                ninstruction = quote_helper(a, singleton(value));
            } else {
                // if the stack state is observed, or this was an unrelated instruction, leave it alone
                ninstruction = recreate_node_identity(&ctx->rewriter, old_instruction);
            }
            assert(ninstruction);
            return let(a, ninstruction, ntail);
        }
        // Unreachable is assumed to never happen, so it doesn't observe the stack state
        case NotATerminator: break;
        default: {
            // All other non-let terminators are considered to leak the stack state
            tag_leaks(ctx);
            break;
        }
    }

    // child_ctx.state = NULL;
    switch (node->tag) {
        case Function_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            child_ctx.state = NULL;
            recreate_decl_body_identity(&child_ctx.rewriter, node, fun);
            return fun;
        }
        default: return recreate_node_identity(&child_ctx.rewriter, node);
    }
}

Module* opt_stack(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .state = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
