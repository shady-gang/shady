#include "shady/ir.h"

#include "../log.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"
#include "../portability.h"
#include "../analysis/scope.h"

#include "list.h"
#include "dict.h"

#include <string.h>
#include <assert.h>

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

typedef size_t CaseId;

typedef struct {
    CaseId case_id;
    const CFNode* cf_node;
} BBMeta;

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static CaseId find_bb_case_id(struct Dict* bbs, const Node* bb) {
    BBMeta* found = find_value_dict(const Node*, BBMeta, bbs, bb);
    if (!found) error("missing case id for BB")
    return found->case_id;
}

static const Node* handle_basic_block(Context* ctx, const Node* next_bb, struct Dict* bbs, const Node* node) {
    assert(node->tag == Function_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    const Block* old_block = &node->payload.fn.block->payload.block;

    //Node* fun = fn(dst_arena, node->payload.fn.atttributes, node->payload.fn.name, nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL));
    Instructions instructions = begin_instructions(dst_arena);
    for (size_t i = 0; i < old_block->instructions.count; i++) {
        append_instr(instructions, recreate_node_identity(&ctx->rewriter, old_block->instructions.nodes[i]));
    }

    const Node* terminator = NULL;
    switch (old_block->terminator->tag) {
        case Jump_TAG: {
            // TODO BB arguments
            assert(old_block->terminator->payload.jump.args.count == 0);
            CaseId target_id = find_bb_case_id(bbs, old_block->terminator->payload.jump.target);
            gen_store(instructions, next_bb, int_literal(dst_arena, (IntLiteral) {.value = target_id}));
            terminator = merge_construct(dst_arena, (MergeConstruct) {
                .args = nodes(dst_arena, 0, NULL),
                .construct = Continue,
            });
            break;
        }
        case Branch_TAG: {
            // TODO BB arguments
            assert(old_block->terminator->payload.branch.args.count == 0);
            CaseId true_id = find_bb_case_id(bbs, old_block->terminator->payload.branch.true_target);
            CaseId false_id = find_bb_case_id(bbs, old_block->terminator->payload.branch.false_target);
            const Node* new_cond = rewrite_node(&ctx->rewriter, old_block->terminator->payload.branch.condition);
            const Node* selected_target = gen_primop(instructions, (PrimOp) {
                .op = select_op,
                .operands = nodes(dst_arena, 3, (const Node* []) { new_cond, int_literal(dst_arena, (IntLiteral) {.value = true_id}), int_literal(dst_arena, (IntLiteral) {.value = false_id})})
            }).nodes[0];
            gen_store(instructions, next_bb, selected_target);
            terminator = merge_construct(dst_arena, (MergeConstruct) {
                .args = nodes(dst_arena, 0, NULL),
                .construct = Continue,
            });
            break;
        }
        case Unreachable_TAG:
        case Return_TAG: terminator = recreate_node_identity(&ctx->rewriter, old_block->terminator); break;
        default: error("Unhandled terminator");
    }

    return block(dst_arena, (Block) {
        .instructions = finish_instructions(instructions),
        .terminator = terminator
    });
}

static void handle_function(Context* ctx, const Node* node, Node* new) {
    assert(node->tag == Function_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    Scope scope = build_scope(node);

    if (scope.size == 1) {
        dispose_scope(&scope);
        recreate_decl_body_identity(&ctx->rewriter, node, new);
        return;
    }
    
    struct Dict* bbs = new_dict(const Node*, BBMeta,  (HashFn) hash_node, (CmpFn) compare_node);

    for (size_t i = 0; i < new->payload.fn.params.count; i++)
        register_processed(&ctx->rewriter, node->payload.fn.params.nodes[i], new->payload.fn.params.nodes[i]);

    // Reserve and assign IDs for basic blocks within this
    LARRAY(const Node*, literals, scope.size);
    LARRAY(const Node*, cases, scope.size);
    for (size_t i = 0; i < scope.size; i++) {
        CFNode* bb = read_list(CFNode*, scope.contents)[i];

        BBMeta bb_ectx = {
            .case_id = (CaseId) i,
            .cf_node = bb
        };
        insert_dict_and_get_result(struct Node*, BBEmissionCtx, bbs, bb->node, bb_ectx);
        literals[i] = int_literal(dst_arena, (IntLiteral) {.value = i});
    }

    Instructions body_instructions = begin_instructions(dst_arena);
    const Node* next_bb = gen_primop(body_instructions, (PrimOp) {
        .op = alloca_op,
        .operands = nodes(dst_arena, 1, (const Node* []) { int_type(dst_arena) })
    }).nodes[0];
    gen_store(body_instructions, next_bb, int_literal(dst_arena, (IntLiteral) { .value = 0 }));

    for (size_t i = 0; i < scope.size; i++) {
        CFNode* bb = read_list(CFNode*, scope.contents)[i];
        cases[i] = handle_basic_block(ctx, next_bb, bbs, bb->node);
    }

    Instructions loop_instructions = begin_instructions(dst_arena);
    const Node* next_bb_loaded = gen_load(loop_instructions, next_bb);
    const Node* match_i = match_instr(dst_arena, (Match) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .inspect = next_bb_loaded,
        .default_case = block(dst_arena, (Block) {
            .instructions = nodes(dst_arena, 0, NULL),
            .terminator = unreachable(dst_arena)
        }),
        .literals = nodes(dst_arena, scope.size, literals),
        .cases = nodes(dst_arena, scope.size, cases),
    });

    append_instr(loop_instructions, match_i);

    const Node* loop_body = block(dst_arena, (Block) {
        .instructions = finish_instructions(loop_instructions),
        .terminator = unreachable(dst_arena)
    });

    const Node* the_loop = loop_instr(dst_arena, (Loop) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .params = nodes(dst_arena, 0, NULL),
        .initial_args = nodes(dst_arena, 0, NULL),
        .body = loop_body
    });

    append_instr(body_instructions, the_loop);

    new->payload.fn.block = block(dst_arena, (Block) {
        .instructions = finish_instructions(body_instructions),
        .terminator = unreachable(dst_arena),
    });

    dispose_scope(&scope);
    destroy_dict(bbs);
}

static const Node* process_node(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    switch (old->tag) {
        case GlobalVariable_TAG:
        case Constant_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            recreate_decl_body_identity(&ctx->rewriter, old, new);
            return new;
        }
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            // leave this out - we rebuild functions afterwards
            return new;
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}


const Node* lower_jumps_loop(CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) process_node,
            .rewrite_decl_body = NULL,
            .processed = done,
        },

        .config = config,
    };

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);
    for (size_t i = 0; i < rewritten->payload.root.declarations.count; i++) {
        const Node* old_decl = src_program->payload.root.declarations.nodes[i];
        if (old_decl->tag != Function_TAG) continue;

        Node* new_decl = (Node*) rewritten->payload.root.declarations.nodes[i];
        if (strcmp(old_decl->payload.fn.name, "top_dispatcher") == 0)
            recreate_decl_body_identity(&ctx.rewriter, old_decl, new_decl);
        else
            handle_function(&ctx, old_decl, new_decl);
    }

    destroy_dict(done);
    return rewritten;
}
