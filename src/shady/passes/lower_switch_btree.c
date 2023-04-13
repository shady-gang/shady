#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../type.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;

    const Node* inspectee;
    const Node* run_default_case;
    Nodes yield_types;
} Context;

typedef struct TreeNode_ TreeNode;

struct TreeNode_ {
    TreeNode* children[2];
    int depth;
    uint64_t key;
    const Node* lam;
};

const Node* find(TreeNode* tree, uint64_t value) {
    if (tree->key == value)
        return tree->lam;
    else if (value < tree->key && tree->children[0])
        return find(tree->children[0], value);
    else if (value > tree->key && tree->children[1])
        return find(tree->children[1], value);
    return NULL;
}

TreeNode* skew(TreeNode* t) {
    if (t->children[0] == NULL)
        return t;
    else if (t->children[0]->depth == t->depth) {
        TreeNode* l = t->children[0];
        t->children[0] = t->children[1];
        l->children[1] = t;
        return l;
    } else
        return t;
}

TreeNode* split(TreeNode* t) {
    if (!t->children[1] || !t->children[1]->children[1])
        return t;
    else if (t->depth == t->children[1]->children[1]->depth) {
        TreeNode* r = t->children[1];
        t->children[1] = r->children[0];
        r->children[0] = t;
        r->depth += 1;
        return r;
    } else
        return t;
}

TreeNode* insert(TreeNode* t, TreeNode* x) {
    if (t == NULL) {
        x->depth = 1;
        return x;
    } else if (x->key < t->key) {
        t->children[0] = insert(t->children[0], x);
    } else if (x->key > t->key) {
        t->children[1] = insert(t->children[1], x);
    } else
        assert(false);

    t = skew(t);
    t = split(t);

    return t;
}

static const Node* generate_default_fallback_case(Context* ctx) {
    IrArena* a = ctx->rewriter.dst_arena;
    BodyBuilder* bb = begin_body(a);
    gen_store(bb, ctx->run_default_case, true_lit(a));
    LARRAY(const Node*, undefs, ctx->yield_types.count);
    for (size_t i = 0; i < ctx->yield_types.count; i++)
        undefs[i] = undef(a, (Undef) { .type = ctx->yield_types.nodes[i] });
    return lambda(a, empty(a), finish_body(bb, yield(a, (Yield) { .args = nodes(a, ctx->yield_types.count, undefs) })));
}

static const Node* wrap_instr_in_lambda(const Node* instr) {
    IrArena* a = instr->arena;
    BodyBuilder* bb = begin_body(a);
    Nodes values = bind_instruction(bb, instr);
    return lambda(a, empty(a), finish_body(bb, yield(a, (Yield) { .args = values })));
}

static const Node* generate_decision_tree(Context* ctx, TreeNode* n, uint64_t min, uint64_t max) {
    IrArena* a = ctx->rewriter.dst_arena;
    assert(n->key >= min && n->key <= max);
    assert(n->lam);

    // instruction in case we match
    const Node* body = n->lam;

    const Type* inspectee_t = ctx->inspectee->type;
    deconstruct_qualified_type(&inspectee_t);
    assert(inspectee_t->tag == Int_TAG);

    const Node* pivot = int_literal(a, (IntLiteral) { .width = inspectee_t->payload.int_type.width, .is_signed = inspectee_t->payload.int_type.is_signed, .value.u64 = n->key });

    if (min < n->key) {
        BodyBuilder* bb = begin_body(a);
        const Node* instr = if_instr(a, (If) {
            .yield_types = ctx->yield_types,
            .condition = gen_primop_e(bb, lt_op, empty(a), mk_nodes(a, ctx->inspectee, pivot)),
            .if_true = n->children[0] ? generate_decision_tree(ctx, n->children[0], min, n->key - 1) : generate_default_fallback_case(ctx),
            .if_false = body,
        });
        Nodes values = bind_instruction(bb, instr);
        body = lambda(a, empty(a), finish_body(bb, yield(a, (Yield) { .args = values })));
    }

    if (max > n->key) {
        BodyBuilder* bb = begin_body(a);
        const Node* instr = if_instr(a, (If) {
            .yield_types = ctx->yield_types,
            .condition = gen_primop_e(bb, gt_op, empty(a), mk_nodes(a, ctx->inspectee, pivot)),
            .if_true = n->children[1] ? generate_decision_tree(ctx, n->children[1], n->key + 1, max) : generate_default_fallback_case(ctx),
            .if_false = body,
        });
        Nodes values = bind_instruction(bb, instr);
        body = lambda(a, empty(a), finish_body(bb, yield(a, (Yield) { .args = values })));
    }

    return body;
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case Match_TAG: {
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, node->payload.match_instr.yield_types);
            Nodes literals = rewrite_nodes(&ctx->rewriter, node->payload.match_instr.literals);
            Nodes cases = rewrite_nodes(&ctx->rewriter, node->payload.match_instr.cases);

            // TODO handle degenerate no-cases case ?
            // TODO or maybe do that in fold()
            assert(cases.count > 0);

            Arena* arena = new_arena();
            TreeNode* root = NULL;
            for (size_t i = 0; i < literals.count; i++) {
                TreeNode* t = arena_alloc(arena, sizeof(TreeNode));
                t->key = get_int_literal_value(literals.nodes[i], false);
                t->lam = cases.nodes[i];
                root = insert(root, t);
            }

            BodyBuilder* bb = begin_body(a);
            const Node* run_default_case = gen_primop_e(bb, alloca_logical_op, singleton(bool_type(a)), empty(a));
            gen_store(bb, run_default_case, false_lit(a));

            Context ctx2 = *ctx;
            ctx2.run_default_case = run_default_case;
            ctx2.yield_types = yield_types;
            ctx2.inspectee = rewrite_node(&ctx->rewriter, node->payload.match_instr.inspect);
            Nodes matched_results = bind_instruction(bb, block(a, (Block) { .yield_types = add_qualifiers(a, ctx2.yield_types, false), .inside = generate_decision_tree(&ctx2, root, 0, UINT64_MAX) }));

            // Check if we need to run the default case
            Nodes final_results = bind_instruction(bb, if_instr(a, (If) {
                .yield_types = ctx2.yield_types,
                .condition = gen_load(bb, run_default_case),
                .if_true = rewrite_node(&ctx->rewriter, node->payload.match_instr.default_case),
                .if_false = lambda(a, empty(a), yield(a, (Yield) {
                    .args = matched_results,
                }))
            }));

            destroy_arena(arena);
            return yield_values_and_wrap_in_block(bb, final_results);
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_switch_btree(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
