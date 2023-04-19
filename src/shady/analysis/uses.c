#include "uses.h"

#include "log.h"

#include <assert.h>
#include <stdlib.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct {
    ScopeUses* scope_uses;

    bool is_callee;

    CFNode* curr_abs;
    CFNode* use_location;

    CFNode* last_selection;
    CFNode* last_loop;
} VisitCtx;

// returns false if the use is dominated by the definition, otherwise true
static bool is_escaping_use(CFNode* definition, CFNode* use) {
    while (use) {
        if (use == definition)
            return false;
        use = use->idom;
    }
    return true;
}

static void visit_values(const VisitCtx*, Nodes nodes);
static void visit_domtree(const VisitCtx* ctx, CFNode* n);

static void visit_value(const VisitCtx* ctx, const Node* n) {
    switch (is_value(n)) {
        case NotAValue: break;
        case Value_Variable_TAG: {
            Uses* uses = *find_value_dict(const Node*, Uses*, ctx->scope_uses->map, n);
            assert(!uses->sealed);
            uses->uses_count++;
            uses->escapes_defining_block |= is_escaping_use(uses->defined, ctx->use_location);
            uses->in_non_callee_position |= !ctx->is_callee;
            break;
        }
        case Value_Composite_TAG: {
            visit_values(ctx, n->payload.composite.contents);
            break;
        }
        case Value_ConstrainedValue_TAG:
            visit_value(ctx, n->payload.constrained.value);
            break;
        case Value_UntypedNumber_TAG:
            break;
        case Value_IntLiteral_TAG:
            break;
        case Value_FloatLiteral_TAG:
            break;
        case Value_True_TAG:
            break;
        case Value_False_TAG:
            break;
        case Value_StringLiteral_TAG:
            break;
        case Value_Fill_TAG:
            visit_value(ctx, n->payload.fill.value);
            break;
        case Value_Undef_TAG:
            break;
        case Value_FnAddr_TAG:
            break;
        case Value_RefDecl_TAG:
            break;
        case Value_AntiQuote_TAG: error("Not handled");
    }
}

static void visit_values(const VisitCtx* ctx, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++)
        visit_value(ctx, nodes.nodes[i]);
}

static void visit_instruction(const VisitCtx* ctx, const Node* instruction) {
    switch (is_instruction(instruction)) {
        case NotAnInstruction: error("");
        case Instruction_PrimOp_TAG: {
            VisitCtx ctx2 = *ctx;
            switch (instruction->payload.prim_op.op) {
                // stores always leak everything
                case store_op: {
                    ctx2.use_location = NULL;
                    break;
                }
                default: break;
            }
            visit_values(ctx, instruction->payload.prim_op.operands);
            break;
        } case Instruction_Call_TAG:{
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = NULL;
            visit_values(&ctx2, instruction->payload.call.args);
            ctx2.is_callee = true;
            visit_value(&ctx2, instruction->payload.call.callee);
            break;
        } case Instruction_If_TAG: {
            visit_value(ctx, instruction->payload.if_instr.condition);
            VisitCtx ctx2 = *ctx;
            ctx2.last_selection = ctx2.curr_abs;
            visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, instruction->payload.if_instr.if_true));
            if (instruction->payload.if_instr.if_false)
                visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, instruction->payload.if_instr.if_false));
            break;
        } case Instruction_Match_TAG: {
            visit_value(ctx, instruction->payload.match_instr.inspect);
            VisitCtx ctx2 = *ctx;
            ctx2.last_selection = ctx2.curr_abs;
            visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, instruction->payload.match_instr.default_case));
            for (size_t i = 0; i < instruction->payload.match_instr.cases.count; i++)
                visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, instruction->payload.match_instr.cases.nodes[i]));
            break;
        } case Instruction_Loop_TAG: {
            visit_values(ctx, instruction->payload.loop_instr.initial_args);
            VisitCtx ctx2 = *ctx;
            ctx2.last_loop = ctx2.curr_abs;
            visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, instruction->payload.loop_instr.body));
            break;
        } case Instruction_Control_TAG: {
            VisitCtx ctx2 = *ctx;
            visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, instruction->payload.control.inside));
            break;
        } case Instruction_Block_TAG: {
            VisitCtx ctx2 = *ctx;
            ctx2.last_selection = ctx2.curr_abs;
            visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, instruction->payload.block.inside));
            break;
        } case Instruction_Comment_TAG:
            break;
    }
}

static void visit_terminator(const VisitCtx* ctx, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case Terminator_LetMut_TAG: error("")
        case NotATerminator: error("");
        case Terminator_Let_TAG: {
            visit_instruction(ctx, terminator->payload.let.instruction);
            VisitCtx ctx2 = *ctx;
            visit_domtree(&ctx2, scope_lookup(ctx->scope_uses->scope, terminator->payload.let.tail));
            break;
        }
        case Terminator_TailCall_TAG: {
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = NULL;
            visit_values(&ctx2, terminator->payload.tail_call.args);

            ctx2.use_location = ctx2.curr_abs;
            ctx2.is_callee = true;
            visit_value(&ctx2, terminator->payload.tail_call.target);
            break;
        } case Terminator_Jump_TAG: {
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = scope_lookup(ctx->scope_uses->scope, terminator->payload.jump.target);
            visit_values(&ctx2, terminator->payload.jump.args);
            break;
        } case Terminator_Branch_TAG: {
            VisitCtx ctx2 = *ctx;
            // there are effectively two uses of those guys, or more exactly, two places where they'll be used.
            // should this count as one use ? should this count as always leaking ? doesn't matter for now
            ctx2.use_location = scope_lookup(ctx->scope_uses->scope, terminator->payload.branch.true_target);
            visit_values(&ctx2, terminator->payload.branch.args);
            ctx2.use_location = scope_lookup(ctx->scope_uses->scope, terminator->payload.branch.false_target);
            visit_values(&ctx2, terminator->payload.branch.args);
            break;
        } case Terminator_Switch_TAG: {
            VisitCtx ctx2 = *ctx;
            // TODO: align with what we do for branches
            ctx2.use_location = NULL;
            visit_value(&ctx2, terminator->payload.br_switch.switch_value);
            visit_values(&ctx2, terminator->payload.br_switch.args);
            break;
        }
        case Terminator_MergeContinue_TAG: {
            // TODO track more precisely
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = NULL;
            visit_values(&ctx2, terminator->payload.merge_continue.args);
            break;
        }
        case Terminator_MergeBreak_TAG: {
            // TODO track more precisely
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = NULL;
            visit_values(&ctx2, terminator->payload.merge_continue.args);
            break;
        }
        case Terminator_Yield_TAG: {
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = ctx->last_selection;
            visit_values(&ctx2, terminator->payload.yield.args);
            break;
        } case Terminator_Join_TAG: {
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = NULL; // by default, we don't know where the join point goes
            const Node* jp = terminator->payload.join.join_point;
            Uses** jp_uses = find_value_dict(const Node*, Uses*, ctx->scope_uses->map, jp);
            if (jp_uses) {
                // it's a known variable! yay!
                assert(jp->tag == Variable_TAG);
                const Node* abs = jp->payload.var.abs;
                assert(abs);
                // check if it's a join point for a control body ...
                if (abs->tag == AnonLambda_TAG) {
                    const Node* structured_construct = abs->payload.anon_lam.structured_construct;
                    assert(structured_construct);
                    if (structured_construct->tag == Control_TAG) {
                        // ok cool it is. let's find who binds that body (it must be a Let node, by virtue of our grammar) and take that guy's tail
                        CFNode* control_body_n = scope_lookup(ctx->scope_uses->scope, abs);
                        assert(control_body_n);
                        CFNode* let_control_n = control_body_n->idom;
                        assert(let_control_n && get_abstraction_body(let_control_n->node)->tag == Let_TAG);
                        CFNode* tail_n = scope_lookup(ctx->scope_uses->scope, get_abstraction_body(let_control_n->node)->payload.let.tail);
                        assert(tail_n);
                        // this is our use location
                        ctx2.use_location = tail_n;
                    }
                }
            }
            visit_values(&ctx2, terminator->payload.join.args);
            ctx2.use_location = ctx->curr_abs;
            ctx2.is_callee = true;
            visit_value(&ctx2, jp);
            break;
        }
        case Terminator_Return_TAG: {
            VisitCtx ctx2 = *ctx;
            ctx2.use_location = NULL;
            visit_values(&ctx2, terminator->payload.fn_ret.args);
            break;
        }
        case Terminator_Unreachable_TAG:
            break;
    }
}

static void visit_domtree(const VisitCtx* ctx, CFNode* n) {
    VisitCtx sub_ctx = *ctx;
    ctx = &sub_ctx;
    sub_ctx.curr_abs = n;
    sub_ctx.use_location = n;

    Nodes params = get_abstraction_params(n->node);
    for (size_t j = 0; j < params.count; j++) {
        Uses* param_uses = arena_alloc(ctx->scope_uses->arena, sizeof(Uses));
        *param_uses = (Uses) { .defined = n, .uses_count = 0, .escapes_defining_block = false, .sealed = false };
        bool not_duplicate = insert_dict_and_get_result(const Node*, Uses*, ctx->scope_uses->map, params.nodes[j], param_uses);
        assert(not_duplicate);
    }

    // if (!is_function(n->node)) {
    //     Uses* block_uses = arena_alloc(scope_uses->arena, sizeof(Uses));
    //     *block_uses = (Uses) { .used = false, .leaks = false };
    //     insert_dict(const Node*, Uses*, scope_uses->map, n->node, block_uses);
    // }

    visit_terminator(ctx, get_abstraction_body(n->node));

    for (size_t i = 0; i < entries_count_list(n->dominates); i++) {
        CFNode* child = read_list(CFNode*, n->dominates)[i];
        // structural dominance is taken care of by the visitor code
        if (!find_key_dict(const Node*, n->structurally_dominated, child->node))
           visit_domtree(ctx, child);
    }
}

ScopeUses* analyse_uses_scope(Scope* s) {
    ScopeUses* scope_uses = calloc(1, sizeof(ScopeUses));
    scope_uses->scope = s;
    scope_uses->map = new_dict(const Node*, Uses*, (HashFn) hash_node, (CmpFn) compare_node);
    scope_uses->arena = new_arena();

    VisitCtx ctx = {
        .scope_uses = scope_uses,
    };
    visit_domtree(&ctx, s->entry);

    return scope_uses;
}

void destroy_uses_scope(ScopeUses* u) {
    destroy_arena(u->arena);
    destroy_dict(u->map);
    free(u);
}

bool is_control_static(ScopeUses* uses, const Node* control) {
    assert(control->tag == Control_TAG);
    const Node* inside = control->payload.control.inside;
    assert(is_anonymous_lambda(inside));
    const Node* jp = first(get_abstraction_params(inside));
    Uses* param_uses = *find_value_dict(const Node*, Uses*, uses->map, jp);
    return !param_uses->escapes_defining_block && !param_uses->in_non_callee_position;
}
