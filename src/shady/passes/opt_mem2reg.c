#include "passes.h"

#include "portability.h"
#include "dict.h"
#include "arena.h"

#include "../analysis/scope.h"
#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"
#include "../analysis/uses.h"

typedef struct {
    CFNode* alloc_in;
    AddressSpace as;
    const Node* ptr_value;
    bool ptr_has_leaked;
    bool* ptr_leaks_ever;
} Knowledge;

typedef struct KB KnowledgeBase;

struct KB {
    const Node* abs; // debug
    // when the associated node has exactly one parent edge, we can safely assume what held true
    // for it will hold true for this one too, unless we have conflicting information
    const KnowledgeBase* dominator_kb;
    struct Dict* map;
    Arena* a;
};

typedef struct {
    Rewriter rewriter;
    struct Dict* abs_to_kb;
    const Node* abs;
} Context;

static const Knowledge* read_node_knowledge(const KnowledgeBase* kb, const Node* n) {
    Knowledge** found = find_value_dict(const Node*, Knowledge*, kb->map, n);
    if (found)
        return *found;
    if (kb->dominator_kb) {
        return read_node_knowledge(kb->dominator_kb, n);
    }
    return NULL;
}

static Knowledge* get_node_knowledge(KnowledgeBase* kb, const Node* n, bool create_if_missing) {
    Knowledge** found = find_value_dict(const Node*, Knowledge*, kb->map, n);
    if (found)
        return *found;
    const Knowledge* old_k = NULL;
    if (kb->dominator_kb)
        old_k = read_node_knowledge(kb->dominator_kb, n);
    if (!create_if_missing)
        return NULL;
    Knowledge* k = arena_alloc(kb->a, sizeof(Knowledge));
    if (old_k) {
        // copy the previous knowledge we had about the node
        *k = *old_k;
    } else {
        k->ptr_leaks_ever = arena_alloc(kb->a, sizeof(bool));
    }
    insert_dict(const Node*, Knowledge*, kb->map, n, k);
    return k;
}

static void insert_node_knowledge(KnowledgeBase* kb, const Node* n, Knowledge* k) {
    Knowledge** found = find_value_dict(const Node*, Knowledge*, kb->map, n);
    assert(!found);
    insert_dict(const Node*, Knowledge*, kb->map, n, k);
}

typedef struct {
    Visitor visitor;
    KnowledgeBase* kb;
} KBVisitor;

static void register_parameters(KBVisitor* v, NodeTag tag, const Node* n) {
    assert(tag == NcVariable);
    Knowledge* k = get_node_knowledge(v->kb, n, true);
}

static void tag_as_leaking_values(KBVisitor* v, NodeTag tag, const Node* n) {
    assert(tag == NcValue);
    Knowledge* k = get_node_knowledge(v->kb, n, false);
    if (k) {
        k->ptr_has_leaked = true;
        *k->ptr_leaks_ever = true;
    }
}

static void visit_instruction(KnowledgeBase* kb, const Node* instruction, Nodes results) {
    IrArena* a = instruction->arena;
    switch (is_instruction(instruction)) {
        case NotAnInstruction: assert(is_instruction(instruction));
        case Instruction_Call_TAG:
            break;
        case Instruction_PrimOp_TAG: {
            PrimOp payload = instruction->payload.prim_op;
            switch (payload.op) {
                case alloca_logical_op:
                case alloca_op: {
                    Knowledge* k = get_node_knowledge(kb, instruction, true);
                    insert_node_knowledge(kb, first(results), k);
                    k->ptr_value = undef(a, (Undef) { .type = first(payload.type_arguments) });
                    break;
                }
                case store_op: {
                    Knowledge* k = get_node_knowledge(kb, first(payload.operands), true);
                    k->ptr_value = payload.operands.nodes[1];
                    break; // let's take care of dead stores another time
                }
                case reinterpret_op: {
                    // if we have knowledge on a particular ptr, the same knowledge propagates if we bitcast it!
                    Knowledge* k = get_node_knowledge(kb, first(payload.operands), false);
                    if (k)
                        insert_node_knowledge(kb, first(results), k);
                    break;
                }
                case memcpy_op: {

                }
                default: break;
            }
            break;
        }
        case Instruction_Control_TAG:
            break;
        case Instruction_Block_TAG:
            break;
        case Instruction_Comment_TAG:
            break;
        case Instruction_If_TAG:
        case Instruction_Match_TAG:
        case Instruction_Loop_TAG:
            assert(false && "unsupported");
            break;
    }

    /*const Node* new_instr = rewrite_node(&ctx->rewriter, instruction);
    return new_instr;*/
}

static void visit_terminator(KBVisitor* v, const Node* old) {
    switch (is_terminator(old)) {
        case Terminator_LetMut_TAG:
        case NotATerminator: assert(false);
        case Terminator_Let_TAG: {
            const Node* otail = get_let_tail(old);
            visit_instruction(v->kb, get_let_instruction(old), get_abstraction_params(otail));
            break;
        }
        case Terminator_TailCall_TAG:
        case Terminator_Return_TAG:
        indirect_join: {
            KBVisitor leaking_visitor = { .visitor = { .visit_op_fn = (VisitOpFn) tag_as_leaking_values }, .kb = v->kb };
            visit_node_operands(&leaking_visitor.visitor, ~NcValue, old);
            break;
        }
        case Terminator_Join_TAG:
            goto indirect_join; // TODO
            break;
        case Terminator_Jump_TAG: {

            break;
        }
        case Terminator_Branch_TAG:
        case Terminator_Switch_TAG: {
            visit_node_operands(&v->visitor, ~NcJump, old);
            break;
        }
        case Terminator_Unreachable_TAG:
            break;
        case Terminator_MergeContinue_TAG:
        case Terminator_MergeBreak_TAG:
        case Terminator_Yield_TAG:
            assert(false && "unsupported");
            break;
    }
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

static void visit_cfnode(Context* ctx, CFNode* node, CFNode* dominator) {
    const Node* oabs = node->node;
    KnowledgeBase kb = {
        .abs = node->node,
        .a = new_arena(),
        .map = new_dict(const Node*, Knowledge*, (HashFn) hash_node, (CmpFn) compare_node),
        .dominator_kb = NULL,
    };
    if (entries_count_list(node->pred_edges) == 1) {
        assert(dominator);
        CFEdge edge = read_list(CFEdge, node->pred_edges)[0];
        assert(edge.dst == node);
        assert(edge.src == dominator);
        const KnowledgeBase* parent_kb = find_value_dict(const Node*, KnowledgeBase, ctx->abs_to_kb, dominator->node);
        assert(parent_kb->map);
        kb.dominator_kb = parent_kb;
    }
    assert(kb.map);
    insert_dict(const Node*, KnowledgeBase, ctx->abs_to_kb, node->node, kb);

    KBVisitor v = { .visitor = { .visit_op_fn = (VisitOpFn) register_parameters }, .kb = &kb };
    visit_node_operands(&v.visitor, ~NcVariable, oabs);

    assert(is_abstraction(oabs));
    visit_terminator(&v, get_abstraction_body(oabs));

    for (size_t i = 0; i < entries_count_list(node->dominates); i++) {
        CFNode* dominated = read_list(CFNode*, node->dominates)[i];
        visit_cfnode(ctx, dominated, node);
    }
}

static const Node* process(Context* ctx, const Node* old) {
    assert(old);
    Context fn_ctx = *ctx;
    if (old->tag == Function_TAG) {
        Scope* s = new_scope(old);
        ctx = &fn_ctx;
        fn_ctx.abs_to_kb = new_dict(const Node*, KnowledgeBase, (HashFn) hash_node, (CmpFn) compare_node);
        visit_cfnode(&fn_ctx, s->entry, NULL);
        fn_ctx.abs = old;
        const Node* new_fn = recreate_node_identity(&fn_ctx.rewriter, old);
        destroy_scope(s);
        size_t i = 0;
        KnowledgeBase kb;
        while (dict_iter(fn_ctx.abs_to_kb, &i, NULL, &kb)) {
            destroy_dict(kb.map);
            destroy_arena(kb.a);
        }
        destroy_dict(fn_ctx.abs_to_kb);
        return new_fn;
    } else if (is_abstraction(old)) {
        fn_ctx.abs = old;
        ctx = &fn_ctx;
    }

    KnowledgeBase* kb = NULL;
    if (ctx->abs) {
        kb = find_value_dict(const Node*, KnowledgeBase, ctx->abs_to_kb, ctx->abs);
        assert(kb);
    }

    IrArena* a = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case PrimOp_TAG: {
            PrimOp payload = old->payload.prim_op;
            switch (payload.op) {
                case load_op: {
                    const Knowledge* k = read_node_knowledge(kb, first(payload.operands));
                    if (k && k->ptr_value && !*k->ptr_leaks_ever)
                        return quote_helper(a, singleton(rewrite_node(&ctx->rewriter, k->ptr_value)));
                    break;
                }
                case store_op: {
                    const Knowledge* k = read_node_knowledge(kb, first(payload.operands));
                    if (k && !*k->ptr_leaks_ever)
                        return quote_helper(a, empty(a));
                    break;
                }
                case alloca_op: {
                    const Knowledge* k = read_node_knowledge(kb, old);
                    if (k && !*k->ptr_leaks_ever)
                        return quote_helper(a, singleton(undef(a, (Undef) { .type = get_unqualified_type(rewrite_node(&ctx->rewriter, old->type)) })));
                    break;
                }
                default: break;
            }
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

Module* opt_mem2reg(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
    };

    ctx.rewriter.config.fold_quote = false;
    ctx.rewriter.config.rebind_let = false;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
