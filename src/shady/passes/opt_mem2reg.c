#include "passes.h"

#include "portability.h"
#include "dict.h"
#include "arena.h"
#include "log.h"

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
    CFNode* cfnode;
    // when the associated node has exactly one parent edge, we can safely assume what held true
    // for it will hold true for this one too, unless we have conflicting information
    const KnowledgeBase* dominator_kb;
    struct Dict* map;
    struct Dict* potential_additional_params;
    Arena* a;
};

typedef struct {
    Rewriter rewriter;
    Scope* scope;
    struct Dict* abs_to_kb;
    const Node* abs;
    Arena* a;

    struct Dict* bb_new_args;
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

static const Node* get_ptr_known_value_(Context* ctx, const KnowledgeBase* kb, const Node* n, bool import) {
    const Knowledge* k = read_node_knowledge(kb, n);
    if (k && !*k->ptr_leaks_ever) {
        if (k->ptr_value) {
            if (import && k->ptr_value->arena == ctx->rewriter.src_arena)
               return rewrite_node(&ctx->rewriter, k->ptr_value);
            return k->ptr_value;
        }
    }
    return NULL;
}

static bool has_ptr_known_value(Context* ctx, const KnowledgeBase* kb, const Node* n) {
    return get_ptr_known_value_(ctx, kb, n, false) != NULL;
}

static const Node* get_ptr_known_value(Context* ctx, const KnowledgeBase* kb, const Node* n) {
    return get_ptr_known_value_(ctx, kb, n, true);
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
                case load_op: {
                    const Node* ptr = first(payload.operands);
                    const Knowledge* k = read_node_knowledge(kb, ptr);
                    if (!k || !k->ptr_value) {
                        const KnowledgeBase* phi_kb = kb;
                        while (phi_kb->dominator_kb) {
                            phi_kb = phi_kb->dominator_kb;
                        }
                        debug_print("mem2reg: It'd sure be nice to know the value of ");
                        log_node(DEBUG, first(payload.operands));
                        debug_print(" at phi-like node %s.\n", get_abstraction_name(phi_kb->cfnode->node));
                        // log_node(DEBUG, phi_location->node);
                        insert_set_get_key(const Node*, phi_kb->potential_additional_params, ptr);
                    }
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
    if (!old)
        return;
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
            KBVisitor jumps_visitor = { .visitor = { .visit_node_fn = (VisitNodeFn) visit_terminator }, .kb = v->kb };
            visit_node_operands(&jumps_visitor.visitor, ~NcJump, old);
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

static void destroy_kb(KnowledgeBase* kb) {
    destroy_dict(kb->map);
    destroy_dict(kb->potential_additional_params);
}

static KnowledgeBase* get_kb(Context* ctx, const Node* abs) {
    KnowledgeBase** found = find_value_dict(const Node*, KnowledgeBase*, ctx->abs_to_kb, abs);
    assert(found);
    return *found;
}

static void visit_cfnode(Context* ctx, CFNode* node, CFNode* dominator) {
    const Node* oabs = node->node;
    KnowledgeBase* kb = arena_alloc(ctx->a, sizeof(KnowledgeBase));
    *kb = (KnowledgeBase) {
        .cfnode = node,
        .a = ctx->a,
        .map = new_dict(const Node*, Knowledge*, (HashFn) hash_node, (CmpFn) compare_node),
        .potential_additional_params = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .dominator_kb = NULL,
    };
    if (entries_count_list(node->pred_edges) == 1) {
        assert(dominator);
        CFEdge edge = read_list(CFEdge, node->pred_edges)[0];
        assert(edge.dst == node);
        assert(edge.src == dominator);
        const KnowledgeBase* parent_kb = get_kb(ctx, dominator->node);
        assert(parent_kb->map);
        kb->dominator_kb = parent_kb;
    }
    assert(kb->map);
    insert_dict(const Node*, KnowledgeBase*, ctx->abs_to_kb, node->node, kb);

    KBVisitor v = { .visitor = { .visit_op_fn = (VisitOpFn) register_parameters }, .kb = kb };
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
    if (old->tag == Function_TAG && !lookup_annotation(old, "Internal")) {
        Scope* s = new_scope(old);
        ctx = &fn_ctx;
        fn_ctx.scope = s;
        fn_ctx.abs_to_kb = new_dict(const Node*, KnowledgeBase**, (HashFn) hash_node, (CmpFn) compare_node);
        visit_cfnode(&fn_ctx, s->entry, NULL);
        fn_ctx.abs = old;
        const Node* new_fn = recreate_node_identity(&fn_ctx.rewriter, old);
        destroy_scope(s);
        size_t i = 0;
        KnowledgeBase* kb;
        while (dict_iter(fn_ctx.abs_to_kb, &i, NULL, &kb)) {
            destroy_kb(kb);
        }
        destroy_dict(fn_ctx.abs_to_kb);
        return new_fn;
    } else if (is_abstraction(old)) {
        fn_ctx.abs = old;
        ctx = &fn_ctx;
    }

    KnowledgeBase* kb = NULL;
    if (ctx->abs && ctx->abs_to_kb) {
        kb = get_kb(ctx, ctx->abs);
        assert(kb);
    }

    if (!kb)
        return recreate_node_identity(&ctx->rewriter, old);

    IrArena* a = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case PrimOp_TAG: {
            PrimOp payload = old->payload.prim_op;
            switch (payload.op) {
                case load_op: {
                    const Node* known_value = get_ptr_known_value(ctx, kb, first(payload.operands));
                    if (known_value)
                        return quote_helper(a, singleton(known_value));
                    break;
                }
                /*case store_op: {
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
                }*/
                default: break;
            }
            break;
        }
        case BasicBlock_TAG: {
            CFNode* cfnode = scope_lookup(ctx->scope, old);
            size_t i = 0;
            const Node* ptr;
            Nodes params = recreate_variables(&ctx->rewriter, get_abstraction_params(old));
            register_processed_list(&ctx->rewriter, get_abstraction_params(old), params);
            Nodes ptrs = empty(ctx->rewriter.src_arena);
            while (dict_iter(kb->potential_additional_params, &i, &ptr, NULL)) {
                // check if all the edges have a value for this!
                for (size_t j = 0; j < entries_count_list(cfnode->pred_edges); j++) {
                    CFEdge edge = read_list(CFEdge, cfnode->pred_edges)[j];
                    if (edge.type == StructuredPseudoExitEdge)
                        continue; // these are not real edges...
                    KnowledgeBase* kb_at_src = get_kb(ctx, edge.src->node);
                    if (has_ptr_known_value(ctx, kb_at_src, ptr)) {
                        log_node(DEBUG, ptr);
                        debug_print(" has a known value in %s ...\n", get_abstraction_name(edge.src->node));
                    } else
                        goto next_potential_param;
                }

                log_node(DEBUG, ptr);
                debug_print(" has a known value in all predecessors! Turning it into a new parameter.\n");
                const Type* t = rewrite_node(&ctx->rewriter, ptr->type);
                bool u = deconstruct_qualified_type(&t);
                deconstruct_pointer_type(&t);
                const Node* param = var(a, qualified_type_helper(t, u), unique_name(a, "ssa_phi"));
                params = append_nodes(a, params, param);
                ptrs = append_nodes(ctx->rewriter.src_arena, ptrs, ptr);

                Knowledge* k = get_node_knowledge(kb, ptr, true);
                assert(!k->ptr_value && "if we had a value already, we wouldn't be creating a param!");
                k->ptr_value = param;

                next_potential_param: continue;
            }

            if (ptrs.count > 0) {
                insert_dict(const Node*, Nodes, ctx->bb_new_args, old, ptrs);
            }

            Node* fn = (Node*) rewrite_node(&ctx->rewriter, ctx->scope->entry->node);
            Node* bb = basic_block(a, fn, params, get_abstraction_name(old));
            register_processed(&ctx->rewriter, old, bb);
            bb->payload.basic_block.body = rewrite_node(&ctx->rewriter, get_abstraction_body(old));
            return bb;
        }
        case Jump_TAG: {
            const Node* new_bb = rewrite_node(&ctx->rewriter, old->payload.jump.target);
            Nodes args = rewrite_nodes(&ctx->rewriter, old->payload.jump.args);

            Nodes* additional_ssa_params = find_value_dict(const Node*, Nodes, ctx->bb_new_args, old->payload.jump.target);
            if (additional_ssa_params) {
                for (size_t i = 0; i < additional_ssa_params->count; i++) {
                    const Node* ptr = additional_ssa_params->nodes[i];
                    const Node* value = get_ptr_known_value(ctx, kb, ptr);
                    assert(value);
                    args = append_nodes(a, args, value);
                }
            }

            return jump_helper(a, new_bb, args);
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
        .bb_new_args = new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
        .a = new_arena(),
    };

    ctx.rewriter.config.fold_quote = false;
    ctx.rewriter.config.rebind_let = false;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.bb_new_args);
    destroy_arena(ctx.a);
    return dst;
}
