#include "passes.h"

#include "portability.h"
#include "dict.h"
#include "arena.h"
#include "log.h"

#include "../analysis/scope.h"
#include "../analysis/uses.h"
#include "../analysis/leak.h"
#include "../analysis/verify.h"

#include "../transform/ir_gen_helpers.h"

#include "../rewrite.h"
#include "../visit.h"
#include "../type.h"

typedef struct {
    AddressSpace as;
    const Type* type;
} PtrSourceKnowledge;

typedef struct {
    enum {
        PSUnknown,
        PSKnownValue,
        PSKnownAlias,
        PSKnownSubElement,
    } state;
    union {
        const Node* ptr_value;
        // for PSKnownAlias: old node to lookup
        const Node* alias_old_address;
        struct { const Node* old_base; Nodes indices; } sub_element;
    };
    bool ptr_has_leaked;
    PtrSourceKnowledge* source;
} PtrKnowledge;

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
    const Node* old_jump;
    Node* wrapper_bb;
    KnowledgeBase* kb;
} TodoJump;

typedef struct {
    Rewriter rewriter;
    Scope* scope;
    struct Dict* abs_to_kb;
    const Node* oabs;
    Arena* a;

    struct Dict* bb_new_args;
    struct List* todo_jumps;
} Context;

static PtrKnowledge* get_last_valid_ptr_knowledge(const KnowledgeBase* kb, const Node* n) {
    PtrKnowledge** found = find_value_dict(const Node*, PtrKnowledge*, kb->map, n);
    if (found)
        return *found;
    PtrKnowledge* k = NULL;
    if (kb->dominator_kb)
        k = get_last_valid_ptr_knowledge(kb->dominator_kb, n);
    return k;
}

static PtrKnowledge* create_ptr_knowledge(KnowledgeBase* kb, const Node* instruction) {
    PtrKnowledge* k = arena_alloc(kb->a, sizeof(PtrKnowledge));
    PtrSourceKnowledge* sk = arena_alloc(kb->a, sizeof(PtrSourceKnowledge));
    *k = (PtrKnowledge) { .source = sk, .state = PSUnknown/*, .ptr_address = address_value*/ };
    *sk = (PtrSourceKnowledge) { 0 };
    bool fresh = insert_dict(const Node*, PtrKnowledge*, kb->map, instruction, k);
    assert(fresh);
    return k;
}

static PtrKnowledge* update_ptr_knowledge(KnowledgeBase* kb, const Node* n, PtrKnowledge* existing) {
    PtrKnowledge* k = arena_alloc(kb->a, sizeof(PtrKnowledge));
    *k = *existing; // copy the data
    bool fresh = insert_dict(const Node*, PtrKnowledge*, kb->map, n, k);
    assert(fresh);
    return k;
}

static void insert_ptr_knowledge(KnowledgeBase* kb, const Node* n, PtrKnowledge* k) {
    PtrKnowledge** found = find_value_dict(const Node*, PtrKnowledge*, kb->map, n);
    assert(!found);
    insert_dict(const Node*, PtrKnowledge*, kb->map, n, k);
}

static const Node* get_known_value(KnowledgeBase* kb, const PtrKnowledge* k) {
    const Node* v = NULL;
    while (k) {
        if (k->state == PSKnownValue) {
            v = k->ptr_value;
            k = NULL;
        } else if (k->state == PSKnownAlias) {
            k = get_last_valid_ptr_knowledge(kb, k->alias_old_address);
        } else
            k = NULL;
    }
    return v;
}

/*static const Node* get_known_address(Rewriter* r, const PtrKnowledge* k) {
    const Node* v = NULL;
    if (k) {
        if (k->ptr_address) {
            v = k->ptr_address;
        }
    }
    if (v)
        assert(v->arena == r->dst_arena);
    // if (r && v && v->arena != r->dst_arena)
    //     return rewrite_node(r, v);
    return v;
}*/

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

static void destroy_kb(KnowledgeBase* kb) {
    destroy_dict(kb->map);
    destroy_dict(kb->potential_additional_params);
}

static KnowledgeBase* get_kb(Context* ctx, const Node* abs) {
    assert(ctx->scope);
    KnowledgeBase** found = find_value_dict(const Node*, KnowledgeBase*, ctx->abs_to_kb, abs);
    assert(found);
    return *found;
}

static KnowledgeBase* create_kb(Context* ctx, const Node* old) {
    assert(ctx->scope);
    arena_alloc(ctx->a, sizeof(KnowledgeBase));
    CFNode* cf_node = scope_lookup(ctx->scope, old);
    KnowledgeBase* kb = arena_alloc(ctx->a, sizeof(KnowledgeBase));
    *kb = (KnowledgeBase) {
        .cfnode = cf_node,
        .a = ctx->a,
        .map = new_dict(const Node*, PtrKnowledge*, (HashFn) hash_node, (CmpFn) compare_node),
        .potential_additional_params = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node),
        .dominator_kb = NULL,
    };
    // log_string(DEBUGVV, "Creating KB for ");
    // log_node(DEBUGVV, old);
    // log_string(DEBUGVV, "\n.");
    if (entries_count_list(cf_node->pred_edges) == 1) {
        CFEdge edge = read_list(CFEdge, cf_node->pred_edges)[0];
        assert(edge.dst == cf_node);
        if (edge.type == LetTailEdge || edge.type == JumpEdge) {
            CFNode* dominator = edge.src;
            const KnowledgeBase* parent_kb = get_kb(ctx, dominator->node);
            assert(parent_kb->map);
            kb->dominator_kb = parent_kb;
        }
    }
    assert(kb->map);
    insert_dict(const Node*, KnowledgeBase*, ctx->abs_to_kb, old, kb);
    return kb;
}

static void wipe_all_leaked_pointers(KnowledgeBase* kb) {
    size_t i = 0;
    const Node* ptr;
    PtrKnowledge* k;
    while (dict_iter(kb->map, &i, &ptr, &k)) {
        if (k->ptr_has_leaked) {
            k->ptr_value = NULL;
        }
    }
}

static void mark_values_as_escaping(KnowledgeBase* kb, Nodes values);
static void mark_value_as_escaping(KnowledgeBase* kb, const Node* value) {
    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, value);
    if (k)
        k->ptr_has_leaked = true;
    switch (is_value(value)) {
        case NotAValue: assert(false);
        case Value_Variable_TAG:
            break;
        case Value_ConstrainedValue_TAG:
            break;
        case Value_UntypedNumber_TAG:
            break;
        case Value_IntLiteral_TAG:
            break;
        case Value_FloatLiteral_TAG:
            break;
        case Value_StringLiteral_TAG:
            break;
        case Value_True_TAG:
            break;
        case Value_False_TAG:
            break;
        case Value_NullPtr_TAG:
            break;
        case Value_Composite_TAG:
            mark_values_as_escaping(kb, value->payload.composite.contents);
            break;
        case Value_Fill_TAG:
            mark_value_as_escaping(kb, value->payload.fill.value);
            break;
        case Value_Undef_TAG:
            break;
        case Value_RefDecl_TAG:
            break;
        case Value_FnAddr_TAG:
            break;
    }
}

static void mark_values_as_escaping(KnowledgeBase* kb, Nodes values) {
    for (size_t i = 0; i < values.count; i++)
        mark_value_as_escaping(kb, values.nodes[i]);
}

static const Node* process_instruction(Context* ctx, KnowledgeBase* kb, const Node* oinstruction) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (is_instruction(oinstruction)) {
        case NotAnInstruction: assert(is_instruction(oinstruction));
        case Instruction_Call_TAG:
            wipe_all_leaked_pointers(kb);
            break;
        case Instruction_PrimOp_TAG: {
            PrimOp payload = oinstruction->payload.prim_op;
            switch (payload.op) {
                case alloca_logical_op:
                case alloca_op: {
                    PtrKnowledge* k = create_ptr_knowledge(kb, oinstruction);
                    const Type* t = oinstruction->type;
                    deconstruct_qualified_type(&t);
                    assert(t->tag == PtrType_TAG);
                    k->source->as = t->payload.ptr_type.address_space;
                    //k->source->type = qualified_type_helper(t, u);
                    k->source->type = rewrite_node(r, first(payload.type_arguments));

                    // TODO: we can only enable this safely once we properly deal with control-flow
                    k->state = PSKnownValue;
                    k->ptr_value = undef(a, (Undef) { .type = rewrite_node(r, first(payload.type_arguments)) });
                    return recreate_node_identity(r, oinstruction);
                }
                case load_op: {
                    const Node* optr = first(payload.operands);
                    PtrKnowledge* ok = get_last_valid_ptr_knowledge(kb, optr);
                    const Node* known_value = get_known_value(kb, ok);
                    if (known_value) {
                        const Type* known_value_t = known_value->type;
                        bool kv_u = deconstruct_qualified_type(&known_value_t);

                        const Type* load_result_t = rewrite_node(&ctx->rewriter, optr->type);
                        bool lrt_u = deconstruct_qualified_type(&load_result_t);
                        deconstruct_pointer_type(&load_result_t);
                        assert(!lrt_u || kv_u);
                        if (is_reinterpret_cast_legal(load_result_t, known_value_t))
                            return prim_op_helper(a, reinterpret_op, singleton(load_result_t), singleton(known_value));
                    } else {
                        const KnowledgeBase* phi_kb = kb;
                        while (phi_kb->dominator_kb) {
                            phi_kb = phi_kb->dominator_kb;
                        }
                        debug_print("mem2reg: It'd sure be nice to know the value of ");
                        log_node(DEBUG, first(payload.operands));
                        debug_print(" at phi-like node %s.\n", get_abstraction_name(phi_kb->cfnode->node));
                        // log_node(DEBUG, phi_location->node);
                        insert_set_get_key(const Node*, phi_kb->potential_additional_params, optr);
                    }
                    // const Node* other_ptr = get_known_address(&ctx->rewriter, ok);
                    // if (other_ptr && optr != other_ptr) {
                    //     return prim_op_helper(a, load_op, empty(a), singleton(other_ptr));
                    // }
                    return prim_op_helper(a, load_op, empty(a), singleton(rewrite_node(r, optr)));
                }
                case store_op: {
                    const Node* optr = first(payload.operands);
                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, optr);
                    if (k) {
                        k = update_ptr_knowledge(kb, optr, k);
                    } else {
                        PtrSourceKnowledge* sk = NULL;
                        CFNode* node = scope_lookup(ctx->scope, ctx->oabs);
                        while (node) {
                            KnowledgeBase* kb2 = get_kb(ctx, node->node);
                            assert(kb2);
                            PtrKnowledge* k2 = get_last_valid_ptr_knowledge(kb2, optr);
                            if (k2) {
                                sk = k2->source;
                                break;
                            }
                            node = node->idom;
                        }
                        if (sk) {
                            k = arena_alloc(ctx->a, sizeof(PtrKnowledge));
                            *k = (PtrKnowledge) {
                                .source = sk,
                                .ptr_has_leaked = true // TODO: this is wrong in the "too conservative" way
                                                       // fixing this requires accounting for the dominance relation properly
                                                       // to visit all predecessors first, then merging the knowledge
                            };
                            insert_ptr_knowledge(kb, optr, k);
                        } else {
                            // just make up a new source and assume it leaks/aliases
                            k = create_ptr_knowledge(kb, optr);
                            const Type* t = optr->type;
                            deconstruct_qualified_type(&t);
                            assert(t->tag == PtrType_TAG);
                            k->source->as = t->payload.ptr_type.address_space;
                            k->source->type = rewrite_node(r, get_pointer_type_element(t));
                            k->ptr_has_leaked = true;
                        }
                    }
                    if (k) {
                        k->state = PSKnownValue;
                        k->ptr_value = rewrite_node(r, payload.operands.nodes[1]);
                    }
                    // let's take care of dead stores another time
                    return recreate_node_identity(r, oinstruction);
                }
                case reinterpret_op: {
                    const Node* rewritten = recreate_node_identity(r, oinstruction);
                    // if we have knowledge on a particular ptr, the same knowledge propagates if we bitcast it!
                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, first(payload.operands));
                    if (k) {
                        k = update_ptr_knowledge(kb, oinstruction, k);
                        k->state = PSKnownAlias;
                        k->alias_old_address = first(payload.operands);
                    }
                    return rewritten;
                }
                case convert_op: {
                    const Node* rewritten = recreate_node_identity(r, oinstruction);
                    // if we convert a pointer to generic AS, we'd like to use the old address instead where possible
                    if (first(payload.type_arguments)->tag == PtrType_TAG) {
                        PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, first(payload.operands));
                        if (k) {
                            debug_print("mem2reg: the converted ptr ");
                            log_node(DEBUG, oinstruction);
                            debug_print(" is the same as ");
                            log_node(DEBUG, first(payload.operands));
                            debug_print(".\n");
                            k = update_ptr_knowledge(kb, oinstruction, k);
                            k->state = PSKnownAlias;
                            k->alias_old_address = first(payload.operands);
                        }
                    }
                    break;
                }
                default: break;
            }

            mark_values_as_escaping(kb, payload.operands);
            if (has_primop_got_side_effects(payload.op))
                wipe_all_leaked_pointers(kb);

            return recreate_node_identity(r, oinstruction);
        }
        case Instruction_Comment_TAG:
            break;
    }

    return recreate_node_identity(r, oinstruction);
}

static const Node* process_terminator(Context* ctx, KnowledgeBase* kb, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (is_terminator(old)) {
        case NotATerminator: assert(false);
        case Control_TAG:
            break;
        case If_TAG:
            break;
        case Match_TAG:
            break;
        case Loop_TAG:
            mark_values_as_escaping(kb, old->payload.structured_loop.initial_args);
            // assert(false && "unsupported");
            break;
        /*case Let_TAG: {
            const Node* oinstruction = get_let_instruction(old);
            const Node* ninstruction = rewrite_node(r, oinstruction);
            PtrKnowledge** found = find_value_dict(const Node*, PtrKnowledge*, kb->map, oinstruction);
            if (found) { // copy any knownledge about an instruction to the bound variable
                const Node* old_case = get_let_tail(old);
                Nodes old_params = get_abstraction_params(old_case);
                assert(old_params.count == 1);
                PtrKnowledge* k = *found;
                const Node* first_param = first(old_params);
                insert_dict(const Node*, PtrKnowledge*, kb->map, first_param, k);
            }

            return let(a, ninstruction, rewrite_node(r, get_let_tail(old)));
        }*/
        case Jump_TAG: {
            const Node* old_target = old->payload.jump.target;
            // rewrite_node(&ctx->rewriter, old_target);
            Nodes args = rewrite_nodes(&ctx->rewriter, old->payload.jump.args);

            //String s = format_string_interned(a, "%s_", get_abstraction_name(old_target));
            String s = get_abstraction_name(old_target);
            Node* wrapper = basic_block(a, (Node*) rewrite_node(r, old_target->payload.basic_block.fn), recreate_variables(r, get_abstraction_params(old_target)), s);
            TodoJump todo = {
                .old_jump = old,
                .wrapper_bb = wrapper,
                .kb = kb,
            };
            append_list(TodoJump, ctx->todo_jumps, todo);

            return jump_helper(a, wrapper, args);
        }
        case Terminator_TailCall_TAG:
            mark_values_as_escaping(kb, old->payload.tail_call.args);
            break;
        case Terminator_Branch_TAG:
            break;
        case Terminator_Switch_TAG:
            break;
        case Terminator_Join_TAG:
            // TODO: local joins are fine
            mark_values_as_escaping(kb, old->payload.join.args);
            break;
        case Terminator_MergeContinue_TAG:
            break;
        case Terminator_MergeBreak_TAG:
            break;
        case Terminator_Yield_TAG:
            break;
        case Terminator_Return_TAG:
            mark_values_as_escaping(kb, old->payload.fn_ret.args);
            break;
        case Terminator_Unreachable_TAG:
            break;
    }
    return recreate_node_identity(r, old);
}

static void handle_bb(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    log_string(DEBUGV, "mem2reg: handling bb %s\n", get_abstraction_name(old));

    KnowledgeBase* kb = create_kb(ctx, old);
    Context fn_ctx = *ctx;
    fn_ctx.oabs = old;
    ctx = &fn_ctx;

    Nodes params = recreate_variables(&ctx->rewriter, get_abstraction_params(old));
    //Nodes let_params = recreate_variables(&ctx->rewriter, get_abstraction_params(old));
    register_processed_list(&ctx->rewriter, get_abstraction_params(old), params);
    const Node* nbody = rewrite_node(&ctx->rewriter, get_abstraction_body(old));
    //nbody = let(a, quote_helper(a, params), case_(a, let_params, nbody));
    //nbody = body(a, quote_helper(a, params), case_(a, let_params, nbody));

    CFNode* cfnode = scope_lookup(ctx->scope, old);
    BodyBuilder* bb = begin_body(a);
    size_t i = 0;
    const Node* ptr;
    Nodes ptrs = empty(ctx->rewriter.src_arena);
    while (dict_iter(kb->potential_additional_params, &i, &ptr, NULL)) {
        PtrSourceKnowledge* source = NULL;
        PtrKnowledge uk = { 0 };
        // check if all the edges have a value for this!
        for (size_t j = 0; j < entries_count_list(cfnode->pred_edges); j++) {
            CFEdge edge = read_list(CFEdge, cfnode->pred_edges)[j];
            if (edge.type == StructuredPseudoExitEdge)
                continue; // these are not real edges...
            KnowledgeBase* kb_at_src = get_kb(ctx, edge.src->node);

            const Node* kv = get_known_value(kb_at_src, get_last_valid_ptr_knowledge(kb_at_src, ptr));
            if (kv) {
                log_node(DEBUG, ptr);
                debug_print(" has a known value (");
                log_node(DEBUG, kv);
                debug_print(") in %s ...\n", get_abstraction_name(edge.src->node));
            } else
                goto next_potential_param;

            PtrKnowledge* k = get_last_valid_ptr_knowledge(kb_at_src, ptr);
            if (!source)
                source = k->source;
            else
                assert(source == k->source);

            const Type* kv_type = get_known_value(kb_at_src, get_last_valid_ptr_knowledge(kb_at_src, ptr))->type;
            deconstruct_qualified_type(&kv_type);
            const Type* alloca_type_t = source->type;
            //deconstruct_qualified_type(&alloca_type_t);
            if (kv_type != source->type && !is_reinterpret_cast_legal(kv_type, alloca_type_t)) {
                log_node(DEBUG, ptr);
                debug_print(" has a known value in %s, but it's type ", get_abstraction_name(edge.src->node));
                log_node(DEBUG, kv_type);
                debug_print(" cannot be reinterpreted into the alloca type ");
                log_node(DEBUG, source->type);
                debug_print("\n.");
                goto next_potential_param;
            }

            uk.ptr_has_leaked |= k->ptr_has_leaked;
        }

        log_node(DEBUG, ptr);
        debug_print(" has a known value in all predecessors! Turning it into a new parameter.\n");

        const Node* param = var(a, qualified_type_helper(source->type, false), unique_name(a, "ssa_phi"));
        params = append_nodes(a, params, param);
        ptrs = append_nodes(ctx->rewriter.src_arena, ptrs, ptr);
        gen_store(bb, rewrite_node(r, ptr), param);

        PtrKnowledge* k = arena_alloc(ctx->a, sizeof(PtrKnowledge));
        *k = (PtrKnowledge) {
            .ptr_value = param,
            .source = source,
            .ptr_has_leaked = uk.ptr_has_leaked
        };
        insert_ptr_knowledge(kb, ptr, k);

        next_potential_param: continue;
    }

    Node* fn = (Node*) rewrite_node(&ctx->rewriter, ctx->scope->entry->node);
    String s = format_string_interned(a, "%s_", get_abstraction_name(old));
    //String s = get_abstraction_name(old);
    Node* new_bb = basic_block(a, fn, params, s);
    register_processed(&ctx->rewriter, old, new_bb);
    new_bb->payload.basic_block.body = finish_body(bb, nbody);

    // new_bb->type = bb_type(a, (BBType) {
    //     .param_types = get_variables_types(a, get_abstraction_params(new_bb)),
    // });

    if (ptrs.count > 0) {
        insert_dict(const Node*, Nodes, ctx->bb_new_args, old, ptrs);
    }
}

static void handle_jump_wrappers(Context* ctx) {
    IrArena* a = ctx->rewriter.dst_arena;
    for (size_t j = 0; j < entries_count_list(ctx->todo_jumps); j++) {
        TodoJump todo = read_list(TodoJump, ctx->todo_jumps)[j];
        const Node* old = todo.old_jump;
        const Node* old_target = old->payload.jump.target;
        const Node* new_target = rewrite_node(&ctx->rewriter, old_target);
        Nodes args = get_abstraction_params(todo.wrapper_bb);

        BodyBuilder* bb = begin_body(a);
        Nodes* additional_ssa_params = find_value_dict(const Node*, Nodes, ctx->bb_new_args, old_target);
        if (additional_ssa_params) {
            assert(additional_ssa_params->count > 0);

            for (size_t i = 0; i < additional_ssa_params->count; i++) {
                const Node* ptr = additional_ssa_params->nodes[i];
                PtrKnowledge* k = get_last_valid_ptr_knowledge(todo.kb, ptr);
                const Node* value = get_known_value(todo.kb, k);

                const Type* known_value_t = value->type;
                deconstruct_qualified_type(&known_value_t);

                const Type* alloca_type_t = k->source->type;

                if (alloca_type_t != known_value_t && is_reinterpret_cast_legal(alloca_type_t, known_value_t))
                    value = first(gen_primop(bb, reinterpret_op, singleton(rewrite_node(&ctx->rewriter, alloca_type_t)), singleton(value)));

                assert(value);
                args = append_nodes(a, args, value);
            }

        }
        todo.wrapper_bb->payload.basic_block.body = finish_body(bb, jump_helper(a, new_target, args));
    }
}

static const Node* process(Context* ctx, const Node* old) {
    assert(old);
    Context fn_ctx = *ctx;
    if (is_abstraction(old)) {
        fn_ctx.oabs = old;
        ctx = &fn_ctx;
    }

    KnowledgeBase* kb = NULL;
    if (old->tag == Function_TAG) {
        // if (lookup_annotation(old, "Internal")) {
        //     fn_ctx.scope = NULL;
        //     return recreate_node_identity(&fn_ctx.rewriter, old);;
        // }
        fn_ctx.scope = new_scope(old);
        fn_ctx.abs_to_kb = new_dict(const Node*, KnowledgeBase**, (HashFn) hash_node, (CmpFn) compare_node);
        fn_ctx.todo_jumps = new_list(TodoJump),
        kb = create_kb(ctx, old);
        const Node* new_fn = recreate_node_identity(&fn_ctx.rewriter, old);

        for (size_t i = 1; i < ctx->scope->size; i++) {
            CFNode* cf_node = ctx->scope->rpo[i];
            if (cf_node->type == CFNodeType_BBNode)
                handle_bb(ctx, cf_node->abstraction);
        }

        //handle_bb_wrappers(ctx);
        //clear_list(ctx->todo_bbs);
        handle_jump_wrappers(ctx);
        destroy_list(fn_ctx.todo_jumps);

        destroy_scope(fn_ctx.scope);
        size_t i = 0;
        while (dict_iter(fn_ctx.abs_to_kb, &i, NULL, &kb)) {
            destroy_kb(kb);
        }
        destroy_dict(fn_ctx.abs_to_kb);
        return new_fn;
    } else if (old->tag == Constant_TAG) {
        fn_ctx.scope = NULL;
        fn_ctx.abs_to_kb = NULL;
        fn_ctx.todo_jumps = NULL;
        ctx = &fn_ctx;
    }

    // setup a new KB if this is a fresh abstraction
    if (is_abstraction(old) && ctx->scope) {
        kb = create_kb(ctx, old);
    } else if (ctx->oabs && ctx->abs_to_kb) {
        // otherwise look up the enclosing one, if any
        kb = get_kb(ctx, ctx->oabs);
        assert(kb);
    }
    if (!kb)
        return recreate_node_identity(&ctx->rewriter, old);

    if (is_instruction(old))
        return process_instruction(ctx, kb, old);
    if (is_terminator(old))
        return process_terminator(ctx, kb, old);

    switch (old->tag) {
        case BasicBlock_TAG: {
            assert(false);
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, old);
}

Module* opt_mem2reg(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* initial_arena = get_module_arena(src);
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = src;

    for (size_t round = 0; round < 5; round++) {
        dst = new_module(a, get_module_name(src));

        Context ctx = {
            .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
            .bb_new_args = new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
            .a = new_arena(),

            .todo_jumps = NULL,
        };

        ctx.rewriter.config.fold_quote = false;
        // ctx.rewriter.config.rebind_let = false;

        rewrite_module(&ctx.rewriter);

        destroy_rewriter(&ctx.rewriter);
        destroy_dict(ctx.bb_new_args);
        destroy_arena(ctx.a);

        verify_module(dst);

        if (get_module_arena(src) != initial_arena)
            destroy_ir_arena(get_module_arena(src));

        dst = cleanup(config, dst);
        src = dst;
    }

    destroy_ir_arena(a);

    return dst;
}
