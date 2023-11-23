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
    bool leaks;
    bool read_from;
    const Type* type;
} PtrSourceKnowledge;

typedef struct {
    const Node* ptr_address;
    const Node* ptr_value;
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
    Rewriter rewriter;
    Scope* scope;
    const UsesMap* scope_uses;
    struct Dict* abs_to_kb;
    const Node* abs;
    Arena* a;

    struct Dict* bb_new_args;
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

static PtrKnowledge* create_ptr_knowledge(KnowledgeBase* kb, const Node* instruction, const Node* address_value) {
    PtrKnowledge* k = arena_alloc(kb->a, sizeof(PtrKnowledge));
    PtrSourceKnowledge* sk = arena_alloc(kb->a, sizeof(PtrSourceKnowledge));
    *k = (PtrKnowledge) { .source = sk, .ptr_address = address_value };
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

static const Node* get_known_value(Rewriter* r, const PtrKnowledge* k) {
    const Node* v = NULL;
    if (k && !k->ptr_has_leaked && !k->source->leaks) {
        if (k->ptr_value) {
            v = k->ptr_value;
        }
    }
    if (r && v && v->arena != r->dst_arena)
        return rewrite_node(r, v);
    return v;
}

static const Node* get_known_address(Rewriter* r, const PtrKnowledge* k) {
    const Node* v = NULL;
    if (k) {
        if (k->ptr_address) {
            v = k->ptr_address;
        }
    }
    if (r && v && v->arena != r->dst_arena)
        return rewrite_node(r, v);
    return v;
}

static void visit_ptr_uses(const Node* ptr_value, PtrSourceKnowledge* k, const UsesMap* map) {
    const Use* use = get_first_use(map, ptr_value);
    for (;use; use = use->next_use) {
        if (is_abstraction(use->user) && use->operand_class == NcVariable)
            continue;
        else if (use->user->tag == Let_TAG && use->operand_class == NcInstruction) {
            Nodes vars = get_abstraction_params(get_let_tail(use->user));
            for (size_t i = 0; i < vars.count; i++) {
                debugv_print("mem2reg leak analysis: following let-bound variable: ");
                log_node(DEBUGV, vars.nodes[i]);
                debugv_print(".\n");
                visit_ptr_uses(vars.nodes[i], k, map);
            }
        } else if (use->user->tag == PrimOp_TAG) {
            PrimOp payload = use->user->payload.prim_op;
            switch (payload.op) {
                case load_op: {
                    k->read_from = true;
                    continue; // loads don't leak the address.
                }
                case store_op: {
                    // stores leak the value if it's stored
                    if (ptr_value == payload.operands.nodes[1])
                        k->leaks = true;
                    continue;
                }
                case reinterpret_op: {
                    debugv_print("mem2reg leak analysis: following reinterpret instr: ");
                    log_node(DEBUGV, use->user);
                    debugv_print(".\n");
                    visit_ptr_uses(use->user, k, map);
                    continue;
                }
                case lea_op:
                case convert_op: {
                    //TODO: follow where those derived pointers are used and establish whether they leak themselves
                    k->leaks = true;
                    continue;
                } default: break;
            }
            switch (payload.op) {
    #define P0(name) break;
    #define P1(name) case name##_op: k->leaks = true; break;
    #define P(has_side_effects, name) P##has_side_effects(name)
                PRIMOPS(P)
                default: break;
            }
        } else if (use->user->tag == Composite_TAG) {
            // todo...
            k->leaks = true;
        } else {
            k->leaks = true;
        }
    }
}

static void visit_instruction(Context* ctx, KnowledgeBase* kb, const Node* instruction, Nodes results) {
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
                    const Node* optr = first(results);
                    PtrKnowledge* k = create_ptr_knowledge(kb, instruction, optr);
                    visit_ptr_uses(optr, k->source, ctx->scope_uses);
                    debugv_print("mem2reg: ");
                    log_node(DEBUGV, optr);
                    if (k->source->leaks)
                        debugv_print(" is leaking so it will not be eliminated.\n");
                    else
                        debugv_print(" was found to not leak.\n");
                    const Type* t = instruction->type;
                    bool u = deconstruct_qualified_type(&t);
                    assert(t->tag == PtrType_TAG);
                    k->source->as = t->payload.ptr_type.address_space;
                    deconstruct_pointer_type(&t);
                    k->source->type = qualified_type_helper(t, u);

                    insert_ptr_knowledge(kb, optr, k);
                    k->ptr_value = undef(a, (Undef) { .type = first(payload.type_arguments) });
                    break;
                }
                case load_op: {
                    const Node* ptr = first(payload.operands);
                    const PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, ptr);
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
                    const Node* ptr = first(payload.operands);
                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, ptr);
                    if (k) {
                        k = update_ptr_knowledge(kb, ptr, k);
                        k->ptr_value = payload.operands.nodes[1];
                    }
                    break; // let's take care of dead stores another time
                }
                case reinterpret_op: {
                    // if we have knowledge on a particular ptr, the same knowledge propagates if we bitcast it!
                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, first(payload.operands));
                    if (k) {
                        k = update_ptr_knowledge(kb, instruction, k);
                        k->ptr_address = first(results);
                        insert_ptr_knowledge(kb, first(results), k);
                    }
                    break;
                }
                case convert_op: {
                    // if we convert a pointer to generic AS, we'd like to use the old address instead where possible
                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, first(payload.operands));
                    if (k) {
                        debug_print("mem2reg: the converted ptr ");
                        log_node(DEBUG, first(results));
                        debug_print(" is the same as ");
                        log_node(DEBUG, first(payload.operands));
                        debug_print(".\n");
                        k = update_ptr_knowledge(kb, instruction, k);
                        k->ptr_address = first(payload.operands);
                        insert_ptr_knowledge(kb, first(results), k);
                    }
                    break;
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
}

static void visit_terminator(Context* ctx, KnowledgeBase* kb, const Node* old) {
    if (!old)
        return;
    switch (is_terminator(old)) {
        case Terminator_LetMut_TAG:
        case NotATerminator: assert(false);
        case Terminator_Let_TAG: {
            const Node* otail = get_let_tail(old);
            visit_instruction(ctx, kb, get_let_instruction(old), get_abstraction_params(otail));
            break;
        }
        default:
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
        .map = new_dict(const Node*, PtrKnowledge*, (HashFn) hash_node, (CmpFn) compare_node),
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
    assert(is_abstraction(oabs));
    visit_terminator(ctx, kb, get_abstraction_body(oabs));

    for (size_t i = 0; i < entries_count_list(node->dominates); i++) {
        CFNode* dominated = read_list(CFNode*, node->dominates)[i];
        visit_cfnode(ctx, dominated, node);
    }
}

static const Node* process(Context* ctx, const Node* old) {
    assert(old);
    Context fn_ctx = *ctx;
    if (old->tag == Function_TAG && !lookup_annotation(old, "Internal")) {
        ctx = &fn_ctx;
        fn_ctx.scope = new_scope(old);
        fn_ctx.scope_uses = create_uses_map(old, (NcDeclaration | NcType));
        fn_ctx.abs_to_kb = new_dict(const Node*, KnowledgeBase**, (HashFn) hash_node, (CmpFn) compare_node);
        visit_cfnode(&fn_ctx, fn_ctx.scope->entry, NULL);
        fn_ctx.abs = old;
        const Node* new_fn = recreate_node_identity(&fn_ctx.rewriter, old);
        destroy_scope(fn_ctx.scope);
        destroy_uses_map(fn_ctx.scope_uses);
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
                    const Node* ptr = first(payload.operands);
                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, ptr);
                    const Node* known_value = get_known_value(&ctx->rewriter, k);
                    if (known_value) {
                        const Type* known_value_t = known_value->type;
                        bool kv_u = deconstruct_qualified_type(&known_value_t);

                        const Type* load_result_t = ptr->type;
                        bool lrt_u = deconstruct_qualified_type(&load_result_t);
                        deconstruct_pointer_type(&load_result_t);
                        assert(!lrt_u || kv_u);
                        if (is_reinterpret_cast_legal(load_result_t, known_value_t))
                            return prim_op_helper(a, reinterpret_op, singleton(rewrite_node(&ctx->rewriter, load_result_t)), singleton(known_value));
                    }
                    const Node* other_ptr = get_known_address(&ctx->rewriter, k);
                    if (other_ptr && ptr != other_ptr) {
                        return prim_op_helper(a, load_op, empty(a), singleton(other_ptr));
                    }
                    break;
                }
                case store_op: {
                    const Node* ptr = first(payload.operands);
                    const PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, ptr);
                    if (k && !k->source->leaks && !k->source->read_from)
                        return quote_helper(a, empty(a));
                    const Node* other_ptr = get_known_address(&ctx->rewriter, k);
                    if (other_ptr && ptr != other_ptr) {
                        return prim_op_helper(a, store_op, empty(a), mk_nodes(a, other_ptr, rewrite_node(&ctx->rewriter, payload.operands.nodes[1])));
                    }
                    break;
                }
                case alloca_op: {
                    const PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, old);
                    if (k && !k->source->leaks && !k->source->read_from)
                        return quote_helper(a, singleton(undef(a, (Undef) { .type = get_unqualified_type(rewrite_node(&ctx->rewriter, old->type)) })));
                    break;
                }
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
                PtrSourceKnowledge* source = NULL;
                PtrKnowledge uk = { 0 };
                // check if all the edges have a value for this!
                for (size_t j = 0; j < entries_count_list(cfnode->pred_edges); j++) {
                    CFEdge edge = read_list(CFEdge, cfnode->pred_edges)[j];
                    if (edge.type == StructuredPseudoExitEdge)
                        continue; // these are not real edges...
                    KnowledgeBase* kb_at_src = get_kb(ctx, edge.src->node);

                    if (get_known_value(NULL, get_last_valid_ptr_knowledge(kb_at_src, ptr))) {
                        log_node(DEBUG, ptr);
                        debug_print(" has a known value in %s ...\n", get_abstraction_name(edge.src->node));
                    } else
                        goto next_potential_param;

                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb_at_src, ptr);
                    if (!source)
                        source = k->source;
                    else
                        assert(source == k->source);

                    const Type* kv_type = get_known_value(NULL, get_last_valid_ptr_knowledge(kb_at_src, ptr))->type;
                    deconstruct_qualified_type(&kv_type);
                    const Type* alloca_type_t = source->type;
                    deconstruct_qualified_type(&alloca_type_t);
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

                const Node* param = var(a, rewrite_node(&ctx->rewriter, source->type), unique_name(a, "ssa_phi"));
                params = append_nodes(a, params, param);
                ptrs = append_nodes(ctx->rewriter.src_arena, ptrs, ptr);

                PtrKnowledge* k = arena_alloc(ctx->a, sizeof(PtrKnowledge));
                *k = (PtrKnowledge) {
                    .ptr_value = param,
                    .source = source,
                    .ptr_has_leaked = uk.ptr_has_leaked
                };
                insert_ptr_knowledge(kb, ptr, k);

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
                assert(additional_ssa_params->count > 0);

                LARRAY(const Type*, tr_params_arr, args.count);
                for (size_t i = 0; i < args.count; i++)
                    tr_params_arr[i] = var(a, args.nodes[i]->type, args.nodes[i]->payload.var.name);
                Nodes tr_params = nodes(a, args.count, tr_params_arr);
                Node* fn = (Node*) rewrite_node(&ctx->rewriter, ctx->scope->entry->node);
                Node* trampoline = basic_block(a, fn, tr_params, format_string_interned(a, "%s_trampoline", get_abstraction_name(new_bb)));
                Nodes tr_args = args;
                BodyBuilder* bb = begin_body(a);

                for (size_t i = 0; i < additional_ssa_params->count; i++) {
                    const Node* ptr = additional_ssa_params->nodes[i];
                    PtrKnowledge* k = get_last_valid_ptr_knowledge(kb, ptr);
                    const Node* value = get_known_value(&ctx->rewriter, k);

                    const Type* known_value_t = value->type;
                    deconstruct_qualified_type(&known_value_t);

                    const Type* alloca_type_t = k->source->type;
                    deconstruct_qualified_type(&alloca_type_t);

                    if (alloca_type_t != known_value_t && is_reinterpret_cast_legal(alloca_type_t, known_value_t))
                        value = first(gen_primop(bb, reinterpret_op, singleton(rewrite_node(&ctx->rewriter, alloca_type_t)), singleton(value)));

                    assert(value);
                    args = append_nodes(a, args, value);
                }

                trampoline->payload.basic_block.body = finish_body(bb, jump_helper(a, new_bb, args));

                return jump_helper(a, trampoline, tr_args);
            }

            return jump_helper(a, new_bb, args);
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

    for (size_t round = 0; round < 2; round++) {
        dst = new_module(a, get_module_name(src));

        Context ctx = {
            .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
            .bb_new_args = new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
            .a = new_arena(),
        };

        ctx.rewriter.config.fold_quote = false;

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
