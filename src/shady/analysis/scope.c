#include "scope.h"
#include "log.h"

#include "list.h"
#include "dict.h"
#include "arena.h"
#include "../ir_private.h"

#include <stdlib.h>
#include <assert.h>

struct List* build_scopes(Module* mod) {
    struct List* scopes = new_list(Scope*);

    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != Function_TAG) continue;
        Scope* scope = new_scope(decl);
        append_list(Scope*, scopes, scope);
    }

    return scopes;
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

typedef struct {
    Arena* arena;
    const Node* entry;
    struct Dict* nodes;
    struct List* queue;
    struct List* contents;
} ScopeBuildContext;

CFNode* scope_lookup(Scope* scope, const Node* block) {
    CFNode** found = find_value_dict(const Node*, CFNode*, scope->map, block);
    if (found) return *found;
    assert(false);
}

static CFNode* get_or_enqueue(ScopeBuildContext* ctx, const Node* abs) {
    assert(is_abstraction(abs));
    assert(!is_function(abs) || abs == ctx->entry);
    CFNode** found = find_value_dict(const Node*, CFNode*, ctx->nodes, abs);
    if (found) return *found;

    CFNode* new = arena_alloc(ctx->arena, sizeof(CFNode));
    *new = (CFNode) {
        .node = abs,
        .succ_edges = new_list(CFEdge),
        .pred_edges = new_list(CFEdge),
        .rpo_index = SIZE_MAX,
        .idom = NULL,
        .dominates = NULL,
    };
    insert_dict(const Node*, CFNode*, ctx->nodes, abs, new);
    append_list(Node*, ctx->queue, new);
    append_list(Node*, ctx->contents, new);
    return new;
}

/// Adds an edge to somewhere inside a basic block
static void add_edge(ScopeBuildContext* ctx, const Node* src, const Node* dst, CFEdgeType type) {
    CFNode* src_node = get_or_enqueue(ctx, src);
    CFNode* dst_node = get_or_enqueue(ctx, dst);
    CFEdge edge = {
        .type = type,
        .src = src_node,
        .dst = dst_node,
    };
    append_list(CFEdge, src_node->succ_edges, edge);
    append_list(CFEdge, dst_node->pred_edges, edge);
}

static void process_instruction(ScopeBuildContext* ctx, CFNode* parent, const Node* instruction) {
    switch (is_instruction(instruction)) {
        case NotAnInstruction: if (instruction->arena->config.check_types) { error("Grammar problem"); } break;
        case Instruction_LeafCall_TAG:
        case Instruction_IndirectCall_TAG:
        case Instruction_PrimOp_TAG: break;
        case Instruction_If_TAG:
            add_edge(ctx, parent->node, instruction->payload.if_instr.if_true, IfBodyEdge);
            if(instruction->payload.if_instr.if_false)
                add_edge(ctx, parent->node, instruction->payload.if_instr.if_false, IfBodyEdge);
            break;
        case Instruction_Match_TAG:
            for (size_t i = 0; i < instruction->payload.match_instr.cases.count; i++)
                add_edge(ctx, parent->node, instruction->payload.match_instr.cases.nodes[i], MatchBodyEdge);
            add_edge(ctx, parent->node, instruction->payload.match_instr.default_case, MatchBodyEdge);
            break;
        case Instruction_Loop_TAG:
            add_edge(ctx, parent->node, instruction->payload.loop_instr.body, LoopBodyEdge);
            break;
        case Instruction_Control_TAG:
            add_edge(ctx, parent->node, instruction->payload.control.inside, ControlBodyEdge);
            break;
        case Instruction_Block_TAG:
            add_edge(ctx, parent->node, instruction->payload.block.inside, BlockBodyEdge);
            break;
    }
}

static void process_cf_node(ScopeBuildContext* ctx, CFNode* node) {
    const Node* const abs = node->node;
    assert(is_abstraction(abs));
    assert(!is_function(abs) || abs == ctx->entry);
    const Node* terminator = get_abstraction_body(abs);
    if (!terminator)
        return;
    switch (is_terminator(terminator)) {
        case Jump_TAG: {
            const Node* target = terminator->payload.jump.target;
            add_edge(ctx, abs, target, ForwardEdge);
            break;
        }
        case Branch_TAG: {
            const Node* true_target = terminator->payload.branch.true_target;
            const Node* false_target = terminator->payload.branch.false_target;
            add_edge(ctx, abs, true_target, ForwardEdge);
            add_edge(ctx, abs, false_target, ForwardEdge);
            break;
        }
        case Switch_TAG: error("TODO")
        case LetMut_TAG:
        case Let_TAG: {
            process_instruction(ctx, node, get_let_instruction(terminator));
            const Node* target = get_let_tail(terminator);
            add_edge(ctx, abs, target, LetTailEdge);
            break;
        }
        case Join_TAG: {
            break;
        }
        case MergeContinue_TAG:
        case MergeBreak_TAG:
        case Yield_TAG: {
            // error("TODO: only allow this if we have traversed structured constructs...")
            break;
        }
        case TailCall_TAG:
        case Return_TAG:
        case Unreachable_TAG: break;
        case NotATerminator: if (terminator->arena->config.check_types) { error("Grammar problem"); } break;
    }
}

Scope* new_scope(const Node* entry) {
    assert(is_abstraction(entry));
    Arena* arena = new_arena();

    ScopeBuildContext context = {
        .arena = arena,
        .entry = entry,
        .nodes = new_dict(const Node*, CFNode*, (HashFn) hash_node, (CmpFn) compare_node),
        .queue = new_list(CFNode*),
        .contents = new_list(CFNode*),
    };

    CFNode* entry_node = get_or_enqueue(&context, entry);

    while (entries_count_list(context.queue) > 0) {
        CFNode* this = pop_last_list(CFNode*, context.queue);
        process_cf_node(&context, this);
    }

    destroy_list(context.queue);

    Scope* scope = calloc(sizeof(Scope), 1);
    *scope = (Scope) {
        .arena = arena,
        .entry = entry_node,
        .size = entries_count_list(context.contents),
        .contents = context.contents,
        .map = context.nodes,
        .rpo = NULL
    };

    compute_rpo(scope);
    compute_domtree(scope);

    return scope;
}

void destroy_scope(Scope* scope) {
    for (size_t i = 0; i < scope->size; i++) {
        CFNode* node = read_list(CFNode*, scope->contents)[i];
        destroy_list(node->pred_edges);
        destroy_list(node->succ_edges);
        if (node->dominates)
            destroy_list(node->dominates);
    }
    destroy_dict(scope->map);
    destroy_arena(scope->arena);
    free(scope->rpo);
    destroy_list(scope->contents);
    free(scope);
}

static size_t post_order_visit(Scope* scope, CFNode* n, size_t i) {
    n->rpo_index = -2;

    for (size_t j = 0; j < entries_count_list(n->succ_edges); j++) {
        CFEdge edge = read_list(CFEdge, n->succ_edges)[j];
        if (edge.dst->rpo_index == SIZE_MAX)
            i = post_order_visit(scope, edge.dst, i);
    }

    n->rpo_index = i - 1;
    scope->rpo[n->rpo_index] = n;
    return n->rpo_index;
}

void compute_rpo(Scope* scope) {
    scope->rpo = malloc(sizeof(const CFNode*) * scope->size);
    size_t index = post_order_visit(scope,  scope->entry, scope->size);
    assert(index == 0);

    // debug_print("RPO: ");
    // for (size_t i = 0; i < scope->size; i++) {
    //     debug_print("%s, ", scope->rpo[i]->node->payload.lam.name);
    // }
    // debug_print("\n");
}

CFNode* least_common_ancestor(CFNode* i, CFNode* j) {
    assert(i && j);
    while (i->rpo_index != j->rpo_index) {
        while (i->rpo_index < j->rpo_index) j = j->idom;
        while (i->rpo_index > j->rpo_index) i = i->idom;
    }
    return i;
}

void compute_domtree(Scope* scope) {
    for (size_t i = 1; i < scope->size; i++) {
        CFNode* n = read_list(CFNode*, scope->contents)[i];
        for (size_t j = 0; j < entries_count_list(n->pred_edges); j++) {
            CFEdge e = read_list(CFEdge, n->pred_edges)[j];
            CFNode* p = e.src;
            if (p->rpo_index < n->rpo_index) {
                n->idom = p;
                goto outer_loop;
            }
        }
        error("no idom found");
        outer_loop:;
    }

    bool todo = true;
    while (todo) {
        todo = false;
        for (size_t i = 1; i < scope->size; i++) {
            CFNode* n = read_list(CFNode*, scope->contents)[i];
            CFNode* new_idom = NULL;
            for (size_t j = 0; j < entries_count_list(n->pred_edges); j++) {
                CFEdge e = read_list(CFEdge, n->pred_edges)[j];
                CFNode* p = e.src;
                new_idom = new_idom ? least_common_ancestor(new_idom, p) : p;
            }
            assert(new_idom);
            if (n->idom != new_idom) {
                n->idom = new_idom;
                todo = true;
            }
        }
    }

    for (size_t i = 0; i < scope->size; i++) {
        CFNode* n = read_list(CFNode*, scope->contents)[i];
        n->dominates = new_list(CFNode*);
    }
    for (size_t i = 1; i < scope->size; i++) {
        CFNode* n = read_list(CFNode*, scope->contents)[i];
        append_list(CFNode*, n->idom->dominates, n);
    }
}

static int extra_uniqueness = 0;

static void dump_cfg_scope(FILE* output, Scope* scope) {
    extra_uniqueness++;

    const Node* entry = scope->entry->node;
    fprintf(output, "subgraph cluster_%s {\n", get_abstraction_name(entry));
    fprintf(output, "label = \"%s\";\n", get_abstraction_name(entry));
    for (size_t i = 0; i < entries_count_list(scope->contents); i++) {
        const Node* bb = read_list(const CFNode*, scope->contents)[i]->node;
        const Node* body = get_abstraction_body(bb);
        if (!body) continue;

        String color = "black";
        if (is_anonymous_lambda(bb))
            color = "green";
        else if (is_basic_block(bb))
            color = "blue";

        String label = node_tags[body->tag];
        switch (body->tag) {
            case Let_TAG:
                body = body->payload.let.instruction;
                if (body->tag == PrimOp_TAG)
                    label = format_string(bb->arena, "%s ... = %s (...)", label, primop_names[body->payload.prim_op.op]);
                else
                    label = format_string(bb->arena, "%s ... = %s (...)", label, node_tags[body->tag]);
                break;
            default: break;
        }

        if (is_basic_block(bb)) {
            label = format_string(entry->arena, "%s\n%s", get_abstraction_name(bb), label);
        }
        fprintf(output, "bb_%zu [label=\"%s\", color=\"%s\"];\n", (size_t) bb, label, color);
    }
    for (size_t i = 0; i < entries_count_list(scope->contents); i++) {
        const CFNode* bb_node = read_list(const CFNode*, scope->contents)[i];
        const Node* bb = bb_node->node;

        for (size_t j = 0; j < entries_count_list(bb_node->succ_edges); j++) {
            CFEdge edge = read_list(CFEdge, bb_node->succ_edges)[j];
            const CFNode* target_node = edge.dst;
            const Node* target_bb = target_node->node;
            fprintf(output, "bb_%zu -> bb_%zu;\n", (size_t) (bb), (size_t) (target_bb));
        }
    }
    fprintf(output, "}\n");
}

void dump_cfg(FILE* output, Module* mod) {
    if (output == NULL)
        output = stderr;

    fprintf(output, "digraph G {\n");
    struct List* scopes = build_scopes(mod);
    for (size_t i = 0; i < entries_count_list(scopes); i++) {
        Scope* scope = read_list(Scope*, scopes)[i];
        dump_cfg_scope(output, scope);
        destroy_scope(scope);
    }
    destroy_list(scopes);
    fprintf(output, "}\n");
}
