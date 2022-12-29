#include "shady/ir.h"
#include "log.h"
#include "visit.h"

#include "analysis/scope.h"

#include <assert.h>

static void visit_node(Visitor* visitor, const Node* node) {
    if (node && visitor->visit_fn)
        visitor->visit_fn(visitor, node);
}

void visit_nodes(Visitor* visitor, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++) {
         visit_node(visitor, nodes.nodes[i]);
    }
}

void visit_fn_blocks_except_head(Visitor* visitor, const Node* function) {
    assert(function->tag == Function_TAG);
    Scope* scope = new_scope(function);
    assert(scope->rpo[0]->node == function);
    for (size_t i = 1; i < scope->size; i++) {
        visit_node(visitor, scope->rpo[i]->node);
    }
    destroy_scope(scope);
}

#pragma GCC diagnostic error "-Wswitch"

#define visit_type(n) visit_node(visitor, n)
#define visit_types(ns) visit_nodes(visitor, ns)
#define visit_value(n) visit_node(visitor, n)
#define visit_values(ns) visit_nodes(visitor, ns)
#define visit_instruction(n) visit_node(visitor, n)
#define visit_terminator(n) visit_node(visitor, n)
#define visit_decl(n) visit_node(visitor, n)
#define visit_anon_lambda(n) visit_node(visitor, n)
#define visit_anon_lambdas(ns) visit_nodes(visitor, ns)
#define visit_basic_block(n) visit_node(visitor, n)
#define visit_basic_blocks(ns) visit_nodes(visitor, ns)

#define VISIT_FIELD_POD(t, n)
#define VISIT_FIELD_STRING(t, n)
#define VISIT_FIELD_STRINGS(t, n)
#define VISIT_FIELD_ANNOTATIONS(t, n)
#define VISIT_FIELD_TYPE(t, n) visit_type(payload.n);
#define VISIT_FIELD_TYPES(t, n) visit_types(payload.n);
#define VISIT_FIELD_VALUE(t, n) visit_value(payload.n);
#define VISIT_FIELD_VALUES(t, n) visit_values(payload.n);
#define VISIT_FIELD_VARIABLES(t, n) visit_values(payload.n);
#define VISIT_FIELD_INSTRUCTION(t, n) visit_instruction(payload.n);
#define VISIT_FIELD_TERMINATOR(t, n) visit_terminator(payload.n);
#define VISIT_FIELD_ANON_LAMBDA(t, n) visit_anon_lambda(payload.n);
#define VISIT_FIELD_ANON_LAMBDAS(t, n) visit_anon_lambdas(payload.n);

#define VISIT_FIELD_DECL(t, n) if (visitor->visit_referenced_decls) visit_decl(payload.n);

#define VISIT_FIELD_BASIC_BLOCK(t, n) if (visitor->visit_continuations) visit_basic_block(payload.n);
#define VISIT_FIELD_BASIC_BLOCKS(t, n) if (visitor->visit_continuations) visit_basic_blocks(payload.n);

void visit_children(Visitor* visitor, const Node* node) {
    if (!node_type_has_payload[node->tag])
        return;

    if (node->tag == Function_TAG) {
        visit_nodes(visitor, node->payload.fun.params);
        visit_nodes(visitor, node->payload.fun.return_types);
        visit_node(visitor, node->payload.fun.body);
        if (visitor->visit_fn_scope_rpo)
            visit_fn_blocks_except_head(visitor, node);
    }

    switch(node->tag) {
        case InvalidNode_TAG: error("")
        #define VISIT_FIELD(hash, ft, t, n) VISIT_FIELD_##ft(t, n)
        #define VISIT_NODE_0(StructName, short_name) case StructName##_TAG: return;
        #define VISIT_NODE_1(StructName, short_name) case StructName##_TAG: { StructName payload = node->payload.short_name; StructName##_Fields(VISIT_FIELD) break; }
        #define VISIT_NODE(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) VISIT_NODE_##has_payload(StructName, short_name)
        NODES(VISIT_NODE)
    }
}

void visit_module(Visitor* visitor, Module* mod) {
    Nodes decls = get_module_declarations(mod);
    visit_nodes(visitor, decls);
}
