#define REWRITE_FIELD_POD(t, n) .n = old_payload.n,
#define REWRITE_FIELD_TYPE(t, n) .n = rewrite_type(rewriter, old_payload.n),
#define REWRITE_FIELD_TYPES(t, n) .n = rewrite_nodes_generic(rewriter, rewrite_type, old_payload.n),
#define REWRITE_FIELD_VALUE(t, n) .n = rewrite_value(rewriter, old_payload.n),
#define REWRITE_FIELD_VALUES(t, n) .n = rewrite_nodes_generic(rewriter, rewrite_value, old_payload.n),
#define REWRITE_FIELD_INSTRUCTION(t, n) .n = rewrite_instruction(rewriter, old_payload.n),
#define REWRITE_FIELD_TERMINATOR(t, n) .n = rewrite_terminator(rewriter, old_payload.n),
#define REWRITE_FIELD_DECL(t, n) .n = rewrite_decl(rewriter, old_payload.n),
#define REWRITE_FIELD_ANON_LAMBDA(t, n) .n = rewrite_anon_lambda(rewriter, old_payload.n),
#define REWRITE_FIELD_ANON_LAMBDAS(t, n) .n = rewrite_nodes_generic(rewriter, rewrite_anon_lambda, old_payload.n),
#define REWRITE_FIELD_BASIC_BLOCK(t, n) .n = rewrite_basic_block(rewriter, old_payload.n),
#define REWRITE_FIELD_BASIC_BLOCKS(t, n) .n = rewrite_nodes_generic(rewriter, rewrite_basic_block, old_payload.n),
#define REWRITE_FIELD_STRING(t, n) .n = string(arena, old_payload.n),
#define REWRITE_FIELD_STRINGS(t, n) .n = import_strings(arena, old_payload.n),
#define REWRITE_FIELD_ANNOTATIONS(t, n) .n = rewrite_nodes_generic(rewriter, rewrite_annotation, old_payload.n),

switch (node->tag) {
    case InvalidNode_TAG:   assert(false);
    #define REWRITE_FIELD(hash, ft, t, n) REWRITE_FIELD_##ft(t, n)
    #define REWRITE_NODE_0_0(StructName, short_name)
    #define REWRITE_NODE_0_1(StructName, short_name) case StructName##_TAG: return short_name(arena);
    #define REWRITE_NODE_1_0(StructName, short_name)
    #define REWRITE_NODE_1_1(StructName, short_name) case StructName##_TAG: { StructName old_payload = node->payload.short_name; return short_name(arena, (StructName) { StructName##_Fields(REWRITE_FIELD) }); }
    #define REWRITE_NODE(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) REWRITE_NODE_##has_payload##_##autogen_ctor(StructName, short_name)
    NODES(REWRITE_NODE)
    case Function_TAG:
    case Constant_TAG:
    case GlobalVariable_TAG: {
        Node* new = recreate_decl_header_identity(rewriter, node);
        recreate_decl_body_identity(rewriter, node, new);
        return new;
    }
    case NominalType_TAG: error("TODO")
    case Variable_TAG: return var(arena, rewrite_type(rewriter, node->payload.var.type), node->payload.var.name);
    case Tuple_TAG: return tuple(arena, rewrite_nodes_generic(rewriter, rewrite_value, node->payload.tuple.contents));
    case Let_TAG: {
        const Node* instruction = rewrite_instruction(rewriter, node->payload.let.instruction);
        const Node* tail = rewrite_anon_lambda(rewriter, node->payload.let.tail);
        return let(arena, instruction, tail);
    }
    case LetMut_TAG: error("De-sugar this by hand")
    case LetIndirect_TAG: {
        const Node* instruction = rewrite_instruction(rewriter, node->payload.let.instruction);
        const Node* tail = rewrite_value(rewriter, node->payload.let.tail);
        return let(arena, instruction, tail);
    }
    case AnonLambda_TAG: {
        Nodes params = rewrite_nodes_generic(rewriter, rewrite_value, node->payload.anon_lam.params);
        Node* lam = lambda(arena, params);
        lam->payload.anon_lam.body = rewrite_terminator(rewriter, node->payload.anon_lam.body);
        return lam;
    }
    case BasicBlock_TAG: {
        Nodes params = rewrite_nodes_generic(rewriter, rewrite_value, node->payload.basic_block.params);
        const Node* fn = rewrite_decl(rewriter, node->payload.basic_block.fn);
        Node* lam = basic_block(arena, fn, params, node->payload.basic_block.name);
        lam->payload.anon_lam.body = rewrite_terminator(rewriter, node->payload.basic_block.body);
        return lam;
    }
}