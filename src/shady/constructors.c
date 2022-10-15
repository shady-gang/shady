#include "ir_private.h"
#include "type.h"
#include "log.h"
#include "fold.h"
#include "portability.h"

#include "dict.h"

#include <string.h>
#include <assert.h>

static Node* create_node_helper(IrArena* arena, Node node) {
    Node* ptr = &node;
    Node** found = find_key_dict(Node*, arena->node_set, ptr);
    // sanity check nominal nodes to be unique, check for duplicates in structural nodes
    if (is_nominal(node.tag))
        assert(!found);
    else if (found)
        return *found;

    if (arena->config.allow_fold) {
        Node* folded = (Node*) fold_node(arena, ptr);
        if (folded != ptr) {
            // The folding process simplified the node, we store a mapping to that simplified node and bail out !
            insert_set_get_result(Node*, arena->node_set, folded);
            return folded;
        }
    }

    // place the node in the arena and return it
    Node* alloc = (Node*) arena_alloc(arena->arena, sizeof(Node));
    *alloc = node;
    insert_set_get_result(const Node*, arena->node_set, alloc);

    return alloc;
}

#define LAST_ARG_1(struct_name) ,struct_name in_node
#define LAST_ARG_0(struct_name)

#define CALL_TYPING_METHOD_11(short_name) arena->config.check_types ? check_type_##short_name(arena, in_node) : NULL
#define CALL_TYPING_METHOD_01(short_name) NULL
#define CALL_TYPING_METHOD_10(short_name) arena->config.check_types ? check_type_##short_name(arena) : NULL
#define CALL_TYPING_METHOD_00(short_name) NULL

#define SET_PAYLOAD_1(short_name) .payload = (union NodesUnion) { .short_name = in_node }
#define SET_PAYLOAD_0(_)

#define NODE_CTOR_1(has_typing_fn, has_payload, struct_name, short_name) const Node* short_name(IrArena* arena LAST_ARG_##has_payload(struct_name)) { \
    Node node;                                                                                                                                        \
    memset((void*) &node, 0, sizeof(Node));                                                                                                           \
    node = (Node) {                                                                                                                                   \
        .arena = arena,                                                                                                                               \
        .type = CALL_TYPING_METHOD_##has_typing_fn##has_payload(short_name),                                                                          \
        .tag = struct_name##_TAG,                                                                                                                     \
        SET_PAYLOAD_##has_payload(short_name)                                                                                                         \
    };                                                                                                                                                \
    return create_node_helper(arena, node);                                                                                                           \
}

#define NODE_CTOR_0(has_typing_fn, has_payload, struct_name, short_name)
#define NODE_CTOR(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) NODE_CTOR_##autogen_ctor(has_typing_fn, has_payload, struct_name, short_name)
NODES(NODE_CTOR)
#undef NODE_CTOR

const Node* var(IrArena* arena, const Type* type, const char* name) {
    Variable variable = {
        .type = type,
        .name = string(arena, name),
        .id = fresh_id(arena)
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_var(arena, variable) : NULL,
        .tag = Variable_TAG,
        .payload.var = variable
    };
    return create_node_helper(arena, node);
}

static const Node* let_internal(IrArena* arena, bool is_mutable, Nodes* provided_types, const Node* instruction, size_t outputs_count, const char* output_names[]) {
    assert(outputs_count > 0 && "do not use let if the outputs count isn't zero !");
    LARRAY(Node*, vars, outputs_count);

    if (provided_types)
        assert(provided_types->count == outputs_count);

    if (arena->config.check_types) {
        Nodes types = unwrap_multiple_yield_types(arena, instruction->type);
        assert(types.count == outputs_count);
        if (provided_types) {
            // Check that the types we got are subtypes of what we care about
            for (size_t i = 0; i < outputs_count; i++)
                assert(is_subtype(provided_types->nodes[i], types.nodes[i]));
            types = *provided_types;
        }

        for (size_t i = 0; i < outputs_count; i++)
            vars[i] = (Node*) var(arena, types.nodes[i], output_names ? output_names[i] : node_tags[instruction->tag]);
    } else {
        for (size_t i = 0; i < outputs_count; i++)
            vars[i] = (Node*) var(arena, provided_types ? provided_types->nodes[i] : NULL, output_names ? output_names[i] : node_tags[instruction->tag]);
    }

    for (size_t i = 0; i < outputs_count; i++) {
        vars[i]->payload.var.instruction = instruction;
        vars[i]->payload.var.output = i;
    }

    Let payload = {
        .instruction = instruction,
        .variables = nodes(arena, outputs_count, (const Node**) vars),
        .is_mutable = is_mutable
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_let(arena, payload) : NULL,
        .tag = Let_TAG,
        .payload.let = payload
    };
    return create_node_helper(arena, node);
}

const Node* let(IrArena* arena, const Node* instruction, size_t outputs_count, const char* output_names[]) {
    return let_internal(arena, false, NULL, instruction, outputs_count, output_names);
}

const Node* let_mut(IrArena* arena, const Node* instruction, Nodes types, size_t outputs_count, const char* output_names[]) {
    return let_internal(arena, true, &types, instruction, outputs_count, output_names);
}

const Node* tuple(IrArena* arena, Nodes contents) {
    Tuple t = {
        .contents = contents
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_tuple(arena, t) : NULL,
        .tag = Tuple_TAG,
        .payload.tuple = t
    };
    return create_node_helper(arena, node);
}

static Node* lambda_internal(IrArena* arena, FnTier tier, Nodes params, const char* name, Nodes annotations, Nodes return_types) {
    Function fn = {
        .tier = tier,
        .params = params,
        .body = NULL,

        .name = string(arena, name),

        .annotations = annotations,
        .return_types = return_types,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_fn(arena, fn) : NULL,
        .tag = Function_TAG,
        .payload.fn = fn
    };
    return create_node_helper(arena, node);
}

Node* lambda(IrArena* arena, Nodes params) {
    return lambda_internal(arena, FnTier_Lambda, params, NULL, nodes(arena, 0, NULL), nodes(arena, 0, NULL));
}

Node* basic_block(IrArena* arena, Nodes params, const char* name) {
    return lambda_internal(arena, FnTier_BasicBlock, params, name, nodes(arena, 0, NULL), nodes(arena, 0, NULL));
}

Node* function   (IrArena* arena, Nodes params, const char* name, Nodes annotations, Nodes return_types) {
    return lambda_internal(arena, FnTier_Function, params, name, annotations, return_types);
}

Node* constant(IrArena* arena, Nodes annotations, String name) {
    Constant cnst = {
        .annotations = annotations,
        .name = string(arena, name),
        .value = NULL,
        .type_hint = NULL,
    };
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = NULL,
        .tag = Constant_TAG,
        .payload.constant = cnst
    };
    return create_node_helper(arena, node);
}

Node* global_var(IrArena* arena, Nodes annotations, const Type* type, const char* name, AddressSpace as) {
    GlobalVariable gvar = {
        .annotations = annotations,
        .name = string(arena, name),
        .type = type,
        .address_space = as,
        .init = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_global_variable(arena, gvar) : NULL,
        .tag = GlobalVariable_TAG,
        .payload.global_variable = gvar
    };
    return create_node_helper(arena, node);
}

Type* nominal_type(IrArena* arena, String name) {
    NominalType payload = {
        .name = string(arena, name),
        .body = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = NULL,
        .tag = NominalType_TAG,
        .payload.nom_type = payload
    };
    return create_node_helper(arena, node);
}

const Node* body(IrArena* arena, Nodes instructions, const Node* terminator, Nodes children_continuations) {
    Body b = {
        .instructions = instructions,
        .terminator = terminator,
        .children_continuations = children_continuations,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_body(arena, b) : NULL,
        .tag = Body_TAG,
        .payload.body = b
    };
    return create_node_helper(arena, node);
}

const Type* int8_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy8 }); }
const Type* int16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16 }); }
const Type* int32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32 }); }
const Type* int64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64 }); }

const Type* int8_literal(IrArena* arena, int8_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,    .value_i64 = i }); }
const Type* int16_literal(IrArena* arena, int16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value_i64 = i }); }
const Type* int32_literal(IrArena* arena, int32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value_i64 = i }); }
const Type* int64_literal(IrArena* arena, int64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value_i64 = i }); }

const Type* uint8_literal(IrArena* arena, uint8_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,    .value_u64 = i }); }
const Type* uint16_literal(IrArena* arena, uint16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value_u64 = i }); }
const Type* uint32_literal(IrArena* arena, uint32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value_u64 = i }); }
const Type* uint64_literal(IrArena* arena, uint64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value_u64 = i }); }
