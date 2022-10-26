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
    if (is_nominal(&node))
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

const Node* let(IrArena* arena, bool is_mutable, const Node* instruction, const Node* tail) {
    assert(is_instruction(instruction));
    Let payload = {
        .is_mutable = is_mutable,
        .instruction = instruction,
        .tail = tail,
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
    Lambda lam = {
        .tier = tier,
        .params = params,
        .body = NULL,
        .name = NULL,
        .annotations = nodes(arena, 0, NULL),
        .return_types = nodes(arena, 0, NULL),
    };

    if (tier >= FnTier_BasicBlock)
        lam.name = string(arena, name);
    else
        assert(name == NULL);

    if (tier >= FnTier_Function) {
        lam.annotations = annotations;
        lam.return_types = return_types;
    } else {
        assert(annotations.count == 0);
        assert(return_types.count == 0);
    }

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_lam(arena, lam) : NULL,
        .tag = Lambda_TAG,
        .payload.lam = lam
    };
    return create_node_helper(arena, node);
}

Node* lambda(IrArena* arena, Nodes params) {
    return lambda_internal(arena, FnTier_Lambda, params, NULL, nodes(arena, 0, NULL), nodes(arena, 0, NULL));
}

Node* basic_block(IrArena* arena, Nodes params, const char* name) {
    return lambda_internal(arena, FnTier_BasicBlock, params, name, nodes(arena, 0, NULL), nodes(arena, 0, NULL));
}

Node* function   (Module* mod, Nodes params, const char* name, Nodes annotations, Nodes return_types) {
    Node* fn = lambda_internal(mod->arena, FnTier_Function, params, name, annotations, return_types);
    register_decl_module(mod, fn);
    return fn;
}

Node* constant(Module* mod, Nodes annotations, String name) {
    IrArena* arena = mod->arena;
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
    Node* decl = create_node_helper(arena, node);
    register_decl_module(mod, decl);
    return decl;
}

Node* global_var(Module* mod, Nodes annotations, const Type* type, const char* name, AddressSpace as) {
    IrArena* arena = mod->arena;
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
    Node* decl = create_node_helper(arena, node);
    register_decl_module(mod, decl);
    return decl;
}

Type* nominal_type(Module* mod, String name) {
    IrArena* arena = mod->arena;
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
    Node* decl = create_node_helper(arena, node);
    register_decl_module(mod, decl);
    return decl;
}

const Node* quote(IrArena* arena, const Node* value) {
     assert(is_value(value));
     return prim_op(arena, (PrimOp) {
         .op = quote_op,
         .type_arguments = nodes(arena, 0, NULL),
         .operands = nodes(arena, 1, (const Node*[]){ value })
     });
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
