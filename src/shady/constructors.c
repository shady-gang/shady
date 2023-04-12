#include "ir_private.h"
#include "type.h"
#include "log.h"
#include "fold.h"
#include "portability.h"

#include "dict.h"

#include <string.h>
#include <assert.h>

Strings import_strings(IrArena*, Strings);

#define VISIT_FIELD_SCRATCH(t, n)
#define VISIT_FIELD_POD(t, n)
#define VISIT_FIELD_STRING(t, n) payload->n = string(arena, payload->n);
#define VISIT_FIELD_STRINGS(t, n) payload->n = import_strings(arena, payload->n);
#define VISIT_FIELD_ANNOTATIONS(t, n)
#define VISIT_FIELD_TYPE(t, n)
#define VISIT_FIELD_TYPES(t, n)
#define VISIT_FIELD_VALUE(t, n)
#define VISIT_FIELD_VALUES(t, n)
#define VISIT_FIELD_VARIABLES(t, n)
#define VISIT_FIELD_INSTRUCTION(t, n)
#define VISIT_FIELD_TERMINATOR(t, n)
#define VISIT_FIELD_ANON_LAMBDA(t, n)
#define VISIT_FIELD_ANON_LAMBDAS(t, n)

#define VISIT_FIELD_DECL(t, n)

#define VISIT_FIELD_BASIC_BLOCK(t, n)
#define VISIT_FIELD_BASIC_BLOCKS(t, n)

static void intern_strings(IrArena* arena, Node* node) {
    switch (node->tag) {
        case InvalidNode_TAG: SHADY_UNREACHABLE;
        #define VISIT_FIELD(hash, ft, t, n) VISIT_FIELD_##ft(t, n)
        #define VISIT_NODE_0(StructName, short_name) case StructName##_TAG: break;
        #define VISIT_NODE_1(StructName, short_name) case StructName##_TAG: { SHADY_UNUSED StructName* payload = &node->payload.short_name; StructName##_Fields(VISIT_FIELD) break; }
        #define VISIT_NODE(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) VISIT_NODE_##has_payload(StructName, short_name)
        NODES(VISIT_NODE)
    }
}

static Node* create_node_helper(IrArena* arena, Node node, bool* pfresh) {
    intern_strings(arena, &node);

    if (pfresh)
        *pfresh = false;

    Node* ptr = &node;
    Node** found = find_key_dict(Node*, arena->node_set, ptr);
    // sanity check nominal nodes to be unique, check for duplicates in structural nodes
    if (is_nominal(&node))
        assert(!found);
    else if (found)
        return *found;

    if (pfresh)
        *pfresh = true;

    if (arena->config.allow_fold) {
        Node* folded = (Node*) fold_node(arena, ptr);
        if (folded != ptr) {
            // The folding process simplified the node, we store a mapping to that simplified node and bail out !
            insert_set_get_result(Node*, arena->node_set, folded);
            return folded;
        }
    }

    if (arena->config.check_types && node.type)
        assert(is_type(node.type));

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
    return create_node_helper(arena, node, NULL);                                                                                                           \
}

#define NODE_CTOR_0(has_typing_fn, has_payload, struct_name, short_name)
#define NODE_CTOR(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) NODE_CTOR_##autogen_ctor(has_typing_fn, has_payload, struct_name, short_name)
NODES(NODE_CTOR)
#undef NODE_CTOR

const Node* let(IrArena* arena, const Node* instruction, const Node* tail) {
    Let payload = {
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
    return create_node_helper(arena, node, NULL);
}

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
    return create_node_helper(arena, node, NULL);
}

const Node* let_mut(IrArena* arena, const Node* instruction, const Node* tail) {
    LetMut payload = {
        .instruction = instruction,
        .tail = tail,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = NULL,
        .tag = LetMut_TAG,
        .payload.let_mut = payload
    };
    return create_node_helper(arena, node, NULL);
}

const Node* composite(IrArena* arena, const Type* elem_type, Nodes contents) {
    Composite c = {
        .type = elem_type,
        .contents = contents,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_composite(arena, c) : NULL,
        .tag = Composite_TAG,
        .payload.composite = c
    };
    return create_node_helper(arena, node, NULL);
}

const Node* tuple(IrArena* arena, Nodes contents) {
    const Type* t = NULL;
    if (arena->config.check_types) {
        // infer the type of the tuple
        Nodes member_types = get_values_types(arena, contents);
        t = record_type(arena, (RecordType) {.members = strip_qualifiers(arena, member_types)});
    }

    return composite(arena, t, contents);
}

Node* function(Module* mod, Nodes params, const char* name, Nodes annotations, Nodes return_types) {
    assert(!mod->sealed);
    IrArena* arena = mod->arena;
    Function payload = {
        .module = mod,
        .params = params,
        .body = NULL,
        .name = name,
        .annotations = annotations,
        .return_types = return_types,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_fun(arena, payload) : NULL,
        .tag = Function_TAG,
        .payload.fun = payload
    };
    Node* fn = create_node_helper(arena, node, NULL);
    register_decl_module(mod, fn);

    for (size_t i = 0; i < params.count; i++) {
        Node* param = (Node*) params.nodes[i];
        assert(param->tag == Variable_TAG);
        assert(!param->payload.var.abs);
        param->payload.var.abs = fn;
        param->payload.var.pindex = i;
    }

    return fn;
}

Node* basic_block(IrArena* arena, Node* fn, Nodes params, const char* name) {
    assert(!fn->payload.fun.module->sealed);
    BasicBlock payload = {
        .params = params,
        .body = NULL,
        .fn = fn,
        .name = name,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_basic_block(arena, payload) : NULL,
        .tag = BasicBlock_TAG,
        .payload.basic_block = payload
    };

    Node* bb = create_node_helper(arena, node, NULL);

    for (size_t i = 0; i < params.count; i++) {
        Node* param = (Node*) params.nodes[i];
        assert(param->tag == Variable_TAG);
        assert(!param->payload.var.abs);
        param->payload.var.abs = bb;
        param->payload.var.pindex = i;
    }

    return bb;
}

const Node* lambda(IrArena* a, Nodes params, const Node* body) {
    AnonLambda payload = {
        .params = params,
        .body = body,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = a,
        .type = a->config.check_types ? check_type_anon_lam(a, payload) : NULL,
        .tag = AnonLambda_TAG,
        .payload.anon_lam = payload
    };

    bool fresh;
    const Node* lam = create_node_helper(a, node, &fresh);

    if (fresh || true) {
        for (size_t i = 0; i < params.count; i++) {
            Node* param = (Node*) params.nodes[i];
            assert(param->tag == Variable_TAG);
            assert(!param->payload.var.abs);
            param->payload.var.abs = lam;
            param->payload.var.pindex = i;
        }
    }

    return lam;
}

Node* constant(Module* mod, Nodes annotations, const Type* hint, String name) {
    IrArena* arena = mod->arena;
    Constant cnst = {
        .annotations = annotations,
        .name = string(arena, name),
        .type_hint = hint,
        .value = NULL,
    };
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = arena->config.check_types ? check_type_constant(arena, cnst) : NULL,
        .tag = Constant_TAG,
        .payload.constant = cnst
    };
    Node* decl = create_node_helper(arena, node, NULL);
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
    Node* decl = create_node_helper(arena, node, NULL);
    register_decl_module(mod, decl);
    return decl;
}

Type* nominal_type(Module* mod, Nodes annotations, String name) {
    IrArena* arena = mod->arena;
    NominalType payload = {
        .name = string(arena, name),
        .module = mod,
        .annotations = annotations,
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
    Node* decl = create_node_helper(arena, node, NULL);
    register_decl_module(mod, decl);
    return decl;
}

const Node* quote(IrArena* arena, Nodes values) {
    for (size_t i = 0; i < values.count; i++)
        assert(is_value(values.nodes[i]));

    return prim_op(arena, (PrimOp) {
        .op = quote_op,
        .type_arguments = nodes(arena, 0, NULL),
        .operands = values
    });
}

const Node* quote_single(IrArena* arena, const Node* value) {
    return quote(arena, singleton(value));
}

const Node* unit(IrArena* arena) {
    return quote(arena, empty(arena));
}

const Node* unit_type(IrArena* arena) {
     return record_type(arena, (RecordType) {
         .members = empty(arena),
         .special = MultipleReturn,
     });
}

const Type* int8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = true }); }
const Type* int16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = true }); }
const Type* int32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = true }); }
const Type* int64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = true }); }

const Type* uint8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = false }); }
const Type* uint16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = false }); }
const Type* uint32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = false }); }
const Type* uint64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = false }); }

const Type* int8_literal(IrArena* arena, int8_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,    .value.i64 = i, .is_signed = true }); }
const Type* int16_literal(IrArena* arena, int16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value.i64 = i, .is_signed = true }); }
const Type* int32_literal(IrArena* arena, int32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value.i64 = i, .is_signed = true }); }
const Type* int64_literal(IrArena* arena, int64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value.i64 = i, .is_signed = true }); }

const Type* uint8_literal(IrArena* arena, uint8_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,    .value.u64 = i }); }
const Type* uint16_literal(IrArena* arena, uint16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value.u64 = i }); }
const Type* uint32_literal(IrArena* arena, uint32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value.u64 = i }); }
const Type* uint64_literal(IrArena* arena, uint64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value.u64 = i }); }

const Type* fp16_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy16 }); }
const Type* fp32_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy32 }); }
const Type* fp64_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy64 }); }
