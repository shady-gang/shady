#include "type.h"
#include "log.h"
#include "arena.h"
#include "portability.h"

#include "murmur3.h"
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
    // place the node in the arena and return it
    Node* alloc = (Node*) arena_alloc(arena, sizeof(Node));
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
      .type = CALL_TYPING_METHOD_##has_typing_fn##has_payload(short_name),                                                                            \
      .tag = struct_name##_TAG,                                                                                                                       \
      SET_PAYLOAD_##has_payload(short_name)                                                                                                           \
    };                                                                                                                                                \
    return create_node_helper(arena, node);                                                                                                           \
}

#define NODE_CTOR_0(has_typing_fn, has_payload, struct_name, short_name)
#define NODEDEF(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) NODE_CTOR_##autogen_ctor(has_typing_fn, has_payload, struct_name, short_name)
NODES()
#undef NODEDEF

bool is_instruction(const Node* node) {
    switch (node->tag) {
#define NODEDEF(_, _2, _3, name, _4) case name##_TAG:
        INSTRUCTION_NODES()
#undef NODEDEF
            return true;
        default: return false;
    }
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
      .type = arena->config.check_types ? check_type_var(arena, variable) : NULL,
      .tag = Variable_TAG,
      .payload.var = variable
    };
    return create_node_helper(arena, node);
}

const Node* let(IrArena* arena, const Node* instruction, size_t outputs_count, const char* output_names[]) {
    assert(outputs_count > 0 && "do not use let if the outputs count isn't zero !");
    LARRAY(Node*, vars, outputs_count);

    if (arena->config.check_types) {
        Nodes types = typecheck_instruction(arena, instruction);
        for (size_t i = 0; i < outputs_count; i++)
            vars[i] = (Node*) var(arena, types.nodes[i], output_names ? output_names[i] : node_tags[instruction->tag]);
    } else {
        for (size_t i = 0; i < outputs_count; i++)
            vars[i] = (Node*) var(arena, NULL, output_names ? output_names[i] : node_tags[instruction->tag]);
    }

    for (size_t i = 0; i < outputs_count; i++) {
        vars[i]->payload.var.instruction = instruction;
        vars[i]->payload.var.output = i;
    }

    Let payload = {
        .instruction = instruction,
        .variables = nodes(arena, outputs_count, (const Node**) vars)
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
      .type = arena->config.check_types ? check_type_let(arena, payload) : NULL,
      .tag = Let_TAG,
      .payload.let = payload
    };
    return create_node_helper(arena, node);
}

Node* fn(IrArena* arena, FnAttributes attributes, const char* name, Nodes params, Nodes return_types) {
    Function fn = {
        .name = string(arena, name),
        .atttributes = attributes,
        .params = params,
        .return_types = return_types,
        .block = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
      .type = arena->config.check_types ? check_type_fn(arena, fn) : NULL,
      .tag = Function_TAG,
      .payload.fn = fn
    };
    return create_node_helper(arena, node);
}

Node* constant(IrArena* arena, String name) {
    Constant cnst = {
        .name = string(arena, name),
        .value = NULL,
        .type_hint = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
      .type = NULL,
      .tag = Constant_TAG,
      .payload.constant = cnst
    };
    return create_node_helper(arena, node);
}

Node* global_var(IrArena* arena, const Type* type, const char* name, AddressSpace as) {
    GlobalVariable gvar = {
        .name = string(arena, name),
        .type = type,
        .address_space = as,
        .init = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
      .type = arena->config.check_types ? check_type_global_variable(arena, gvar) : NULL,
      .tag = GlobalVariable_TAG,
      .payload.global_variable = gvar
    };
    return create_node_helper(arena, node);
}

const char* node_tags[] = {
#define NODEDEF(_, _2, _3, _4, str) #str,
NODES()
#undef NODEDEF
};

const char* primop_names[] = {
#define PRIMOP(str) #str,
PRIMOPS()
#undef PRIMOP
};

const bool node_type_has_payload[] = {
#define NODEDEF(_, _2, has_payload, _4, _5) has_payload,
NODES()
#undef NODEDEF
};

String merge_what_string[] = { "join", "continue", "break" };

KeyHash hash_murmur(const void* data, size_t size) {
    int32_t out[4];
    MurmurHash3_x64_128(data, (int) size, 0x1234567, &out);

    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

#define FIELDS                        \
case Variable_TAG: {                  \
    field(var.id);                    \
    break;                            \
}                                     \
case IntLiteral_TAG: {                \
    field(int_literal.value);         \
    break;                            \
}                                     \
case Let_TAG: {                       \
    field(let.variables);             \
    field(let.instruction);           \
    break;                            \
}                                     \
case QualifiedType_TAG: {             \
    field(qualified_type.type);       \
    field(qualified_type.is_uniform); \
    break;                            \
}                                     \
case FnType_TAG: {                    \
    field(fn_type.is_continuation);   \
    field(fn_type.return_types);      \
    field(fn_type.param_types);       \
    break;                            \
}                                     \
case PtrType_TAG: {                   \
    field(ptr_type.address_space);    \
    field(ptr_type.pointed_type);     \
    break;                            \
}                                     \

KeyHash hash_node(Node** pnode) {
    const Node* node = *pnode;
    KeyHash combined;

    if (is_nominal(node->tag)) {
        size_t ptr = (size_t) node;
        uint32_t upper = ptr >> 32;
        uint32_t lower = ptr;
        combined = upper ^ lower;
        goto end;
    }

    KeyHash tag_hash = hash_murmur(&node->tag, sizeof(NodeTag));
    KeyHash payload_hash = 0;

    #define field(d) payload_hash ^= hash_murmur(&node->payload.d, sizeof(node->payload.d));

    if (node_type_has_payload[node->tag]) {
        switch (node->tag) {
            FIELDS
            default: payload_hash = hash_murmur(&node->payload, sizeof(node->payload)); break;
        }
    }
    combined = tag_hash ^ payload_hash;

    end:
    // debug_print("hash of :");
    // debug_node(node);
    // debug_print(" = [%u] %u\n", combined, combined % 32);
    return combined;
}

bool compare_node(Node** pa, Node** pb) {
    if ((*pa)->tag != (*pb)->tag) return false;
    if (is_nominal((*pa)->tag)) {
        // debug_node(*pa);
        // debug_print(" vs ");
        // debug_node(*pb);
        // debug_print(" ptrs: %lu vs %lu %d\n", *pa, *pb, *pa == *pb);
        return *pa == *pb;
    }

    const Node* a = *pa;
    const Node* b = *pb;

    #undef field
    #define field(w) eq &= memcmp(&a->payload.w, &b->payload.w, sizeof(a->payload.w)) == 0;

    if (node_type_has_payload[a->tag]) {
        bool eq = true;
        switch ((*pa)->tag) {
            FIELDS
            default: return memcmp(&a->payload, &b->payload, sizeof(a->payload)) == 0;
        }
        return eq;
    } else return true;
}

String get_decl_name(const Node* node) {
    switch (node->tag) {
        case Constant_TAG: return node->payload.constant.name;
        case Function_TAG: return node->payload.fn.name;
        case Variable_TAG: return node->payload.var.name;
        default: return NULL;
    }
}

const IntLiteral* resolve_to_literal(const Node* node) {
    while (true) {
        switch (node->tag) {
            case Constant_TAG: return resolve_to_literal(node->payload.constant.value);
            case IntLiteral_TAG: {
                return &node->payload.int_literal;
            }
            default: return NULL;
        }
    }
}
