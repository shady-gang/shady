#include "type.h"
#include "log.h"
#include "ir_private.h"

#include "murmur3.h"
#include "dict.h"

#include <string.h>
#include <assert.h>

TypeTag is_type(const Node* node) {
    switch (node->tag) {
#define IS_TYPE(_, _2, _3, name, _4) case name##_TAG: return Type_##name##_TAG;
TYPE_NODES(IS_TYPE)
#undef IS_TYPE
        default: return NotAType;
    }
}

ValueTag is_value(const Node* node) {
    switch (node->tag) {
#define IS_VALUE(_, _2, _3, name, _4) case name##_TAG: return Value_##name##_TAG;
        VALUE_NODES(IS_VALUE)
#undef IS_VALUE
        default: return NotAValue;
    }
}

InstructionTag is_instruction(const Node* node) {
    switch (node->tag) {
#define IS_INSTRUCTION(_, _2, _3, name, _4) case name##_TAG: return Instruction_##name##_TAG;
        INSTRUCTION_NODES(IS_INSTRUCTION)
#undef IS_INSTRUCTION
        default: return NotAnInstruction;
    }
}

TerminatorTag is_terminator(const Node* node) {
    switch (node->tag) {
#define IS_TERMINATOR(_, _2, _3, name, _4) case name##_TAG: return Terminator_##name##_TAG;
        TERMINATOR_NODES(IS_TERMINATOR)
#undef IS_TERMINATOR
        default: return NotATerminator;
    }
}

const char* node_tags[] = {
#define NODE_NAME(_, _2, _3, _4, str) #str,
NODES(NODE_NAME)
#undef NODE_NAME
};

const char* primop_names[] = {
#define DECLARE_PRIMOP_NAME(se, str) #str,
PRIMOPS(DECLARE_PRIMOP_NAME)
#undef DECLARE_PRIMOP_NAME
};

const bool primop_side_effects[] = {
#define PRIMOP_SIDE_EFFECTFUL(se, str) se,
PRIMOPS(PRIMOP_SIDE_EFFECTFUL)
#undef PRIMOP_SIDE_EFFECTFUL
};

bool has_primop_got_side_effects(Op op) {
    return primop_side_effects[op];
}

const bool node_type_has_payload[] = {
#define NODE_HAS_PAYLOAD(_, _2, has_payload, _4, _5) has_payload,
NODES(NODE_HAS_PAYLOAD)
#undef NODE_HAS_PAYLOAD
};

String get_decl_name(const Node* node) {
    switch (node->tag) {
        case Constant_TAG: return node->payload.constant.name;
        case Lambda_TAG: {
            assert(node->payload.lam.tier != FnTier_Lambda && "lambdas are not decls");
            return node->payload.lam.name;
        }
        case GlobalVariable_TAG: return node->payload.global_variable.name;
        default: error("Not a decl !");
    }
}

int64_t extract_int_literal_value(const Node* node, bool sign_extend) {
    assert(node->tag == IntLiteral_TAG);
    if (sign_extend) {
        switch (node->payload.int_literal.width) {
            case IntTy8:  return (int64_t) node->payload.int_literal.value_i8;
            case IntTy16: return (int64_t) node->payload.int_literal.value_i16;
            case IntTy32: return (int64_t) node->payload.int_literal.value_i32;
            case IntTy64: return           node->payload.int_literal.value_i64;
            default: assert(false);
        }
    } else {
        switch (node->payload.int_literal.width) {
            case IntTy8:  return (int64_t) ((uint64_t) (node->payload.int_literal.value_u8 ));
            case IntTy16: return (int64_t) ((uint64_t) (node->payload.int_literal.value_u16));
            case IntTy32: return (int64_t) ((uint64_t) (node->payload.int_literal.value_u32));
            case IntTy64: return                        node->payload.int_literal.value_i64  ;
            default: assert(false);
        }
    }
}

const IntLiteral* resolve_to_literal(const Node* node) {
    while (true) {
        switch (node->tag) {
            case Constant_TAG:   return resolve_to_literal(node->payload.constant.value);
            case IntLiteral_TAG: return &node->payload.int_literal;
            default: return NULL;
        }
    }
}

const char* extract_string_literal(const Node* node) {
    assert(node->tag == StringLiteral_TAG);
    return node->payload.string_lit.string;
}

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
    field(int_literal.width);         \
    field(int_literal.value_i64);     \
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
case PackType_TAG: {                  \
    field(pack_type.element_type);    \
    field(pack_type.width);           \
    break;                            \
}                                     \
case RecordType_TAG: {                \
    field(record_type.members);       \
    field(record_type.names);         \
    field(record_type.special);       \
    break;                            \
}                                     \
case FnType_TAG: {                    \
    field(fn_type.tier);              \
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

    if (is_nominal(node)) {
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
    if (is_nominal((*pa))) {
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
