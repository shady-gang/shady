#include "type.h"
#include "log.h"
#include "ir_private.h"
#include "portability.h"

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

DeclTag is_declaration(const Node* node) {
    switch (node->tag) {
#define IS_DECL(_, _2, _3, name, _4) case name##_TAG: return Decl_##name##_TAG;
        DECL_NODES(IS_DECL)
#undef IS_DECL
        default: return NotADecl;
    }
}

const char* node_tags[] = {
    "invalid",
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
    false,
#define NODE_HAS_PAYLOAD(_, _2, has_payload, _4, _5) has_payload,
NODES(NODE_HAS_PAYLOAD)
#undef NODE_HAS_PAYLOAD
};

String get_decl_name(const Node* node) {
    switch (node->tag) {
        case Constant_TAG: return node->payload.constant.name;
        case Function_TAG: return node->payload.fun.name;
        case GlobalVariable_TAG: return node->payload.global_variable.name;
        case NominalType_TAG: return node->payload.nom_type.name;
        default: error("Not a decl !");
    }
}

int64_t get_int_literal_value(const Node* node, bool sign_extend) {
    const IntLiteral* literal = resolve_to_literal(node);
    if (sign_extend) {
        switch (literal->width) {
            case IntTy8:  return (int64_t) literal->value.i8;
            case IntTy16: return (int64_t) literal->value.i16;
            case IntTy32: return (int64_t) literal->value.i32;
            case IntTy64: return           literal->value.i64;
            default: assert(false);
        }
    } else {
        switch (literal->width) {
            case IntTy8:  return (int64_t) ((uint64_t) (literal->value.u8 ));
            case IntTy16: return (int64_t) ((uint64_t) (literal->value.u16));
            case IntTy32: return (int64_t) ((uint64_t) (literal->value.u32));
            case IntTy64: return                        literal->value.i64  ;
            default: assert(false);
        }
    }
}

const IntLiteral* resolve_to_literal(const Node* node) {
    while (true) {
        switch (node->tag) {
            case Constant_TAG:   return resolve_to_literal(node->payload.constant.value);
            case RefDecl_TAG:    return resolve_to_literal(node->payload.ref_decl.decl);
            case IntLiteral_TAG: return &node->payload.int_literal;
            default: return NULL;
        }
    }
}

const char* get_string_literal(IrArena* arena, const Node* node) {
    switch (node->tag) {
        case Constant_TAG:   return get_string_literal(arena, node->payload.constant.value);
        case RefDecl_TAG:    return get_string_literal(arena, node->payload.ref_decl.decl);
        case StringLiteral_TAG: return node->payload.string_lit.string;
        case Composite_TAG: {
            Nodes contents = node->payload.composite.contents;
            LARRAY(char, chars, contents.count);
            for (size_t i = 0; i < contents.count; i++) {
                const Node* value = contents.nodes[i];
                assert(value->tag == IntLiteral_TAG && value->payload.int_literal.width == IntTy8);
                chars[i] = (unsigned char) get_int_literal_value(value, false);
            }
            assert(chars[contents.count - 1] == 0);
            return string(arena, chars);
        }
        default: error("This is not a string literal and it doesn't look like one either");
    }
}

String get_abstraction_name(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.name;
        case BasicBlock_TAG: return abs->payload.basic_block.name;
        case AnonLambda_TAG: return "anonymous";
        default: assert(false);
    }
}

const Node* get_abstraction_body(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.body;
        case BasicBlock_TAG: return abs->payload.basic_block.body;
        case AnonLambda_TAG: return abs->payload.anon_lam.body;
        default: assert(false);
    }
}

Nodes get_abstraction_params(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.params;
        case BasicBlock_TAG: return abs->payload.basic_block.params;
        case AnonLambda_TAG: return abs->payload.anon_lam.params;
        default: assert(false);
    }
}

const Node* get_let_instruction(const Node* let) {
    switch (let->tag) {
        case Let_TAG: return let->payload.let.instruction;
        case LetMut_TAG: return let->payload.let_mut.instruction;
        default: assert(false);
    }
}

const Node* get_let_tail(const Node* let) {
    switch (let->tag) {
        case Let_TAG: return let->payload.let.tail;
        case LetMut_TAG: return let->payload.let_mut.tail;
        default: assert(false);
    }
}

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

    if (node_type_has_payload[node->tag]) {
        switch (node->tag) {
            #define HASH_FIELD_1(ft, t, n) payload_hash ^= hash_murmur(&payload.n, sizeof(payload.n));
            #define HASH_FIELD_0(ft, t, n)
            #define HASH_FIELD(dohash, ft, t, n) HASH_FIELD_##dohash(ft, t, n)
            #define HASH_NODE_FIELDS_1(StructName, short_name) case StructName##_TAG: { StructName payload = node->payload.short_name; StructName##_Fields(HASH_FIELD) break; }
            #define HASH_NODE_FIELDS_0(StructName, short_name)
            #define HASH_NODE_FIELDS(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) HASH_NODE_FIELDS_##has_payload(StructName, short_name)
            NODES(HASH_NODE_FIELDS)
            default: payload_hash = hash_murmur(&node->payload, sizeof(node->payload)); break;
        }
    }
    combined = tag_hash ^ payload_hash;

    end:
    return combined;
}

bool compare_node(Node** pa, Node** pb) {
    if ((*pa)->tag != (*pb)->tag) return false;
    if (is_nominal((*pa)))
        return *pa == *pb;

    const Node* a = *pa;
    const Node* b = *pb;

    #undef field
    #define field(w) eq &= memcmp(&a->payload.w, &b->payload.w, sizeof(a->payload.w)) == 0;

    if (node_type_has_payload[a->tag]) {
        bool eq = true;
        switch ((*pa)->tag) {
            #define CMP_FIELD_1(ft, t, n) eq &= memcmp(&a_payload.n, &b_payload.n, sizeof(a_payload.n)) == 0;
            #define CMP_FIELD_0(ft, t, n)
            #define CMP_FIELD(dohash, ft, t, n) CMP_FIELD_##dohash(ft, t, n)
            #define CMP_NODE_FIELDS_1(StructName, short_name) case StructName##_TAG: { StructName a_payload = a->payload.short_name; StructName b_payload = b->payload.short_name; StructName##_Fields(CMP_FIELD) break; }
            #define CMP_NODE_FIELDS_0(StructName, short_name)
            #define CMP_NODE_FIELDS(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) CMP_NODE_FIELDS_##has_payload(StructName, short_name)
            NODES(CMP_NODE_FIELDS)
            default: return memcmp(&a->payload, &b->payload, sizeof(a->payload)) == 0;
        }
        return eq;
    } else return true;
}
