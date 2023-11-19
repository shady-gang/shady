#include "type.h"
#include "log.h"
#include "ir_private.h"
#include "portability.h"

#include "dict.h"

#include <string.h>
#include <assert.h>

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
    assert(literal);
    if (sign_extend) {
        switch (literal->width) {
            case IntTy8:  return (int64_t) (int8_t)  (literal->value & 0xFF);
            case IntTy16: return (int64_t) (int16_t) (literal->value & 0xFFFF);
            case IntTy32: return (int64_t) (int32_t) (literal->value & 0xFFFFFFFF);
            case IntTy64: return (int64_t) literal->value;
            default: assert(false);
        }
    } else {
        switch (literal->width) {
            case IntTy8:  return literal->value & 0xFF;
            case IntTy16: return literal->value & 0xFFFF;
            case IntTy32: return literal->value & 0xFFFFFFFF;
            case IntTy64: return literal->value;
            default: assert(false);
        }
    }
}

const Node* get_quoted_value(const Node* instruction) {
    if (instruction->payload.prim_op.op == quote_op)
        return first(instruction->payload.prim_op.operands);
    return NULL;
}

const IntLiteral* resolve_to_literal(const Node* node) {
    if (!node)
        return NULL;
    while (true) {
        switch (node->tag) {
            case Constant_TAG:   return resolve_to_literal(get_quoted_value(node->payload.constant.instruction));
            case RefDecl_TAG:    return resolve_to_literal(node->payload.ref_decl.decl);
            case IntLiteral_TAG: return &node->payload.int_literal;
            default: return NULL;
        }
    }
}

static bool is_zero(const Node* node) {
    //node = resolve_known_vars(node, false);
    if (node->tag == IntLiteral_TAG) {
        if (get_int_literal_value(node, false) == 0)
            return true;
    }
    return false;
}

const char* get_string_literal(IrArena* arena, const Node* node) {
    switch (node->tag) {
        case Constant_TAG:   return get_string_literal(arena, get_quoted_value(node->payload.constant.instruction));
        case RefDecl_TAG:    return get_string_literal(arena, node->payload.ref_decl.decl);
        case Variable_TAG: {
            if (node->payload.var.pindex != 0)
                return NULL;
            const Node* abs = node->payload.var.abs;
            if (!abs || abs->tag != Case_TAG)
                return NULL;
            const Node* user = abs->payload.case_.structured_construct;
            if (user->tag != Let_TAG)
                return NULL;
            return get_string_literal(arena, user->payload.let.instruction);
        }
        case PrimOp_TAG: {
            switch (node->payload.prim_op.op) {
                case lea_op: {
                    Nodes ops = node->payload.prim_op.operands;
                    if (ops.count == 3 && is_zero(ops.nodes[1]) && is_zero(ops.nodes[2])) {
                        const Node* ref = first(ops);
                        if (ref->tag != RefDecl_TAG)
                            return NULL;
                        const Node* decl = ref->payload.ref_decl.decl;
                        if (decl->tag != GlobalVariable_TAG || !decl->payload.global_variable.init)
                            return NULL;
                        return get_string_literal(arena, decl->payload.global_variable.init);
                    }
                    break;
                }
                default: break;
            }
            return NULL;
        }
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
        default: return NULL; // error("This is not a string literal and it doesn't look like one either");
    }
}

bool is_abstraction(const Node* node) {
    NodeTag tag = node->tag;
    return tag == Function_TAG || tag == BasicBlock_TAG || tag == Case_TAG;
}

String get_abstraction_name(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.name;
        case BasicBlock_TAG: return abs->payload.basic_block.name;
        case Case_TAG: return "case";
        default: assert(false);
    }
}

const Node* get_abstraction_body(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.body;
        case BasicBlock_TAG: return abs->payload.basic_block.body;
        case Case_TAG: return abs->payload.case_.body;
        default: assert(false);
    }
}

void set_abstraction_body(Node* abs, const Node* body) {
    assert(is_abstraction(abs));
    assert(!body || is_terminator(body));
    switch (abs->tag) {
        case Function_TAG: abs->payload.fun.body = body; break;
        case BasicBlock_TAG: abs->payload.basic_block.body = body; break;
        case Case_TAG: abs->payload.case_.body = body; break;
        default: assert(false);
    }
}

Nodes get_abstraction_params(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.params;
        case BasicBlock_TAG: return abs->payload.basic_block.params;
        case Case_TAG: return abs->payload.case_.params;
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

KeyHash hash_node_payload(const Node* node);

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
        payload_hash = hash_node_payload(node);
    }
    combined = tag_hash ^ payload_hash;

    end:
    return combined;
}

bool compare_node_payload(const Node*, const Node*);

bool compare_node(Node** pa, Node** pb) {
    if ((*pa)->tag != (*pb)->tag) return false;
    if (is_nominal((*pa)))
        return *pa == *pb;

    const Node* a = *pa;
    const Node* b = *pb;

    #undef field
    #define field(w) eq &= memcmp(&a->payload.w, &b->payload.w, sizeof(a->payload.w)) == 0;

    if (node_type_has_payload[a->tag]) {
        return compare_node_payload(a, b);
    } else return true;
}

#include "node_generated.c"
