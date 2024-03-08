#include "type.h"
#include "log.h"
#include "ir_private.h"
#include "portability.h"

#include "dict.h"

#include <string.h>
#include <assert.h>

String get_value_name(const Node* v) {
    assert(v && is_value(v));
    if (v->tag == Variable_TAG)
        return v->payload.var.name;
    return NULL;
}

String get_value_name_safe(const Node* v) {
    String name = get_value_name(v);
    if (name)
        return name;
    if (v->tag == Variable_TAG)
        return format_string_interned(v->arena, "v%d", v->id);
    return node_tags[v->tag];
}

void set_variable_name(Node* var, String name) {
    assert(var->tag == Variable_TAG);
    var->payload.var.name = string(var->arena, name);
}

int64_t get_int_literal_value(IntLiteral literal, bool sign_extend) {
    if (sign_extend) {
        switch (literal.width) {
            case IntTy8:  return (int64_t) (int8_t)  (literal.value & 0xFF);
            case IntTy16: return (int64_t) (int16_t) (literal.value & 0xFFFF);
            case IntTy32: return (int64_t) (int32_t) (literal.value & 0xFFFFFFFF);
            case IntTy64: return (int64_t) literal.value;
            default: assert(false);
        }
    } else {
        switch (literal.width) {
            case IntTy8:  return literal.value & 0xFF;
            case IntTy16: return literal.value & 0xFFFF;
            case IntTy32: return literal.value & 0xFFFFFFFF;
            case IntTy64: return literal.value;
            default: assert(false);
        }
    }
}

static_assert(sizeof(float) == sizeof(uint64_t) / 2, "floats aren't the size we expect");
double get_float_literal_value(FloatLiteral literal) {
    double r;
    switch (literal.width) {
        case FloatTy16:
            error_print("TODO: fp16 literals");
            error_die();
            SHADY_UNREACHABLE;
            break;
        case FloatTy32: {
            float f;
            memcpy(&f, &literal.value, sizeof(float));
            r = (double) f;
            break;
        }
        case FloatTy64:
            memcpy(&r, &literal.value, sizeof(double));
            break;
    }
    return r;
}

const Node* get_quoted_value(const Node* instruction) {
    if (instruction->payload.prim_op.op == quote_op)
        return first(instruction->payload.prim_op.operands);
    return NULL;
}

const Node* resolve_ptr_to_value(const Node* ptr, NodeResolveConfig config) {
    while (ptr) {
        ptr = resolve_node_to_definition(ptr, config);
        switch (ptr->tag) {
            case PrimOp_TAG: {
                switch (ptr->payload.prim_op.op) {
                    case convert_op: { // allow address space conversions
                        ptr = first(ptr->payload.prim_op.operands);
                        continue;
                    }
                    default: break;
                }
            }
            case GlobalVariable_TAG:
                if (config.assume_globals_immutability)
                    return ptr->payload.global_variable.init;
                break;
            default: break;
        }
        ptr = NULL;
    }
    return NULL;
}

NodeResolveConfig default_node_resolve_config() {
    return (NodeResolveConfig) {
        .enter_loads = true,
        .allow_incompatible_types = false,
        .assume_globals_immutability = false,
    };
}

const Node* resolve_node_to_definition(const Node* node, NodeResolveConfig config) {
    while (node) {
        switch (node->tag) {
            case Constant_TAG:
                node = node->payload.constant.instruction;
                continue;
            case RefDecl_TAG:
                node = node->payload.ref_decl.decl;
                continue;
            case PrimOp_TAG: {
                switch (node->payload.prim_op.op) {
                    case quote_op: {
                        node = first(node->payload.prim_op.operands);;
                        continue;
                    }
                    case load_op: {
                        if (config.enter_loads) {
                            const Node* source = first(node->payload.prim_op.operands);
                            const Node* result = resolve_ptr_to_value(source, config);
                            if (!result)
                                break;
                            node = result;
                            continue;
                        }
                    }
                    case reinterpret_op: {
                        if (config.allow_incompatible_types) {
                            node = first(node->payload.prim_op.operands);
                            continue;
                        }
                    }
                    default: break;
                }
                break;
            }
            default: break;
        }
        break;
    }
    return node;
}

const IntLiteral* resolve_to_int_literal(const Node* node) {
    node = resolve_node_to_definition(node, default_node_resolve_config());
    if (!node)
        return NULL;
    if (node->tag == IntLiteral_TAG)
        return &node->payload.int_literal;
    return NULL;
}

const FloatLiteral* resolve_to_float_literal(const Node* node) {
    node = resolve_node_to_definition(node, default_node_resolve_config());
    if (!node)
        return NULL;
    if (node->tag == FloatLiteral_TAG)
        return &node->payload.float_literal;
    return NULL;
}

static bool is_zero(const Node* node) {
    const IntLiteral* lit = resolve_to_int_literal(node);
    if (lit && get_int_literal_value(*lit, false) == 0)
        return true;
    return false;
}

const char* get_string_literal(IrArena* arena, const Node* node) {
    if (!node)
        return NULL;
    switch (node->tag) {
        case RefDecl_TAG: {
            const Node* decl = node->payload.ref_decl.decl;
            switch (is_declaration(decl)) {
                case Declaration_GlobalVariable_TAG: {
                    const Node* init = decl->payload.global_variable.init;
                    if (init)
                        return get_string_literal(arena, init);
                    break;
                }
                default:
                    break;
            }
            return NULL;
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
                chars[i] = (unsigned char) get_int_literal_value(*resolve_to_int_literal(value), false);
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

const Node* get_insert_helper_end(InsertHelper h) {
    const Node* terminator = h.body;
    while (true) {
        if (is_structured_construct(terminator)) {
            terminator = get_structured_construct_tail(terminator);
            continue;
        } else if (terminator->tag == Body_TAG) {
            terminator = terminator->payload.body.terminator;
            continue;
        } else if (terminator->tag == InsertHelperEnd_TAG) {
            return terminator;
        }
        error("Invalid syntax: InsertHelper chain should end with InsertHelperEnd.")
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
