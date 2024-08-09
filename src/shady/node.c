#include "type.h"
#include "log.h"
#include "ir_private.h"
#include "portability.h"

#include "dict.h"

#include <string.h>
#include <assert.h>

String get_value_name_unsafe(const Node* v) {
    assert(v && is_value(v));
    if (v->tag == Param_TAG)
        return v->payload.param.name;
    return NULL;
}

String get_value_name_safe(const Node* v) {
    String name = get_value_name_unsafe(v);
    if (name && strlen(name) > 0)
        return name;
    //if (v->tag == Variable_TAG)
    return format_string_interned(v->arena, "%%%d", v->id);
    //return node_tags[v->tag];
}

void set_value_name(const Node* var, String name) {
    // TODO: annotations
    // if (var->tag == Variablez_TAG)
    //     var->payload.varz.name = string(var->arena, name);
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

static bool is_zero(const Node* node) {
    const IntLiteral* lit = resolve_to_int_literal(node);
    if (lit && get_int_literal_value(*lit, false) == 0)
        return true;
    return false;
}

const Node* chase_ptr_to_source(const Node* ptr, NodeResolveConfig config) {
    while (true) {
        ptr = resolve_node_to_definition(ptr, config);
        switch (ptr->tag) {
            case Lea_TAG: {
                Lea lea = ptr->payload.lea;
                if (!is_zero(lea.offset))
                    goto outer_break;
                for (size_t i = 0; i < lea.indices.count; i++) {
                    if (!is_zero(lea.indices.nodes[i]))
                        goto outer_break;
                }
                ptr = lea.ptr;
                continue;
                outer_break:
                break;
            }
            case PrimOp_TAG: {
                switch (ptr->payload.prim_op.op) {
                    case convert_op: {
                        // chase generic pointers to their source
                        if (first(ptr->payload.prim_op.type_arguments)->tag == PtrType_TAG) {
                            ptr = first(ptr->payload.prim_op.operands);
                            continue;
                        }
                        break;
                    }
                    case reinterpret_op: {
                        // chase ptr casts to their source
                        // TODO: figure out round-trips through integer casts?
                        if (first(ptr->payload.prim_op.type_arguments)->tag == PtrType_TAG) {
                            ptr = first(ptr->payload.prim_op.operands);
                            continue;
                        }
                        break;
                    }
                    default: break;
                }
                break;
            }
            default: break;
        }
        break;
    }
    return ptr;
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
                node = node->payload.constant.value;
                continue;
            case RefDecl_TAG:
                node = node->payload.ref_decl.decl;
                continue;
            case Load_TAG: {
                if (config.enter_loads) {
                    const Node* source = node->payload.load.ptr;
                    const Node* result = resolve_ptr_to_value(source, config);
                    if (!result)
                        break;
                    node = result;
                    continue;
                }
            }
            case PrimOp_TAG: {
                switch (node->payload.prim_op.op) {
                    case convert_op:
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

const char* get_string_literal(IrArena* arena, const Node* node) {
    if (!node)
        return NULL;
    if (node->type && get_unqualified_type(node->type)->tag == PtrType_TAG) {
        NodeResolveConfig nrc = default_node_resolve_config();
        const Node* ptr = chase_ptr_to_source(node, nrc);
        const Node* value = resolve_ptr_to_value(ptr, nrc);
        if (value)
            return get_string_literal(arena, value);
    }
    switch (node->tag) {
        case Declaration_GlobalVariable_TAG: {
            const Node* init = node->payload.global_variable.init;
            if (init) {
                return get_string_literal(arena, init);
            }
            break;
        }
        case Declaration_Constant_TAG: {
            return get_string_literal(arena, node->payload.constant.value);
        }
        case RefDecl_TAG: {
            const Node* decl = node->payload.ref_decl.decl;
            return get_string_literal(arena, decl);
        }
        case Lea_TAG: {
            Lea lea = node->payload.lea;
            if (lea.indices.count == 3 && is_zero(lea.offset) && is_zero(first(lea.indices))) {
                const Node* ref = lea.ptr;
                if (ref->tag != RefDecl_TAG)
                    return NULL;
                const Node* decl = ref->payload.ref_decl.decl;
                if (decl->tag != GlobalVariable_TAG || !decl->payload.global_variable.init)
                    return NULL;
                return get_string_literal(arena, decl->payload.global_variable.init);
            }
            break;
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
        default: break;
    }
    return NULL;
}

const Node* get_abstraction_mem(const Node* abs) {
    return abs_mem(abs->arena, (AbsMem) { .abs = abs });
}

String get_abstraction_name(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.name;
        case BasicBlock_TAG: return abs->payload.basic_block.name;
        default: assert(false);
    }
}

String get_abstraction_name_unsafe(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.name;
        case BasicBlock_TAG: return abs->payload.basic_block.name;
        default: assert(false);
    }
}

String get_abstraction_name_safe(const Node* abs) {
    String name = get_abstraction_name_unsafe(abs);
    if (name)
        return name;
    return format_string_interned(abs->arena, "%%%d", abs->id);
}

void set_abstraction_body(Node* abs, const Node* body) {
    assert(is_abstraction(abs));
    assert(!body || is_terminator(body));
    IrArena* a = abs->arena;
    switch (abs->tag) {
        case Function_TAG: abs->payload.fun.body = body; break;
        case BasicBlock_TAG: {
            while (true) {
                const Node* mem0 = get_original_mem(get_terminator_mem(body));
                assert(mem0->tag == AbsMem_TAG);
                const Node* mem_abs = mem0->payload.abs_mem.abs;
                if (is_basic_block(mem_abs)) {
                    BodyBuilder* insert = mem_abs->payload.basic_block.insert;
                    if (insert) {
                        const Node* mem = insert->mem0;
                        set_abstraction_body((Node*) mem_abs, finish_body(insert, body));
                        body = jump_helper(a, mem_abs, empty(a), mem);
                        continue;
                    }
                    assert(mem_abs == abs);
                }
                break;
            }

            abs->payload.basic_block.body = body;
            break;
        }
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
